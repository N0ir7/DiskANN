#include <mutex>
#include <numeric>
#include <random>
#include <omp.h>
#include <cstring>
#include <ctime>
#include <timer.h>
#include <iomanip>
#include <atomic>
#include <future>
#include <dirent.h>
#include <filesystem>
#include "aux_utils.h"
#include "utils.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "lsm/lsm_index.h"
#include "lsm/options.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <pthread.h>
#include <sched.h>

#define NUM_INSERT_THREADS 2
#define NUM_DELETE_THREADS 1
#define NUM_SEARCH_THREADS 6

// random number generator
std::random_device dev;
std::mt19937       rng(dev());

tsl::robin_map<std::string, uint32_t> params;
float                                 mem_alpha, merge_alpha;
std::atomic_bool      _insertions_done(true);
std::atomic_bool      _del_done(true);
std::atomic_bool      _redistribute_done(false);
std::atomic_bool      _merge_mem_start(false);
std::atomic_bool      _merge_mem_done(true);
std::atomic_bool      _merge_level0_done(true);
std::atomic_bool      _merge_level0_start(false);
std::vector<uint32_t> Lvec;
std::future<void>     delete_future;
std::future<void>     insert_future;
std::future<void>     merge_mem_future;
std::future<void>     merge_disk_future;
diskann::Timer        global_timer;
std::string           all_points_file = "";
bool                  save_index_as_one_file;
bool                  include_delete;
std::string           query_file = "";
std::string           truthset_file = "";
std::string           log_prefix = "";
std::string           index_dir = "";
unsigned *gt_ids = nullptr;
uint32_t *gt_tags = nullptr;
float    *gt_dists = nullptr;
size_t    gt_num, gt_dim;

void ShowMemoryStatus(int iter) {
  int current_time = global_timer.elapsed() / 1000000;

  int           tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                      1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb /1024;

  std::cout << "memory current time: " << current_time << " RSS : " << rss
            << " MB" << std::endl;
  const char*    dir = ::index_dir.c_str();
  DIR*           dp;
  struct dirent* entry;
  struct stat    statbuf;
  long           dir_size = 0;

  if ((dp = opendir(dir)) == NULL) {
    fprintf(stderr, "Cannot open dir: %s\n", dir);
    exit(0);
  }

  chdir(dir);

  while ((entry = readdir(dp)) != NULL) {
    lstat(entry->d_name, &statbuf);
    dir_size += statbuf.st_size;
  }
  chdir("..");
  closedir(dp);
  dir_size /= (1024 * 1024);
  std::cout << "disk usage : " << dir_size << " MB" << std::endl;
  /**
   * Output file
  */
  std::string log_file_path = ::log_prefix + "_storage_use.csv";
  bool is_new_file = !file_exists(log_file_path);
  std::ofstream log_file(log_file_path, std::ios::app);
  if (log_file.is_open()) {
      // 如果是新文件，则写入表头
      if (is_new_file) {
          log_file << "iter,period,mem_use(MB),disk_use(MB)" << std::endl;
      }
      // 追加数据
      log_file << iter << ","<< current_time << "," << rss << "," << dir_size << std::endl;
      log_file.close();
  } else {
      std::cerr << "Failed to open storage_use log file!" << std::endl;
  }
}

template<typename T, typename TagT = uint32_t>
void seed_insert_iter(tsl::robin_set<uint32_t> &active_set,
               tsl::robin_set<uint32_t> &inactive_set,
               const std::string        &inserted_points_file,
               const std::string        &inserted_tags_file) {
  
  const uint32_t insert_count = params[std::string("insert_count")];
  const uint32_t ndims = params[std::string("ndims")];

  std::cout << "ITER: start = " << active_set.size() << ", "
            << inactive_set.size() << "\n";

  // pick `insert_count` tags
  std::vector<uint32_t> inactive_vec(inactive_set.begin(), inactive_set.end());
  std::shuffle(inactive_vec.begin(), inactive_vec.end(), rng);
  std::vector<uint32_t> insert_vec;
  if (inactive_vec.size() < insert_count)
    insert_vec.insert(insert_vec.end(), inactive_vec.begin(),
                      inactive_vec.end());
  else
    insert_vec.insert(insert_vec.end(), inactive_vec.begin(),
                      inactive_vec.begin() + insert_count);
  inactive_set.clear();
  inactive_set.insert(inactive_vec.begin() + insert_vec.size(),
                      inactive_vec.end());
  std::cout << "ITER: INSERT - " << insert_vec.size() << " IDs in "
            << inserted_tags_file << "\n";
  std::sort(insert_vec.begin(), insert_vec.end());
  TagT *tag_data = new TagT[insert_vec.size()];
  for (size_t i = 0; i < insert_vec.size(); i++)
    tag_data[i] = insert_vec[i];
  diskann::save_bin<TagT>(inserted_tags_file, tag_data, insert_vec.size(), 1);
  delete[] tag_data;

  // use ifstream reader to load node coordinates
  std::ifstream base_reader;
  base_reader.open(::all_points_file, std::ios::binary | std::ios::ate);

  base_reader.seekg(2 * sizeof(uint32_t), std::ios::beg);

  std::ofstream inserted_points_writer(inserted_points_file, std::ios::binary);
  T *new_pts = new T[(uint32_t) insert_vec.size() * (uint32_t) ndims];
  for (uint64_t idx = 0; idx < insert_vec.size(); idx++) {
    uint32_t actual_idx = insert_vec[idx];
    T       *point = new T[ndims];
    base_reader.seekg(
        (2 * sizeof(uint32_t) + actual_idx * (uint64_t) ndims * sizeof(T)),
        std::ios::beg);
    base_reader.read((char *) point, ((uint64_t) ndims) * sizeof(T));
    T *dest_ptr = new_pts + idx * (uint64_t) ndims;
    std::memcpy(dest_ptr, point, ndims * sizeof(T));
    delete[] point;
  }
  base_reader.close();

  uint32_t npts_u32 = (uint32_t) insert_vec.size();
  uint32_t ndims_u32 = ndims;
  inserted_points_writer.write((char *) &npts_u32, sizeof(uint32_t));
  inserted_points_writer.write((char *) &ndims_u32, sizeof(uint32_t));
  inserted_points_writer.write(
      (char *) new_pts,
      (uint64_t) insert_vec.size() * (uint64_t) ndims * sizeof(T));
  inserted_points_writer.close();
  delete[] new_pts;

  // balance tags
  active_set.insert(insert_vec.begin(), insert_vec.end());

  // diskann::cout << "ITER: end = " << active_set.size() << ", "
  //               << inactive_set.size() << "\n";
#ifndef _WINDOWS
  std::cout << "ITER: end = " << active_set.size() << ", "
            << inactive_set.size() << "\n";
  // malloc_stats();
#endif
}
/**
 * 用于在每次迭代中进行数据集的动态管理，包括从活跃集合中选取要删除的元素，
 * 从非活跃集合中选取要插入的元素，并将这些数据保存在指定的文件中。
 * 通过该函数，保证了在每次迭代中，活跃数据集和非活跃数据集可以进行平衡和更新
 */
template<typename T, typename TagT = uint32_t>
void seed_iter(tsl::robin_set<uint32_t> &active_set,
               tsl::robin_set<uint32_t> &inactive_set,
               const std::string        &inserted_points_file,
               const std::string        &inserted_tags_file,
               tsl::robin_set<TagT>     &deleted_tags) {
  
  const uint32_t insert_count = params[std::string("insert_count")];
  const uint32_t delete_count = params[std::string("delete_count")];
  const uint32_t ndims = params[std::string("ndims")];

  std::cout << "ITER: start = " << active_set.size() << ", "
            << inactive_set.size() << "\n";

  // pick `delete_count` tags
  std::vector<uint32_t> active_vec(active_set.begin(), active_set.end());
  std::shuffle(active_vec.begin(), active_vec.end(), rng);
  std::vector<uint32_t> delete_vec;
  if (active_vec.size() < delete_count)
    delete_vec.insert(delete_vec.end(), active_vec.begin(), active_vec.end());
  else
    delete_vec.insert(delete_vec.end(), active_vec.begin(),
                      active_vec.begin() + delete_count);
  for (auto iter : delete_vec)
    deleted_tags.insert(iter);
  active_set.clear();
  active_set.insert(active_vec.begin() + delete_vec.size(), active_vec.end());
  std::cout << "ITER: DELETE - " << delete_vec.size() << " IDs\n";

  // pick `insert_count` tags
  std::vector<uint32_t> inactive_vec(inactive_set.begin(), inactive_set.end());
  std::shuffle(inactive_vec.begin(), inactive_vec.end(), rng);
  std::vector<uint32_t> insert_vec;
  if (inactive_vec.size() < insert_count)
    insert_vec.insert(insert_vec.end(), inactive_vec.begin(),
                      inactive_vec.end());
  else
    insert_vec.insert(insert_vec.end(), inactive_vec.begin(),
                      inactive_vec.begin() + insert_count);
  inactive_set.clear();
  inactive_set.insert(inactive_vec.begin() + insert_vec.size(),
                      inactive_vec.end());
  std::cout << "ITER: INSERT - " << insert_vec.size() << " IDs in "
            << inserted_tags_file << "\n";
  std::sort(insert_vec.begin(), insert_vec.end());
  TagT *tag_data = new TagT[insert_vec.size()];
  for (size_t i = 0; i < insert_vec.size(); i++)
    tag_data[i] = insert_vec[i];
  diskann::save_bin<TagT>(inserted_tags_file, tag_data, insert_vec.size(), 1);
  delete[] tag_data;

  // use ifstream reader to load node coordinates
  std::ifstream base_reader;
  base_reader.open(::all_points_file, std::ios::binary | std::ios::ate);

  base_reader.seekg(2 * sizeof(uint32_t), std::ios::beg);

  std::ofstream inserted_points_writer(inserted_points_file, std::ios::binary);
  T *new_pts = new T[(uint32_t) insert_vec.size() * (uint32_t) ndims];
  for (uint64_t idx = 0; idx < insert_vec.size(); idx++) {
    uint32_t actual_idx = insert_vec[idx];
    T       *point = new T[ndims];
    base_reader.seekg(
        (2 * sizeof(uint32_t) + actual_idx * (uint64_t) ndims * sizeof(T)),
        std::ios::beg);
    base_reader.read((char *) point, ((uint64_t) ndims) * sizeof(T));
    T *dest_ptr = new_pts + idx * (uint64_t) ndims;
    std::memcpy(dest_ptr, point, ndims * sizeof(T));
    delete[] point;
  }
  base_reader.close();

  uint32_t npts_u32 = (uint32_t) insert_vec.size();
  uint32_t ndims_u32 = ndims;
  inserted_points_writer.write((char *) &npts_u32, sizeof(uint32_t));
  inserted_points_writer.write((char *) &ndims_u32, sizeof(uint32_t));
  inserted_points_writer.write(
      (char *) new_pts,
      (uint64_t) insert_vec.size() * (uint64_t) ndims * sizeof(T));
  inserted_points_writer.close();
  delete[] new_pts;

  // balance tags
  inactive_set.insert(delete_vec.begin(), delete_vec.end());
  active_set.insert(insert_vec.begin(), insert_vec.end());

#ifndef _WINDOWS
  std::cout << "ITER: end = " << active_set.size() << ", "
            << inactive_set.size() << "\n";
#endif
}

template<typename T, typename TagT = uint32_t>
void search_kernel(int iter, lsmidx::LSMVectorIndex<T, TagT> &lsm_index,
                   const tsl::robin_set<uint32_t> &active_tags,
                   std::string reason = "",
                   bool                            print_stats = false) {
  uint64_t recall_at = params[std::string("recall_k")];

  // hold data
  T        *query = nullptr;
  size_t    query_num, query_dim, query_aligned_dim;
  // load query + truthset
  std::cout << "Loading query : " << ::query_file << std::endl;
  diskann::load_aligned_bin<T>(::query_file, query, query_num, query_dim,
                               query_aligned_dim);

  // prep for search
  std::vector<uint32_t> query_result_ids;
  std::vector<uint32_t> query_result_tags;
  std::vector<float>    query_result_dists;
  query_result_ids.resize(recall_at * query_num);
  query_result_dists.resize(recall_at * query_num);
  query_result_tags.resize(recall_at * query_num);
  std::vector<uint32_t> query_result_ids_32(recall_at * query_num);

  for (size_t test_id = 0; test_id < ::Lvec.size(); test_id++) {
    diskann::QueryStats *stats = new diskann::QueryStats[query_num];
    uint32_t             L = Lvec[test_id];
    std::vector<double>  latency_stats(query_num, 0);
    auto                 s = std::chrono::high_resolution_clock::now();
    auto                 start = ::global_timer.elapsed() / 1000000;
    omp_set_max_active_levels(4);
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      lsmidx::SearchOptions sopts; 
      sopts.K = recall_at;
      sopts.search_L = L;
      sopts.beamwidth = params[std::string("beam_width")];
      lsmidx::VecSlice<T> vec(query + (i * query_aligned_dim), query_aligned_dim);

      auto qs = std::chrono::high_resolution_clock::now();
      lsm_index.Search(sopts, vec,
                               (query_result_tags.data() + (i * recall_at)),
                               query_result_dists.data() + (i * recall_at),
                               stats + i);
      auto qe = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000;
      //      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    auto                          end = ::global_timer.elapsed() / 1000000;
    std::chrono::duration<double> diff = e - s;
    float qps = (float) (((double) query_num) / diff.count());
    // compute mean recall, IOs
    float mean_recall = 0.0f;
    mean_recall = diskann::calculate_recall(
        (unsigned) query_num, ::gt_ids, ::gt_dists, (unsigned) ::gt_dim,
        query_result_tags.data(), (unsigned) recall_at, (unsigned) recall_at,
        active_tags);
    float mean_ios = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats &stats) { return stats.n_ios; });
    float sum_skip = (float) diskann::get_sum_stats(
        stats, query_num,
        [](const diskann::QueryStats &stats) { return stats.n_skip_level0_num; });
    float sum_level0_num = (float) diskann::get_sum_stats(
        stats, query_num,
        [](const diskann::QueryStats &stats) { return stats.level0_num; });
    std::sort(latency_stats.begin(), latency_stats.end());
    /**
     * Output log
    */
    std::string recall_string = "Recall@" + std::to_string(recall_at);
    std::cout << std::setw(6) << "Iter"<< std::setw(14) << "Reason" << std::setw(14) << "period start" << std::setw(14) << "period end" << std::setw(4) << "Ls"
              << std::setw(12) << "QPS " << std::setw(18) << "Mean Latency (ms)"
              << std::setw(12) << "90 Latency" << std::setw(12) << "95 Latency"
              << std::setw(12) << "99 Latency" << std::setw(12)
              << "99.9 Latency" << std::setw(12) << recall_string
              << std::setw(12) << "Mean disk IOs" << std::setw(18) << "sum level0_skip" << std::setw(18) << "sum level0_num" << std::endl;
    std::cout
        << "=============================search=================================="
           "==============="
        << std::endl;
    std::cout << std::setw(6) << iter << std::setw(14) << reason << std::setw(14) << start << std::setw(14) << end << std::setw(4) << L
              << std::setw(12) << qps << std::setw(18)
              << (std::accumulate(latency_stats.begin(),
                                          latency_stats.end(), 0.0)) /
                     (float) query_num
              << std::setw(12)
              << (float) latency_stats[(_u64) (0.90 * ((double) query_num))]
              << std::setw(12)
              << (float) latency_stats[(_u64) (0.95 * ((double) query_num))]
              << std::setw(12)
              << (float) latency_stats[(_u64) (0.99 * ((double) query_num))]
              << std::setw(12)
              << (float) latency_stats[(_u64) (0.999 * ((double) query_num))]
              << std::setw(12) << mean_recall << std::setw(12) << mean_ios
              << std::setw(18) << sum_skip
              << std::setw(18) << sum_level0_num
              << std::endl;
    if(mean_recall<95 && reason == "before insert"){
      diskann::cout<<"low recall,stop!"<<std::endl;
    }
    /**
     * output csv
    */
    std::string log_file_path = ::log_prefix + "_search.csv";
    bool is_new_file = !file_exists(log_file_path);
    std::ofstream log_file(log_file_path, std::ios::app);
    if (log_file.is_open()) {
        // 如果是新文件，则写入表头
        if (is_new_file) {
            log_file << "iter,reason,period start,period end,Ls,QPS,Mean Latency (ms),90 Latency,"
                        "95 Latency,99 Latency,99.9 Latency," << recall_string
                    << ",Mean disk IOs,level0_skip,level0_num" << std::endl;
        }

        // 追加数据
        log_file << iter << ","<< reason << ","<< start << "," << end << "," << L << "," << qps << ","
                << (std::accumulate(latency_stats.begin(),
                                            latency_stats.end(), 0.0)) /
                        (float) query_num
                << ","
                << (float) latency_stats[(_u64) (0.90 * ((double) query_num))]
                << ","
                << (float) latency_stats[(_u64) (0.95 * ((double) query_num))]
                << ","
                << (float) latency_stats[(_u64) (0.99 * ((double) query_num))]
                << ","
                << (float) latency_stats[(_u64) (0.999 * ((double) query_num))]
                << "," << mean_recall << "," << mean_ios << "," << sum_skip << "," << sum_level0_num << std::endl;

        log_file.close();
    } else {
        std::cerr << "Failed to open search log file!" << std::endl;
    }
    delete[] stats;
  }
  diskann::aligned_free(query);
  // delete[] gt_ids;
  // delete[] gt_dists;
  // delete[] gt_tags;
}

template<typename T, typename TagT = uint32_t>
void merge_mem_kernel(lsmidx::LSMVectorIndex<T, TagT> &lsm_index) {
  while(::_merge_mem_start.load()){
    bool expected = true;
    if (::_merge_mem_done.compare_exchange_strong(expected, false)) {
      auto start = ::global_timer.elapsed() / 1000000;
      lsm_index.TriggerMergeMemIndex();
      auto end = ::global_timer.elapsed() / 1000000;
      auto diff = end - start;
      /**
       * Output logs
      */
      std::cout << std::setw(14) << "period start" << std::setw(14) << "period end" << std::setw(15) << "merge mem time(s)"
                << std::endl;
      std::cout << "===============merge_mem================" << std::endl;
      std::cout << std::setw(14) << start << std::setw(14) << end << std::setw(15) << diff
                << std::endl;
      /**
       * Output csv
      */
      std::string merge_log_file_path = log_prefix + "_merge_mem.csv";
      bool is_new_file = !file_exists(merge_log_file_path);  // 判断是否是新文件

      std::ofstream merge_log_file(merge_log_file_path, std::ios::app);
      if (merge_log_file.is_open()) {
          // 只有新文件才写入表头
          if (is_new_file) {
              merge_log_file << "period start,period end,merge mem time(s)" << std::endl;
          }

          // 追加数据
          merge_log_file << start << "," << end << "," << diff << std::endl;

          merge_log_file.close();
      } else {
          std::cerr << "Failed to open merge_mem log file!" << std::endl;
      }
      ::_merge_mem_done.store(true);
    }else{
      std::cout << "_merge_mem_done is already true" << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }
}
template<typename T, typename TagT = uint32_t>
void merge_disk_kernel(lsmidx::LSMVectorIndex<T, TagT> &lsm_index) {
  while(::_merge_level0_start.load()){
    bool expected = true;
    if(::_merge_level0_done.compare_exchange_strong(expected, false)){
      diskann::MergeStats stats;
      auto start = ::global_timer.elapsed() / 1000000;
      lsm_index.TriggerMergeDiskIndex(0, &stats);
      auto end = ::global_timer.elapsed() / 1000000;
      auto diff = end - start;
      if(stats.delete_phase_random_read_4k != 0 && stats.delete_phase_seq_read_4k != 0){
        float thresh = stats.delete_phase_seq_read_4k * lsmidx::config::redistribute_factor;
        if(stats.delete_phase_random_read_4k >= thresh){
          ::_redistribute_done.store(false);
        }
      }
      /**
       * output log
      */
      std::cout << std::setw(14) << "period start" << std::setw(14) << "period end" << std::setw(15) << "merge disk time(s)"
                << std::endl;
      std::cout << "===============merge_level0================" << std::endl;
      std::cout << std::setw(14) << start << std::setw(14) << end << std::setw(15) << diff
              << std::endl;
      /**
       * Output csv
      */
      std::string merge_log_file_path = log_prefix + "_merge_disk.csv";
      bool is_new_file = !file_exists(merge_log_file_path);  // 判断是否是新文件

      std::ofstream merge_log_file(merge_log_file_path, std::ios::app);
      if (merge_log_file.is_open()) {
          // 只有新文件才写入表头
          if (is_new_file) {
            merge_log_file
              << "period start,period end,merge disk time(s),"
              << "delete_phase_time,delete_phase_io_time,delete_phase_random_read_4k,"
              << "delete_phase_seq_read_4k,delete_phase_random_write_4k,delete_phase_seq_write_4k,"
              << "insert_phase_time,insert_phase_io_time,insert_phase_random_read_4k,"
              << "insert_phase_seq_read_4k,insert_phase_random_write_4k,insert_phase_seq_write_4k,"
              << "patch_phase_time,patch_phase_io_time,patch_phase_random_read_4k,"
              << "patch_phase_seq_read_4k,patch_phase_random_write_4k,patch_phase_seq_write_4k"
              << std::endl;
          }

          // 追加数据
          merge_log_file 
            << start << "," << end << "," << diff << ","
            << stats.delete_phase_time << "," << stats.delete_phase_io_time << "," << stats.delete_phase_random_read_4k << ","
            << stats.delete_phase_seq_read_4k << "," << stats.delete_phase_random_write_4k << "," << stats.delete_phase_seq_write_4k << ","
            << stats.insert_phase_time << "," << stats.insert_phase_io_time << "," << stats.insert_phase_random_read_4k << ","
            << stats.insert_phase_seq_read_4k << "," << stats.insert_phase_random_write_4k << "," << stats.insert_phase_seq_write_4k << ","
            << stats.patch_phase_time << "," << stats.patch_phase_io_time << "," << stats.patch_phase_random_read_4k << ","
            << stats.patch_phase_seq_read_4k << "," << stats.patch_phase_random_write_4k << "," << stats.patch_phase_seq_write_4k
            << std::endl;

          merge_log_file.close();
      } else {
          std::cerr << "Failed to open merge_disk log file!" << std::endl;
      }
      ::_merge_level0_done.store(true);
    }else{
      std::cout << "_merge_level0_done is already run" << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::seconds(20));
  }
}
template<typename T, typename TagT = uint32_t>
void insertion_kernel(lsmidx::LSMVectorIndex<T, TagT> &lsm_index,
                      std::string insert_pts_file, std::string insert_tags_file) {
  if (::_insertions_done.load()) {
    std::cout << "Insertions_done is true at the beginning of insertion kernel"
              << std::endl;
    exit(-1);
  }
  T     *data_insert = nullptr;
  size_t npts, ndim, aligned_dim;
  diskann::load_aligned_bin<T>(insert_pts_file, data_insert, npts, ndim,
                               aligned_dim);
  size_t tag_num, tag_dim;
  TagT  *tag_data;
  diskann::load_bin<TagT>(insert_tags_file, tag_data, tag_num, tag_dim);
  if (tag_num != npts) {
    std::cout << "In insertion_kernel(), number of tags loaded is not equal to "
                 "number of points loaded. Exiting....."
              << std::endl;
    exit(-1);
  }
  _s64                i;
  std::vector<double> insert_latencies(npts, 0);
  diskann::InsertStats *stats = new diskann::InsertStats[npts];
  diskann::Timer      timer;
  auto                start = ::global_timer.elapsed() / 1000000;
#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (i = 0; i < (_s64) npts; i++) {
    lsmidx::WriteOptions opt;
    lsmidx::VecSlice<T> point_slice(data_insert + i * aligned_dim, aligned_dim);
    lsmidx::TagSlice<TagT> tag_slice(tag_data[i]);
    int ret = 0;
    diskann::Timer insert_timer;
    while((ret = lsm_index.Put(opt, point_slice, tag_slice,stats + i)) != 0){
      if(ret == -2){
        diskann::cout << "wait for 1 ms in insert" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        std::cout << "Point " << i << "could not be inserted." << std::endl;
        break;
      }
    }
    insert_latencies[i] = ((double) insert_timer.elapsed());

    if ((i % 1000 == 0) && (i > 0)){
      std::cout << "Inserted another 1k points" << std::endl;
    }
    
  }
  auto  diff = timer.elapsed() / 1000;
  auto  end = ::global_timer.elapsed() / 1000000;
  float qps = (float) (((double) npts) * 1000 / diff);
  std::sort(insert_latencies.begin(), insert_latencies.end());
  /**
   * Output log
  */
  std::cout << std::setw(14) << "Period start" << std::setw(14) << "Period end"
            << std::setw(11) << "insert qps"
            << std::setw(18) << "total time(ms)"
            << std::setw(10) << "total pts" << std::setw(18)
            << "mean latency(us)" << std::setw(18) << "10 latency(us)"
            << std::setw(18) << "50 latency(us)" << std::setw(18)
            << "90 latency(us)" << std::setw(18) << "95 latency(us)"
            << std::setw(18) << "99 latency(us)" << std::setw(18)
            << "99.9 latency(us)" << std::endl;
  std::cout << "==============================insertion================================="
               "==============================================================="
               "=========================="
            << std::endl;
  std::cout << std::setw(14) << start << std::setw(14) << end << std::setw(11) << qps 
            << std::setw(18) << diff << std::setw(10) << npts << std::setw(18)
            << ( std::accumulate(insert_latencies.begin(),
                                        insert_latencies.end(), 0.0)) /
                   (float) npts
            << std::setw(18)
            << insert_latencies[(size_t) (0.1 * ((double) npts))]
            << std::setw(18)
            << insert_latencies[(size_t) (0.5 * ((double) npts))]
            << std::setw(18)
            << insert_latencies[(size_t) (0.9 * ((double) npts))]
            << std::setw(18)
            << insert_latencies[(size_t) (0.95 * ((double) npts))]
            << std::setw(18)
            << insert_latencies[(size_t) (0.99 * ((double) npts))]
            << std::setw(18)
            << insert_latencies[(size_t) (0.999 * ((double) npts))]
            << std::endl;
  /**
   * Output csv
  */
  std::string insert_log_file_path = log_prefix + "_insert.csv";
  bool is_new_file = !file_exists(insert_log_file_path);  // 判断是否是新文件

  std::ofstream insert_log_file(insert_log_file_path, std::ios::app);
  if (insert_log_file.is_open()) {
      // 只有新文件才写入表头
      if (is_new_file) {
          insert_log_file << "Period start,Period end,Insert QPS,Total time(ms),total pts,Mean latency(us),"
                            "10 latency(us),50 latency(us),90 latency(us),95 latency(us),"
                            "99 latency(us),99.9 latency(us)" << std::endl;
      }

      // 计算 Mean Latency
      double mean_latency = (std::accumulate(insert_latencies.begin(), insert_latencies.end(), 0.0)) /
                          (float) npts;

      // 追加数据
      insert_log_file << start << "," << end << "," << qps << "," << diff << "," << npts << ","
                      << mean_latency << ","
                      << insert_latencies[(size_t) (0.1 * ((double) npts))] << ","
                      << insert_latencies[(size_t) (0.5 * ((double) npts))] << ","
                      << insert_latencies[(size_t) (0.9 * ((double) npts))] << ","
                      << insert_latencies[(size_t) (0.95 * ((double) npts))] << ","
                      << insert_latencies[(size_t) (0.99 * ((double) npts))] << ","
                      << insert_latencies[(size_t) (0.999 * ((double) npts))]
                      << std::endl;

      insert_log_file.close();
  } else {
      std::cerr << "Failed to open insert log file!" << std::endl;
  }
  ::_insertions_done.store(true);
  diskann::InsertStats totalStats;
  for (size_t i = 0; i < npts; i++) {
      totalStats.Aggregate(stats[i]);
  }
  std::cout << "Insert Count: " << npts << " ;InsertStats: " << totalStats.ToString() << std::endl;
  delete[] stats;
  delete[] data_insert;
  delete[] tag_data;
}
template<typename T, typename TagT = uint32_t>
void deletion_kernel(lsmidx::LSMVectorIndex<T, TagT> &lsm_index,
                     tsl::robin_set<uint32_t> del_tags) {
  if (::_del_done.load()) {
    std::cout << "_del_done is already true" << std::endl;
    exit(-1);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  diskann::Timer timer;
  auto           start = ::global_timer.elapsed() / 1000000;
  for (auto iter : del_tags) {
    lsmidx::WriteOptions opt;
    lsmidx::TagSlice<TagT> tag_slice(iter);
    lsm_index.Delete(opt, tag_slice);
  }
  auto  diff = timer.elapsed() / 1000;
  auto  end = ::global_timer.elapsed() / 1000000;
  float qps = (float) (((double) del_tags.size()) * 1000 / diff);
  /**
   * Output log
  */
  std::cout << std::setw(14) << "Period start" << std::setw(14) << "Period end" 
            << std::setw(10) << "qps"
            << std::setw(25) << "total Deletion time(ms)" << std::endl;
  std::cout << "============Deletion===================" << std::endl;
  std::cout << std::setw(14) << start << std::setw(14) << end << std::setw(10) << qps
            << std::setw(25) << diff << std::endl;
  /**
   * Output csv
  */
  std::string delete_log_file_path = log_prefix + "_delete.csv";
  bool is_new_file = !file_exists(delete_log_file_path);  // 判断是否是新文件

  std::ofstream delete_log_file(delete_log_file_path, std::ios::app);
  if (delete_log_file.is_open()) {
      // 只有新文件才写入表头
      if (is_new_file) {
          delete_log_file << "Period start,Period end,QPS,Total Deletion time(ms)" << std::endl;
      }

      // 追加数据
      delete_log_file << start << "," << end << "," << qps << "," << diff << std::endl;

      delete_log_file.close();
  } else {
      std::cerr << "Failed to open delete log file!" << std::endl;
  }
  ::_del_done.store(true);
}
namespace fs = std::filesystem;

void copyDirectory(const std::string& source, const std::string& destination) {
    try {
        // Recursively copy the directory and its contents
        fs::copy(source, destination, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        std::cout << "Directory copied successfully: " << source << " to " << destination << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error copying directory: " << e.what() << std::endl;
    }
}

void copyToIteratedDirectory(int ITER, std::string source) {
    std::string destination = source + std::to_string(ITER);

    copyDirectory(source, destination);
}
template<typename T, typename TagT = uint32_t>
void check_redistribute(lsmidx::LSMVectorIndex<T, TagT>& lsm_index){
  if(!::_redistribute_done.load()){
    while (!(::_insertions_done.load() && ::_del_done.load())){
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    // 进行redistribute
    auto start = ::global_timer.elapsed() / 1000000;
    lsm_index.RedistributeDiskIndex();
    auto end = ::global_timer.elapsed() / 1000000;
    auto diff = end - start;
    /**
     * Output logs
    */
    std::cout << std::setw(14) << "period start" << std::setw(14) << "period end" << std::setw(15) << "redistribute time(s)"
              << std::endl;
    std::cout << "===============redistribute================" << std::endl;
    std::cout << std::setw(14) << start << std::setw(14) << end << std::setw(15) << diff
              << std::endl;
    /**
     * Output csv
    */
    std::string redistribute_log_file_path = log_prefix + "_redistribute.csv";
    bool is_new_file = !file_exists(redistribute_log_file_path);  // 判断是否是新文件

    std::ofstream redistribute_log_file(redistribute_log_file_path, std::ios::app);
    if (redistribute_log_file.is_open()) {
        // 只有新文件才写入表头
        if (is_new_file) {
            redistribute_log_file << "period start,period end,redistribute time(s)" << std::endl;
        }

        // 追加数据
        redistribute_log_file << start << "," << end << "," << diff << std::endl;
        redistribute_log_file.close();
    } else {
        std::cerr << "Failed to open redistribute log file!" << std::endl;
    }
    ::_redistribute_done.store(true);
  }
}
template<typename T, typename TagT = uint32_t>
void run_merge_insert_iter(int iter,lsmidx::LSMVectorIndex<T, TagT>& lsm_index,
              std::string directory,
              tsl::robin_set<uint32_t> &active_set,
              tsl::robin_set<uint32_t> &inactive_set) {
  // files for insert
  std::string insert_prefix = directory + "/insert";
  std::string insert_pts_file = insert_prefix + ".data_orig";
  std::string insert_tags_file = insert_prefix + ".tags_orig";
  std::this_thread::sleep_for(std::chrono::seconds(1));  // 休眠1秒以确保同步
  ShowMemoryStatus(iter);
  bool expected = false;
  if(::_merge_level0_start.compare_exchange_strong(expected, true)){
    ::merge_disk_future =
        std::async(std::launch::async, merge_disk_kernel<T, TagT>, std::ref(lsm_index));
  }
  bool expected2 = false;
  if (::_merge_mem_start.compare_exchange_strong(expected2, true)) {
    ::merge_mem_future =
    std::async(std::launch::async, merge_mem_kernel<T, TagT>, std::ref(lsm_index));
  }
  // 在插入和删除操作未完成时，不断执行搜索操作
  while (!(::_insertions_done.load())) {
    ShowMemoryStatus(iter);
    // 调用 search_kernel 执行搜索操作，使用 active_set
    
    search_kernel<T>(iter, lsm_index, active_set,"while insert");

    // 每次搜索后休眠 1 秒
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  // 如果插入操作已完成，重置状态并执行后续操作
  if (::_insertions_done.load()) {
    ::_insertions_done.store(false);
    ShowMemoryStatus(iter);
    // 调用 search_kernel 执行搜索操作，使用 active_set
    search_kernel<T>(iter,lsm_index, active_set, "before insert");

    std::cout << "ITER: Seeding iteration"
              << "\n";
    // seed the iteration
    seed_insert_iter<T, TagT>(active_set, inactive_set, insert_pts_file, insert_tags_file);

    // 异步启动插入操作，调用 insertion_kernel 函数
    ::insert_future =
        std::async(std::launch::async, insertion_kernel<T>,
                   std::ref(lsm_index), insert_pts_file, insert_tags_file);
  }

  // 检查合并任务的状态
  while (!(::_merge_mem_done.load() && ::_merge_level0_done.load())) {
    ShowMemoryStatus(iter);
    // 在合并任务进行过程中，不断执行搜索操作

    search_kernel<T>(iter,lsm_index, active_set, "while merge");
    // 每次搜索后休眠1秒
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  };
}
template<typename T, typename TagT = uint32_t>
void run_iter(int iter,lsmidx::LSMVectorIndex<T, TagT>& lsm_index,
              std::string directory,
              tsl::robin_set<uint32_t> &active_set,
              tsl::robin_set<uint32_t> &inactive_set) {
  // files for insert
  std::string insert_prefix = directory + "/insert";
  std::string insert_pts_file = insert_prefix + ".data_orig";
  std::string insert_tags_file = insert_prefix + ".tags_orig";
  std::this_thread::sleep_for(std::chrono::seconds(1));  // 休眠1秒以确保同步
  ShowMemoryStatus(iter);
  bool expected = false;
  if(::_merge_level0_start.compare_exchange_strong(expected, true)){
    ::merge_disk_future =
        std::async(std::launch::async, merge_disk_kernel<T, TagT>, std::ref(lsm_index));
  }

  bool expected2 = false;
  if (::_merge_mem_start.compare_exchange_strong(expected2, true)) {
    ::merge_mem_future =
    std::async(std::launch::async, merge_mem_kernel<T, TagT>, std::ref(lsm_index));
  }
  // 在插入和删除操作未完成时，不断执行搜索操作
  while (!(::_insertions_done.load() && ::_del_done.load())) {
    // 调用 search_kernel 执行搜索操作，使用 active_set
    search_kernel<T>(iter,lsm_index, active_set, "while insert");
    // 每次搜索后休眠 1 秒
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  // 如果插入和删除操作已完成，重置状态并执行后续操作
  if (::_insertions_done.load() && ::_del_done.load()) {
    ::_insertions_done.store(false);
    ::_del_done.store(false);

    ShowMemoryStatus(iter);
    // 调用 search_kernel 执行搜索操作，使用 active_set
    search_kernel<T>(iter,lsm_index, active_set, "before insert");

    std::cout << "ITER: Seeding iteration"
              << "\n";
    // seed the iteration
    tsl::robin_set<uint32_t> deleted_tags;
    seed_iter<T, TagT>(active_set, inactive_set, insert_pts_file, insert_tags_file,
                       deleted_tags);

    // 异步启动删除操作，调用 deletion_kernel 函数
    ::delete_future = std::async(std::launch::async, deletion_kernel<T, TagT>,
                                 std::ref(lsm_index), deleted_tags);

    // 异步启动插入操作，调用 insertion_kernel 函数
    ::insert_future =
        std::async(std::launch::async, insertion_kernel<T>,
                   std::ref(lsm_index), insert_pts_file, insert_tags_file);
  }

  // 检查合并任务的状态
  while (!(::_merge_mem_done.load() && ::_merge_level0_done.load())) {
    // 非阻塞式等待合并任务完成，每次等待1毫秒
    ShowMemoryStatus(iter);
    // 在合并任务进行过程中，不断执行搜索操作
    search_kernel<T>(iter,lsm_index, active_set, "while merge");

    // 每次搜索后休眠1秒
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  };
  check_redistribute(lsm_index);
}

template<typename T, typename TagT = uint32_t>
void run_all_iters(const std::string working_dir, const std::string index_name, diskann::Distance<T> *dist_cmp) {
  // get data metadata
  uint64_t npts = 0, ndims = 0;
  diskann::get_bin_metadata(::all_points_file, npts, ndims);
  params[std::string("ndims")] = (uint32_t) ndims;
  uint32_t n_iters = params["n_iters"];
  ShowMemoryStatus(-1);
  /**
   * Build a lsm index
  */
  lsmidx::BuildOptions options;
  options.dist_metric = diskann::Metric::L2;
  options.is_single_file_index = ::save_index_as_one_file;
  options.dimension = ndims;
  diskann::Parameters& paras = options.params;
  paras.Set<unsigned>("L_mem", params[std::string("mem_l_index")]);
  paras.Set<unsigned>("R_mem", params[std::string("range")]);
  paras.Set<float>("alpha_mem", ::mem_alpha);
  paras.Set<unsigned>("L_disk", params[std::string("merge_l_index")]);
  paras.Set<unsigned>("R_disk", params[std::string("range")]);
  paras.Set<float>("alpha_disk", ::merge_alpha);
  paras.Set<unsigned>("C", params[std::string("merge_maxc")]);
  paras.Set<unsigned>("beamwidth", params[std::string("beam_width")]);
  paras.Set<unsigned>("nodes_to_cache",
                      params[std::string("disk_search_node_cache_count")]);
  paras.Set<unsigned>("num_search_threads",
                      params[std::string("disk_search_nthreads")]);

  lsmidx::LSMVectorIndex<T, TagT> lsm_index(options, working_dir, index_name, dist_cmp);
  
  // load active tags
  tsl::robin_set<uint32_t> active_tags;
  std::cout << "【 Load Active Tags 】" << std::endl;
  lsm_index.GetActiveTags(active_tags);
  std::cout << "Loaded " << active_tags.size() << " tags" << std::endl;
  size_t tag_num = active_tags.size();

  // generate inactive tags
  tsl::robin_set<uint32_t> inactive_tags;
  inactive_tags.reserve(npts - tag_num);
  for (uint32_t i = 0; i < npts; i++) {
    auto iter = active_tags.find(i);
    if (iter == active_tags.end()) {
      inactive_tags.insert(i);
    }
  }
  std::cout << "Inactive tags : " << inactive_tags.size() << std::endl;
  // remove medoid from active_set
  std::vector<TagT> medoid_vec;
  lsm_index.GetMedoid(medoid_vec);
  for(auto med: medoid_vec){
    active_tags.erase(med);
  }

  for (size_t i = 0; i < n_iters; i++) {
    std::cout << "============ITER : " << i <<"============="<< std::endl;
    if(::include_delete){
      run_iter<T>(i, lsm_index, working_dir+'/'+index_name, active_tags, inactive_tags);
    }else{
      run_merge_insert_iter<T>(i, lsm_index, working_dir+'/'+index_name ,active_tags, inactive_tags);
    }
  }
  while (!(::_insertions_done.load())) {

    // 调用 search_kernel 执行搜索操作，使用 active_set
    search_kernel<T>(n_iters,lsm_index, active_tags,"while insert");
    // 每次搜索后休眠 10 秒
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  }
  ::_merge_level0_start.store(false);
  ::_merge_mem_start.store(false);
  while (!(::_merge_mem_done.load() && ::_merge_level0_done.load())) {
    // 在合并任务进行过程中，不断执行搜索操作
    search_kernel<T>(n_iters, lsm_index, active_tags, "while merge");
    // 每次搜索后休眠10秒
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  };
}

int main(int argc, char **argv) {
  std::cout << "Entering main()" << std::endl;
  if (argc < 20) {
    std::cout << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <WORKING_FOLDER> <base_prefix> "
                 "<merge_prefix> <mem_prefix> <L_mem> <alpha_mem> <L_disk> "
                 "<alpha_disk> "
              << " <full_data_bin> <single_file[0/1]> <query_bin> <truthset>"
              << " <n_iters> <total_insert_count> <total_delete_count> <range> "
                 "<recall_k> "
                 "<search_L1> <search_L2> <search_L3> ...."
              << "\n WARNING: Other parameters set inside CPP source."
              << std::endl;
    exit(-1);
  } else {
    std::cout << "This driver file only works with uint32 type tags"
              << std::endl;
  }
  std::cout.setf(std::ios::unitbuf);

  int         arg_no = 1;
  std::string index_type = argv[arg_no++];
  std::string working_dir = argv[arg_no++];
  std::string index_name = argv[arg_no++];
  ::log_prefix = std::string(argv[arg_no++]);
  ::all_points_file = std::string(argv[arg_no++]);
  ::query_file = std::string(argv[arg_no++]);
  ::truthset_file = std::string(argv[arg_no++]);
  unsigned    L_mem = (unsigned) atoi(argv[arg_no++]);
  float       alpha_mem = (float) atof(argv[arg_no++]);
  unsigned    L_disk = (unsigned) atoi(argv[arg_no++]);
  float       alpha_disk = (float) atof(argv[arg_no++]);
  bool        single_file = atoi(argv[arg_no++]);
  int         n_iters = atoi(argv[arg_no++]);
  uint32_t    insert_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    delete_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    range = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    recall_k = (uint32_t) atoi(argv[arg_no++]);
  ::include_delete = atoi(argv[arg_no++]);

  for (int ctr = arg_no; ctr < argc; ctr++) {
    _u32 curL = std::atoi(argv[ctr]);
    if (curL >= recall_k)
      ::Lvec.push_back(curL);
  }

  std::cout << "Assigning parameters" << std::endl;
  params[std::string("n_iters")] = n_iters;
  params[std::string("insert_count")] = insert_count;
  params[std::string("delete_count")] = delete_count;
  params[std::string("range")] = range;
  params[std::string("recall_k")] = recall_k;

  // hard-coded params
  params[std::string("disk_search_node_cache_count")] = 100;
  params[std::string("disk_search_nthreads")] = 16;
  params[std::string("beam_width")] = 4;
  params[std::string("mem_l_index")] = L_mem;
  mem_alpha = alpha_mem;
  merge_alpha = alpha_disk;
  params[std::string("mem_nthreads")] = 32;
  params[std::string("merge_maxc")] = (uint32_t) (range * 2.5);
  params[std::string("merge_l_index")] = L_disk;

  ::save_index_as_one_file = single_file;
  ::index_dir = working_dir + '/' + index_name;
  // load truthset
  std::cout << "Loading truthset : " << ::truthset_file << std::endl;
  diskann::load_truthset(::truthset_file, ::gt_ids, ::gt_dists, ::gt_num, ::gt_dim,
                         &::gt_tags);
  std::cout << "Calling run_all_iters()" << std::endl;
  if (index_type == std::string("float")) {
    diskann::DistanceL2 dist_cmp;
    run_all_iters<float>(working_dir, index_name, &dist_cmp);
  } else if (index_type == std::string("uint8")) {
    diskann::DistanceL2UInt8 dist_cmp;
    run_all_iters<uint8_t>(working_dir, index_name, &dist_cmp);
  } else if (index_type == std::string("int8")) {
    diskann::DistanceL2Int8 dist_cmp;
    run_all_iters<int8_t>(working_dir, index_name, &dist_cmp);
  } else {
    std::cout << "Unsupported type : " << index_type << "\n";
  }
  delete[] ::gt_ids;
  delete[] ::gt_dists;
  delete[] ::gt_tags;
  std::cout << "Exiting\n";
  return 0;
}