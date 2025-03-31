// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "v2/index_merger.h"
#include "v2/merge_insert.h"

#include <mutex>
#include <numeric>
#include <random>
#include <omp.h>
#include <cstring>
#include <ctime>
#include <timer.h>
#include <iomanip>
#include <atomic>

#include "aux_utils.h"
#include "utils.h"
#include "math_utils.h"
#include "partition_and_pq.h"

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
uint32_t              medoid_id = std::numeric_limits<uint32_t>::max();
std::atomic_bool      _insertions_done(true);
std::atomic_bool      _del_done(true);
std::atomic_bool      _merge_done(true);
std::vector<uint32_t> Lvec;
std::future<void>     delete_future;
std::future<void>     insert_future;
std::future<void>     merge_future;
diskann::Timer        global_timer;
std::string           all_points_file;
bool                  save_index_as_one_file;
bool                  search_only;
bool                  merge_only;
bool                  insert_only;
bool                  merge_insert_only;

std::string TMP_FOLDER;
std::string query_file = "";
std::string truthset_file = "";
std::string log_prefix = "";
unsigned   *gt_ids = nullptr;
uint32_t   *gt_tags = nullptr;
float      *gt_dists = nullptr;
size_t      gt_num, gt_dim;
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

  std::cout << "ITER: INSERT - " << insert_vec.size() << " IDs in "
            << inserted_tags_file << "\n";
  inactive_set.insert(inactive_vec.begin() + insert_vec.size(),
                      inactive_vec.end());
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

  diskann::cout << "ITER: end = " << active_set.size() << ", "
                << inactive_set.size() << "\n";
#ifndef _WINDOWS
  std::cout << "ITER: end = " << active_set.size() << ", "
            << inactive_set.size() << "\n";
  malloc_stats();
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

  std::cout << "ITER: INSERT - " << insert_vec.size() << " IDs in "
            << inserted_tags_file << "\n";
  inactive_set.insert(inactive_vec.begin() + insert_vec.size(),
                      inactive_vec.end());
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

  diskann::cout << "ITER: end = " << active_set.size() << ", "
                << inactive_set.size() << "\n";
#ifndef _WINDOWS
  std::cout << "ITER: end = " << active_set.size() << ", "
            << inactive_set.size() << "\n";
  malloc_stats();
#endif
}

float compute_active_recall(const uint32_t *result_tags,
                            const uint32_t  result_count,
                            const uint32_t *gs_tags, const uint64_t gs_count,
                            const tsl::robin_set<uint32_t> &inactive_set) {
  tsl::robin_set<uint32_t> active_gs;
  for (uint32_t i = 0; i < gs_count && active_gs.size() < result_count; i++) {
    auto iter = inactive_set.find(gs_tags[i]);
    if (iter == inactive_set.end()) {
      active_gs.insert(gs_tags[i]);
    }
  }
  uint32_t match = 0;
  for (uint32_t i = 0; i < result_count; i++) {
    match += (active_gs.find(result_tags[i]) != active_gs.end());
  }
  return ((float) match / (float) result_count) * 100;
}

template<typename T, typename TagT = uint32_t>
void search_disk_index(const std::string              &index_prefix_path,
                       const tsl::robin_set<uint32_t> &inactive_tags,
                       const std::string              &query_path,
                       const std::string              &gs_path) {
  std::string pq_prefix = index_prefix_path + "_pq";
  std::string disk_index_file = index_prefix_path + "_disk.index";
  std::string warmup_query_file = index_prefix_path + "_sample_data.bin";
  uint32_t    beamwidth = params[std::string("beam_width")];
  uint32_t    num_threads = 60;
  std::string query_bin = query_path;
  std::string truthset_bin = gs_path;
  uint64_t    recall_at = params[std::string("recall_k")];
  uint64_t    search_L = ::Lvec[0];
  // hold data
  T        *query = nullptr;
  unsigned *gt_ids = nullptr;
  uint32_t *gt_tags = nullptr;
  float    *gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;

  // load query + truthset
  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);
  diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim);
  if (gt_num != query_num) {
    std::cout << "Error. Mismatch in number of queries and ground truth data"
              << std::endl;
  }

  // load PQ Flash Index
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<T, uint32_t>> _pFlashIndex(
      new diskann::PQFlashIndex<T, uint32_t>(diskann::Metric::L2, reader,
                                             ::save_index_as_one_file, true));
  int res = _pFlashIndex->load(num_threads, pq_prefix.c_str(),
                               disk_index_file.c_str());
  if (res != 0) {
    std::cerr << "Failed to load index.\n";
    exit(-1);
  }

  // prep for search
  std::vector<uint32_t> query_result_ids;
  std::vector<uint32_t> query_result_tags;
  std::vector<float>    query_result_dists;
  query_result_ids.resize(recall_at * query_num);
  query_result_dists.resize(recall_at * query_num);
  query_result_tags.resize(recall_at * query_num);
  diskann::QueryStats  *stats = new diskann::QueryStats[query_num];
  std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
#pragma omp parallel for schedule(dynamic, 1)  // num_threads(1)
  for (_s64 i = 0; i < (int64_t) query_num; i++) {
    _pFlashIndex->cached_beam_search(
        query + (i * query_aligned_dim), recall_at, search_L,
        query_result_ids_64.data() + (i * recall_at),
        query_result_dists.data() + (i * recall_at), beamwidth, stats + i,
        query_result_tags.data() + (i * recall_at));
  }

  // compute mean recall, IOs
  float mean_recall = 0.0f;
  for (uint32_t i = 0; i < query_num; i++) {
    auto *result_tags = query_result_tags.data() + (i * recall_at);
    auto *gs_tags = gt_tags + (i * gt_dim);
    float query_recall = compute_active_recall(
        result_tags, (uint32_t) recall_at, gs_tags, gt_dim, inactive_tags);
    mean_recall += query_recall;
  }
  mean_recall /= query_num;

  float mean_ios = (float) diskann::get_mean_stats(
      stats, query_num,
      [](const diskann::QueryStats &stats) { return stats.n_ios; });
  std::cout << "PQFlashIndex :: recall-" << recall_at << "@" << recall_at
            << ": " << mean_recall << ", mean IOs: " << mean_ios << "\n";
  diskann::aligned_free(query);
  delete[] stats;
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
}

template<typename T, typename TagT = uint32_t>
void search_kernel(diskann::MergeInsert<T>        &merge_insert,
                   const tsl::robin_set<uint32_t> &active_tags, int iter = -1,
                   std::string reason = "", bool print_stats = false) {
  uint64_t recall_at = params[std::string("recall_k")];

  // hold data
  T *query = nullptr;
  // unsigned *gt_ids = nullptr;
  // uint32_t *gt_tags = nullptr;
  // float    *gt_dists = nullptr;
  // size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  size_t query_num, query_dim, query_aligned_dim;

  std::cout << "Loading query : " << ::query_file << std::endl;
  // load query + truthset
  diskann::load_aligned_bin<T>(::query_file, query, query_num, query_dim,
                               query_aligned_dim);
  // std::cout << "Loading gt : " << ::query_file << std::endl;
  // diskann::load_truthset(::truthset_file, gt_ids, gt_dists, gt_num, gt_dim,
  //                        &gt_tags);
  // std::cout << "Loaded gt" << std::endl;
  // if (gt_num != query_num) {
  //   std::cout << "Error. Mismatch in number of queries and ground truth data"
  //             << std::endl;
  // }

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
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      merge_insert.search_sync(query + (i * query_aligned_dim), recall_at, L,
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
    std::sort(latency_stats.begin(), latency_stats.end());
    /**
     * Output logs
     */
    std::string recall_string = "Recall@" + std::to_string(recall_at);
    std::cout << std::setw(6) << "Iter" << std::setw(14) << "Reason"
              << std::setw(14) << "period start" << std::setw(14)
              << "period end" << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(18) << "Mean Latency (ms)" << std::setw(12)
              << "90 Latency" << std::setw(12) << "95 Latency" << std::setw(12)
              << "99 Latency" << std::setw(12) << "99.9 Latency"
              << std::setw(12) << recall_string << std::setw(12)
              << "Mean disk IOs" << std::endl;
    std::cout << "=============================search=========================="
                 "========"
                 "==============="
              << std::endl;
    std::cout << std::setw(6) << iter << std::setw(14) << reason
              << std::setw(14) << start << std::setw(14) << end << std::setw(4)
              << L << std::setw(12) << qps << std::setw(18)
              << ((float) std::accumulate(latency_stats.begin(),
                                          latency_stats.end(), 0)) /
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
              << std::endl;
    /**
     * Output csv
     */
    std::string   log_file_path = ::log_prefix + "_search.csv";
    bool          is_new_file = !file_exists(log_file_path);
    std::ofstream log_file(log_file_path, std::ios::app);
    if (log_file.is_open()) {
      // 如果是新文件，则写入表头
      if (is_new_file) {
        log_file << "iter,reason,period start,period end,Ls,QPS,Mean Latency "
                    "(ms),90 Latency,"
                    "95 Latency,99 Latency,99.9 Latency,"
                 << recall_string << ",Mean disk IOs" << std::endl;
      }

      // 追加数据
      log_file << iter << "," << reason << "," << start << "," << end << ","
               << L << "," << qps << ","
               << ((float) std::accumulate(latency_stats.begin(),
                                           latency_stats.end(), 0)) /
                      (float) query_num
               << ","
               << (float) latency_stats[(_u64) (0.90 * ((double) query_num))]
               << ","
               << (float) latency_stats[(_u64) (0.95 * ((double) query_num))]
               << ","
               << (float) latency_stats[(_u64) (0.99 * ((double) query_num))]
               << ","
               << (float) latency_stats[(_u64) (0.999 * ((double) query_num))]
               << "," << mean_recall << "," << mean_ios << std::endl;

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

template<typename T>
void merge_kernel(diskann::MergeInsert<T> &merge_insert) {
  if (::_merge_done.load()) {
    std::cout << "_merge_done is already true" << std::endl;
    exit(-1);
  }
  auto start = ::global_timer.elapsed() / 1000000;
  merge_insert.final_merge();
  auto end = ::global_timer.elapsed() / 1000000;
  auto diff = end - start;
  /**
   * Output logs
   */
  std::cout << std::setw(14) << "period start" << std::setw(14) << "period end"
            << std::setw(15) << "merge time(s)" << std::endl;
  std::cout << "===============merge================" << std::endl;
  std::cout << std::setw(14) << start << std::setw(14) << end << std::setw(15)
            << diff << std::endl;
  // output csv
  std::string merge_log_file_path = log_prefix + "_merge.csv";
  bool is_new_file = !file_exists(merge_log_file_path);  // 判断是否是新文件

  std::ofstream merge_log_file(merge_log_file_path, std::ios::app);
  if (merge_log_file.is_open()) {
    // 只有新文件才写入表头
    if (is_new_file) {
      merge_log_file << "period start,period end,merge time(s)" << std::endl;
    }

    // 追加数据
    merge_log_file << start << "," << end << "," << diff << std::endl;

    merge_log_file.close();
  } else {
    std::cerr << "Failed to open merge log file!" << std::endl;
  }
  ::_merge_done.store(true);
}

template<typename T, typename TagT = uint32_t>
void insertion_kernel(diskann::MergeInsert<T> &merge_insert,
                      std::string mem_pts_file, std::string mem_tags_file) {
  if (::_insertions_done.load()) {
    std::cout << "Insertions_done is true at the beginning of insertion kernel"
              << std::endl;
    exit(-1);
  }
  T     *data_insert = nullptr;
  size_t npts, ndim, aligned_dim;
  diskann::load_aligned_bin<T>(mem_pts_file, data_insert, npts, ndim,
                               aligned_dim);
  size_t tag_num, tag_dim;
  TagT  *tag_data;
  diskann::load_bin<TagT>(mem_tags_file, tag_data, tag_num, tag_dim);
  if (tag_num != npts) {
    std::cout << "In insertion_kernel(), number of tags loaded is not equal to "
                 "number of points loaded. Exiting....."
              << std::endl;
    exit(-1);
  }
  _s64                i;
  std::vector<double> insert_latencies(npts, 0);
  diskann::Timer      timer;
  auto                start = ::global_timer.elapsed() / 1000000;
#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (i = 0; i < (_s64) npts; i++) {
    diskann::Timer insert_timer;
    // if (merge_insert.insert(data_insert + i * aligned_dim, tag_data[i]) == 0)
    // {
    //   insert_latencies[i] = ((double) insert_timer.elapsed());
    // } else {
    //   std::cout << "Point " << i << "could not be inserted." << std::endl;
    // }
    while (merge_insert.insert(data_insert + i * aligned_dim, tag_data[i]) !=
           0) {
      // std::this_thread::sleep_for(std::chrono::milliseconds(500));
      bool expected = true;
      if (::_merge_done.compare_exchange_strong(expected, false)) {
        diskann::cout << "trigger a merge_mem in insert" << std::endl;
        ::merge_future = std::async(std::launch::async, merge_kernel<T>,
                                    std::ref(merge_insert));
      }
      if (expected) {
        diskann::cout << "wait for 1 ms in insert" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        diskann::cout << "wait for 1 s in insert" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
    insert_latencies[i] = ((double) insert_timer.elapsed());
    if ((i % 1000 == 0) && (i > 0))
      std::cout << "Inserted another 1k points" << std::endl;
  }
  auto  diff = timer.elapsed() / 1000;
  auto  end = ::global_timer.elapsed() / 1000000;
  float qps = (float) (((double) npts) * 1000 / diff);
  std::sort(insert_latencies.begin(), insert_latencies.end());
  /**
   * Output log
   */
  std::cout << std::setw(14) << "Period start" << std::setw(14) << "Period end"
            << std::setw(11) << "insert qps" << std::setw(18)
            << "total time(ms)" << std::setw(18) << "mean latency(us)"
            << std::setw(18) << "10 latency(us)" << std::setw(18)
            << "50 latency(us)" << std::setw(18) << "90 latency(us)"
            << std::setw(18) << "95 latency(us)" << std::setw(18)
            << "99 latency(us)" << std::setw(18) << "99.9 latency(us)"
            << std::endl;
  std::cout << "============================insertion=========================="
               "========="
               "==============================================================="
               "=========================="
            << std::endl;
  std::cout << std::setw(14) << start << std::setw(14) << end << std::setw(11)
            << qps << std::setw(18) << diff << std::setw(18)
            << ((float) std::accumulate(insert_latencies.begin(),
                                        insert_latencies.end(), 0)) /
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
      insert_log_file
          << "Period start,Period end,Insert QPS,Total time(ms),Mean "
             "latency(us),"
             "10 latency(us),50 latency(us),90 latency(us),95 latency(us),"
             "99 latency(us),99.9 latency(us)"
          << std::endl;
    }

    // 计算 Mean Latency
    float mean_latency = ((float) std::accumulate(insert_latencies.begin(),
                                                  insert_latencies.end(), 0)) /
                         (float) npts;

    // 追加数据
    insert_log_file << start << "," << end << "," << qps << "," << diff << ","
                    << mean_latency << ","
                    << insert_latencies[(size_t) (0.1 * ((double) npts))] << ","
                    << insert_latencies[(size_t) (0.5 * ((double) npts))] << ","
                    << insert_latencies[(size_t) (0.9 * ((double) npts))] << ","
                    << insert_latencies[(size_t) (0.95 * ((double) npts))]
                    << ","
                    << insert_latencies[(size_t) (0.99 * ((double) npts))]
                    << ","
                    << insert_latencies[(size_t) (0.999 * ((double) npts))]
                    << std::endl;

    insert_log_file.close();
  } else {
    std::cerr << "Failed to open insert log file!" << std::endl;
  }

  ::_insertions_done.store(true);
  delete[] data_insert;
  delete[] tag_data;
}
template<typename T, typename TagT = uint32_t>
void deletion_kernel(diskann::MergeInsert<T> &merge_insert,
                     tsl::robin_set<uint32_t> del_tags) {
  std::cout << "inside deletion kernel" << std::endl;
  std::cout << "merge_insert_only: " << ::merge_insert_only << std::endl;
  if (::_del_done.load()) {
    std::cout << "_del_done is already true" << std::endl;
    exit(-1);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  diskann::Timer timer;
  auto           start = ::global_timer.elapsed() / 1000000;
  for (auto iter : del_tags) {
    merge_insert.lazy_delete(iter);
  }
  auto  diff = timer.elapsed() / 1000;
  auto  end = ::global_timer.elapsed() / 1000000;
  float qps = (float) (((double) del_tags.size()) * 1000 / diff);
  /**
   * Output Log
   */
  std::cout << std::setw(14) << "Period start" << std::setw(14) << "Period end"
            << std::setw(5) << "qps" << std::setw(25)
            << "total Deletion time(ms)" << std::endl;
  std::cout << "===============Deletion================" << std::endl;
  std::cout << std::setw(14) << start << std::setw(14) << end << std::setw(5)
            << qps << std::setw(25) << diff << std::endl;
  /**
   * Output csv
   */
  std::string delete_log_file_path = log_prefix + "_delete.csv";
  bool is_new_file = !file_exists(delete_log_file_path);  // 判断是否是新文件

  std::ofstream delete_log_file(delete_log_file_path, std::ios::app);
  if (delete_log_file.is_open()) {
    // 只有新文件才写入表头
    if (is_new_file) {
      delete_log_file << "Period start,Period end,QPS,Total Deletion time(ms)"
                      << std::endl;
    }

    // 追加数据
    delete_log_file << start << "," << end << "," << qps << "," << diff
                    << std::endl;

    delete_log_file.close();
  } else {
    std::cerr << "Failed to open delete log file!" << std::endl;
  }
  ::_del_done.store(true);
}
template<typename T, typename TagT = uint32_t>
void run_merge_insert_iter(int iter, diskann::MergeInsert<T> &merge_insert,
                           const std::string        &mem_prefix,
                           tsl::robin_set<uint32_t> &active_set,
                           tsl::robin_set<uint32_t> &inactive_set) {
  // files for mem-DiskANN
  std::string mem_pts_file = mem_prefix + ".data_orig";
  std::string mem_tags_file = mem_prefix + ".tags_orig";
  std::this_thread::sleep_for(std::chrono::seconds(10));  // 休眠10秒以确保同步
  // search_kernel<T>(merge_insert, active_set);
  bool expected = true;
  if (::_merge_done.compare_exchange_strong(expected, false)) {
    // 异步启动合并任务，调用 merge_kernel 函数
    ::merge_future =
        std::async(std::launch::async, merge_kernel<T>, std::ref(merge_insert));
  }
  // 在插入和删除操作未完成时，不断执行搜索操作
  while (!(::_insertions_done.load())) {
    // std::cout << "Search while insert at " << ::global_timer.elapsed() /
    // 1000000
    // << std::endl;

    // 调用 search_kernel 执行搜索操作，使用 active_set
    search_kernel<T>(merge_insert, active_set, iter, "while insert");

    // 每次搜索后休眠 10 秒
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  }

  // 如果插入和删除操作已完成，重置状态并执行后续操作
  if (::_insertions_done.load()) {
    ::_insertions_done.store(false);

    // 调用 search_kernel 执行搜索操作，使用 active_set
    search_kernel<T>(merge_insert, active_set, iter, "before insert");

    std::cout << "ITER: Seeding iteration"
              << "\n";
    // seed the iteration
    seed_insert_iter<T, TagT>(active_set, inactive_set, mem_pts_file,
                              mem_tags_file);

    // 异步启动插入操作，调用 insertion_kernel 函数
    ::insert_future =
        std::async(std::launch::async, insertion_kernel<T>,
                   std::ref(merge_insert), mem_pts_file, mem_tags_file);
  }
  // 检查合并任务的状态
  while (!(::_merge_done.load())) {
    // 在合并任务进行过程中，不断执行搜索操作
    search_kernel<T>(merge_insert, active_set, iter, "while merge");

    // 每次搜索后休眠10秒
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  }
}

template<typename T, typename TagT = uint32_t>
void run_iter(diskann::MergeInsert<T>  &merge_insert,
              const std::string        &mem_prefix,
              tsl::robin_set<uint32_t> &active_set,
              tsl::robin_set<uint32_t> &inactive_set) {
  // files for mem-DiskANN
  std::string mem_pts_file = mem_prefix + ".data_orig";
  std::string mem_tags_file = mem_prefix + ".tags_orig";
  std::this_thread::sleep_for(std::chrono::seconds(10));  // 休眠10秒以确保同步

  // 异步启动合并任务，调用 merge_kernel 函数
  ::merge_future =
      std::async(std::launch::async, merge_kernel<T>, std::ref(merge_insert));

  // 在插入和删除操作未完成时，不断执行搜索操作
  while (!(::_insertions_done.load() && ::_del_done.load())) {
    std::cout << "Search at " << ::global_timer.elapsed() / 1000000
              << " seconds " << std::endl;

    // 调用 search_kernel 执行搜索操作，使用 active_set
    search_kernel<T>(merge_insert, active_set);

    // 每次搜索后休眠 5 秒
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  }

  // 如果插入和删除操作已完成，重置状态并执行后续操作
  if (::_insertions_done.load() && ::_del_done.load()) {
    ::_insertions_done.store(false);
    ::_del_done.store(false);

    std::cout << "Searching all indices" << std::endl;
    std::cout << "Search at " << ::global_timer.elapsed() / 1000000
              << " seconds " << std::endl;

    // 调用 search_kernel 执行搜索操作，使用 active_set
    search_kernel<T>(merge_insert, active_set, true);

    std::cout << "ITER: Seeding iteration"
              << "\n";
    // seed the iteration
    tsl::robin_set<uint32_t> deleted_tags;
    seed_iter<T, TagT>(active_set, inactive_set, mem_pts_file, mem_tags_file,
                       deleted_tags);

    // 异步启动删除操作，调用 deletion_kernel 函数
    ::delete_future = std::async(std::launch::async, deletion_kernel<T, TagT>,
                                 std::ref(merge_insert), deleted_tags);

    // 异步启动插入操作，调用 insertion_kernel 函数
    ::insert_future =
        std::async(std::launch::async, insertion_kernel<T>,
                   std::ref(merge_insert), mem_pts_file, mem_tags_file);
  }
  // 检查合并任务的状态
  std::future_status merge_status;
  do {
    // 非阻塞式等待合并任务完成，每次等待1毫秒
    merge_status = ::merge_future.wait_for(std::chrono::milliseconds(1));
    std::cout << "Search at " << ::global_timer.elapsed() / 1000000
              << " seconds " << std::endl;

    // 在合并任务进行过程中，不断执行搜索操作
    search_kernel<T>(merge_insert, active_set);

    // 每次搜索后休眠1秒
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  } while ((merge_status != std::future_status::ready));
}
template<typename T, typename TagT = uint32_t>
void run_search_iter(diskann::MergeInsert<T>  &merge_insert,
                     tsl::robin_set<uint32_t> &active_set) {
  // 不断执行搜索操作
  // 调用 search_kernel 执行搜索操作，使用 active_set
  search_kernel<T>(merge_insert, active_set);

  // 每次搜索后休眠 0.5 秒
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
}
template<typename T, typename TagT = uint32_t>
void run_single_iter(diskann::MergeInsert<T>  &merge_insert,
                     const std::string        &base_prefix,
                     const std::string        &merge_prefix,
                     const std::string        &mem_prefix,
                     tsl::robin_set<uint32_t> &active_set,
                     tsl::robin_set<uint32_t> &inactive_set,
                     diskann::Distance<T>     *dist_cmp) {
  // files for mem-DiskANN
  std::string mem_pts_file = mem_prefix + ".data_orig";
  std::string mem_tags_file = mem_prefix + ".tags_orig";
  if (::_insertions_done.load() && ::_del_done.load()) {
    ::_insertions_done.store(false);
    ::_del_done.store(false);

    /*    std::cout << "Searching all indices" << std::endl;
        std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                  << " seconds " << std::endl;
        search_kernel<T>(merge_insert, active_set, true);
        */
    std::cout << "ITER: Seeding iteration"
              << "\n";
    // seed the iteration
    tsl::robin_set<uint32_t> deleted_tags;
    seed_iter<T, TagT>(active_set, inactive_set, mem_pts_file, mem_tags_file,
                       deleted_tags);
    ::delete_future = std::async(std::launch::async, deletion_kernel<T, TagT>,
                                 std::ref(merge_insert), deleted_tags);
    ::insert_future =
        std::async(std::launch::async, insertion_kernel<T>,
                   std::ref(merge_insert), mem_pts_file, mem_tags_file);
  }
  std::future_status insert_status, delete_status;
  do {
    insert_status = ::insert_future.wait_for(std::chrono::milliseconds(1));
    delete_status = ::delete_future.wait_for(std::chrono::milliseconds(1));
    std::this_thread::sleep_for(std::chrono::seconds(60));
  } while ((insert_status != std::future_status::ready) ||
           (delete_status != std::future_status::ready));

  ::merge_future =
      std::async(std::launch::async, merge_kernel<T>, std::ref(merge_insert));

  std::future_status merge_status;
  do {
    merge_status = ::merge_future.wait_for(std::chrono::milliseconds(1));
    /*  std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                << " seconds " << std::endl;
        search_kernel<T>(merge_insert, active_set);
        */
    //    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  } while ((merge_status != std::future_status::ready));
}

template<typename T, typename TagT = uint32_t>
void run_all_iters(std::string base_prefix, std::string merge_prefix,
                   const std::string mem_prefix, const std::string data_file,
                   const std::string     active_tags_file,
                   diskann::Distance<T> *dist_cmp) {
  // load all data points
  uint64_t npts = 0, ndims = 0;
  diskann::get_bin_metadata(data_file, npts, ndims);
  std::cout << "Loaded base bin" << std::endl;
  params[std::string("ndims")] = (uint32_t) ndims;

  uint32_t n_iters = params["n_iters"];
  // load active tags
  tsl::robin_set<uint32_t> active_tags;
  TagT                    *tag_data;
  size_t                   tag_num, tag_dim;
  if (::save_index_as_one_file) {
    uint64_t *metadata;
    size_t    nr, nc;
    diskann::load_bin<uint64_t>(active_tags_file, metadata, nr, nc);
    diskann::load_bin<TagT>(active_tags_file, tag_data, tag_num, tag_dim,
                            metadata[7]);
  } else {
    diskann::load_bin<TagT>(active_tags_file, tag_data, tag_num, tag_dim);
  }

  size_t tags_loaded = 0;
  size_t del_tags_found = 0;
  active_tags.reserve(tag_num);
  for (size_t i = 0; i < tag_num; i++) {
    if (tag_data[i] != std::numeric_limits<uint32_t>::max()) {
      active_tags.insert(tag_data[i]);
      tags_loaded++;
    } else {
      if (del_tags_found < 5)
        std::cout << "Driver file found invalid tag in active tag file : "
                  << tag_data[i] << std::endl;
      del_tags_found++;
    }
  }
  std::cout << "Loaded " << tags_loaded << " tags" << std::endl;
  delete[] tag_data;
  std::cout << del_tags_found
            << " deleted/invalid tags found in active tags file" << std::endl;
  // read medoid ID from base_prefix
  std::ifstream disk_reader(base_prefix + "_disk.index", std::ios::binary);
  disk_reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
  disk_reader.seekg(2 * sizeof(uint64_t), std::ios::cur);
  uint64_t medoid = std::numeric_limits<uint64_t>::max();
  disk_reader.read((char *) &medoid, sizeof(uint64_t));
  std::cout << "Detected medoid = " << medoid
            << " ==> excluding from insert/deletes.\n";
  ::medoid_id = (uint32_t) medoid;

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
  active_tags.erase(::medoid_id);

  diskann::Parameters paras;
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

  const std::string             working_folder = ::TMP_FOLDER;
  diskann::Metric               metric = diskann::Metric::L2;
  diskann::MergeInsert<T, TagT> merge_insert(
      paras, ndims, mem_prefix, base_prefix, merge_prefix, dist_cmp, metric,
      ::save_index_as_one_file, working_folder);
  //  search_kernel<T, TagT>(merge_insert, active_tags, true);
  for (size_t i = 0; i < n_iters; i++) {
    std::cout << "============ITER : " << i << "=============" << std::endl;
    if (::search_only) {
      if (file_exists(mem_prefix)) {
        merge_insert._mem_index_0->load(mem_prefix.c_str());
        tsl::robin_set<uint32_t> mem_active_tags;
        merge_insert._mem_index_0->get_active_tags(mem_active_tags);
        for (auto iter : mem_active_tags) {
          if (active_tags.find(iter) != active_tags.end()) {
            active_tags.insert(iter);
          }
        }
      }
      run_search_iter(merge_insert, active_tags);
    } else if (::merge_only) {
      merge_insert._mem_index_0->load(mem_prefix.c_str());
      merge_insert._mem_points = merge_insert._mem_index_0->get_num_points();
      merge_kernel<T>(merge_insert);
      break;
    } else if (::merge_insert_only) {
      run_merge_insert_iter<T>(i, merge_insert, mem_prefix, active_tags,
                               inactive_tags);
    } else {
      run_iter<T>(merge_insert, mem_prefix, active_tags, inactive_tags);
    }
  }
  while (!(::_insertions_done.load())) {
    // 调用 search_kernel 执行搜索操作，使用 active_set
    search_kernel<T>(merge_insert, active_tags, n_iters, "while insert");

    // 每次搜索后休眠 10 秒
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  }
  while (!(::_merge_done.load())) {
    // 在合并任务进行过程中，不断执行搜索操作
    search_kernel<T>(merge_insert, active_tags, n_iters, "while merge");

    // 每次搜索后休眠10秒
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  }
  /*
  std::cout << "Done running all iterations, now merging any leftover points."
            << std::endl;
  std::future_status merge_status, insert_status, delete_status;
  do {
    merge_status = ::merge_future.wait_for(std::chrono::milliseconds(1));
    insert_status = ::insert_future.wait_for(std::chrono::milliseconds(1));
    delete_status = ::delete_future.wait_for(std::chrono::milliseconds(1));

    //    search_kernel<T>(merge_insert, active_tags,
    //    false);
  } while ((merge_status != std::future_status::ready) ||
           (insert_status != std::future_status::ready) ||
           (delete_status != std::future_status::ready));
  merge_kernel(merge_insert);
  */
  //  search_kernel<T, TagT>(merge_insert, active_tags,
  //  true);
}

int main(int argc, char **argv) {
  std::cout << "Entering main()" << std::endl;
  if (argc < 22) {
    std::cout << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <WORKING_FOLDER> <base_prefix> "
                 "<merge_prefix> <mem_prefix> <L_mem> <alpha_mem> <L_disk> "
                 "<alpha_disk> "
              << " <full_data_bin> <single_file[0/1]> <query_bin> <truthset>"
              << " <n_iters> <total_insert_count> <total_delete_count> <range> "
                 "<recall_k> "
                 "<search/merge/insert_only> (0: search; 1: merge; 2: insert)"
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
  TMP_FOLDER = argv[arg_no++];
  std::string base_prefix(argv[arg_no++]);
  std::string merge_prefix(argv[arg_no++]);
  std::string mem_prefix(argv[arg_no++]);
  ::log_prefix = std::string(argv[arg_no++]);
  unsigned    L_mem = (unsigned) atoi(argv[arg_no++]);
  float       alpha_mem = (float) atof(argv[arg_no++]);
  unsigned    L_disk = (unsigned) atoi(argv[arg_no++]);
  float       alpha_disk = (float) atof(argv[arg_no++]);
  std::string data_bin(argv[arg_no++]);
  int         single_file = atoi(argv[arg_no++]);
  std::string query_path(argv[arg_no++]);
  std::string gt_file(argv[arg_no++]);
  int         n_iters = atoi(argv[arg_no++]);
  uint32_t    insert_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    delete_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    range = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    recall_k = (uint32_t) atoi(argv[arg_no++]);
  int         which_only = atoi(argv[arg_no++]);

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

  ::query_file = ::query_file + query_path;
  ::truthset_file = gt_file;
  ::all_points_file = data_bin;
  if (single_file == 1)
    ::save_index_as_one_file = true;
  else
    ::save_index_as_one_file = false;

  ::search_only = false;
  ::merge_only = false;
  ::insert_only = false;
  ::merge_insert_only = false;
  if (which_only == 1)
    ::search_only = true;
  else if (which_only == 0) {
    ::merge_only = true;
    ::_merge_done.store(false);
  } else if (which_only == -1) {
    ::insert_only = true;
  } else if (which_only == 2) {
    ::merge_insert_only = true;
  }

  std::string active_tags_filename;
  if (single_file)
    active_tags_filename = base_prefix + "_disk.index";
  else
    active_tags_filename = base_prefix + "_disk.index.tags";

  // load truthset
  if (!::merge_only) {
    std::cout << "Loading truthset : " << ::truthset_file << std::endl;
    diskann::load_truthset(::truthset_file, ::gt_ids, ::gt_dists, ::gt_num,
                           ::gt_dim, &::gt_tags);
  }
  std::cout << "Calling run_all_iters()" << std::endl;
  if (index_type == std::string("float")) {
    diskann::DistanceL2 dist_cmp;
    run_all_iters<float>(base_prefix, merge_prefix, mem_prefix, data_bin,
                         active_tags_filename, &dist_cmp);
  } else if (index_type == std::string("uint8")) {
    diskann::DistanceL2UInt8 dist_cmp;
    run_all_iters<uint8_t>(base_prefix, merge_prefix, mem_prefix, data_bin,
                           active_tags_filename, &dist_cmp);
  } else if (index_type == std::string("int8")) {
    diskann::DistanceL2Int8 dist_cmp;
    run_all_iters<int8_t>(base_prefix, merge_prefix, mem_prefix, data_bin,
                          active_tags_filename, &dist_cmp);
  } else {
    std::cout << "Unsupported type : " << index_type << "\n";
  }
  delete[] ::gt_ids;
  delete[] ::gt_dists;
  delete[] ::gt_tags;
  std::cout << "Exiting\n";
  return 0;
}
