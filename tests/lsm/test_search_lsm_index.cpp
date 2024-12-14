// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
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
#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <pthread.h>
#include <sched.h>

#include "omp.h"
#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"
#include "lsm/lsm_index.h"
#include "lsm/options.h"

#define NUM_SEARCH_THREADS 6

// random number generator
std::random_device dev;
std::mt19937       rng(dev());

tsl::robin_map<std::string, uint32_t> params;
float                                 mem_alpha, merge_alpha;
uint32_t              medoid_id = std::numeric_limits<uint32_t>::max();
std::vector<uint32_t> Lvec;
diskann::Timer        global_timer;
std::string           all_points_file;
bool                  save_index_as_one_file;
std::string           query_file = "";
std::string           truthset_file = "";

template<typename T, typename TagT = uint32_t>
void search_kernel(lsmidx::LSMVectorIndex<T, TagT>        &lsm_index,
                   const tsl::robin_set<uint32_t> &active_tags,
                   bool                            print_stats = false) {
  uint64_t recall_at = params[std::string("recall_k")];

  // hold data
  T        *query = nullptr;
  unsigned *gt_ids = nullptr;
  uint32_t *gt_tags = nullptr;
  float    *gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;

  std::cout << "Loading query : " << ::query_file << std::endl;
  // load query + truthset
  diskann::load_aligned_bin<T>(::query_file, query, query_num, query_dim,
                               query_aligned_dim);
  std::cout << "Loaded query : " << ::truthset_file << std::endl;
  std::cout << "Loading gt : " << ::query_file << std::endl;
  diskann::load_truthset(::truthset_file, gt_ids, gt_dists, gt_num, gt_dim,
                         &gt_tags);
  std::cout << "Loaded gt" << std::endl;
  if (gt_num != query_num) {
    std::cout << "Error. Mismatch in number of queries and ground truth data"
              << std::endl;
  }

  if (print_stats) {
    std::string recall_string = "SS-Recall@" + std::to_string(recall_at);
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(18) << "Mean Latency (ms)" << std::setw(12)
              << "90 Latency" << std::setw(12) << "95 Latency" << std::setw(12)
              << "99 Latency" << std::setw(12) << "99.9 Latency"
              << std::setw(12) << recall_string << std::setw(12)
              << "Mean disk IOs" << std::endl;

    std::cout

        << "==============================================================="
           "==============="
        << std::endl;
  } else {
    std::string recall_string = "Recall@" + std::to_string(recall_at);
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(18) << "Mean Latency (ms)" << std::setw(12)
              << "90 Latency" << std::setw(12) << "95 Latency" << std::setw(12)
              << "99 Latency" << std::setw(12) << "99.9 Latency"
              << std::setw(12) << recall_string << std::setw(12)
              << "Mean disk IOs" << std::endl;
    std::cout
        << "==============================================================="
           "==============="
        << std::endl;
  }

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
    std::chrono::duration<double> diff = e - s;
    float qps = (float) (((double) query_num) / diff.count());
    // compute mean recall, IOs
    float mean_recall = 0.0f;
    mean_recall = diskann::calculate_recall(
        (unsigned) query_num, gt_ids, gt_dists, (unsigned) gt_dim,
        query_result_tags.data(), (unsigned) recall_at, (unsigned) recall_at,
        active_tags);
    //    mean_recall /= (float) query_num;
    float mean_ios = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats &stats) { return stats.n_ios; });
    std::sort(latency_stats.begin(), latency_stats.end());
    std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
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
    delete[] stats;
  }
  diskann::aligned_free(query);
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
}

template<typename T, typename TagT = uint32_t>
void run_search_iter(lsmidx::LSMVectorIndex<T, TagT>  &lsm_index,
                     tsl::robin_set<uint32_t> &active_set) {
  // 不断执行搜索操作
  // 调用 search_kernel 执行搜索操作，使用 active_set
  search_kernel<T, TagT>(lsm_index, active_set);

  // 每次搜索后休眠 0.5 秒
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
}
template<typename T, typename TagT = uint32_t>
void run_all_iters(const std::string working_dir, const std::string index_name, diskann::Distance<T> *dist_cmp) {
  uint32_t n_iters = params["n_iters"];
  uint64_t npts = 0, ndims = 0;
  diskann::get_bin_metadata(::query_file, npts, ndims);

  lsmidx::BuildOptions options;
  options.dist_metric = diskann::Metric::L2;
  options.is_single_file_index = ::save_index_as_one_file;
  options.dimension = ndims;
  diskann::Parameters& paras = options.params;
  paras.Set<unsigned>("L_mem", params[std::string("L")]);
  paras.Set<unsigned>("R_mem", params[std::string("range")]);
  paras.Set<float>("alpha_mem", ::mem_alpha);
  paras.Set<unsigned>("L_disk", params[std::string("L")]);
  paras.Set<unsigned>("R_disk", params[std::string("range")]);
  paras.Set<float>("alpha_disk", ::merge_alpha);
  paras.Set<unsigned>("C", params[std::string("merge_maxc")]);
  paras.Set<unsigned>("beamwidth", params[std::string("beam_width")]);
  paras.Set<unsigned>("nodes_to_cache",
                      params[std::string("disk_search_node_cache_count")]);
  paras.Set<unsigned>("num_search_threads",
                      params[std::string("disk_search_nthreads")]);
  
  lsmidx::LSMVectorIndex<T, TagT>  lsm_index(options, working_dir, index_name, dist_cmp);
  tsl::robin_set<uint32_t> active_tags;
  lsm_index.GetActiveTags(active_tags);
  std::cout << "Loaded " << active_tags.size() << " tags" << std::endl;
  for (size_t i = 0; i < n_iters; i++) {
    std::cout << "ITER : " << i << std::endl;
    run_search_iter(lsm_index, active_tags);
  }
}
int main(int argc, char** argv) {
  std::vector<std::string> args = {
    "[data_type<float/int8/uint8>] ",
    "[working_dir] ",
    "[index_name] ",
    "<query_bin> ",
    "<truthset> ",
    "<single_file_index(0/1)> ",
    "<n_iters> ",
    "<range> ",
    "<alpha> ",
    "<recall_k> ",
    "<search_L> "
  };
  if (argc != args.size() + 1) {
    diskann::cout << "Usage: " << argv[0];
    for(auto arg: args){
      diskann::cout << arg;
    }
    diskann::cout << std::endl;
    return 0;
  }
  std::cout.setf(std::ios::unitbuf);

  int         arg_no = 1;
  std::string index_type = argv[arg_no++];
  std::string working_dir = argv[arg_no++];
  std::string index_name = argv[arg_no++];
  ::query_file = std::string(argv[arg_no++]);
  ::truthset_file = std::string(argv[arg_no++]);
  bool        single_file = atoi(argv[arg_no++]) == 1;
  int         n_iters = atoi(argv[arg_no++]);
  uint32_t    range = (uint32_t) atoi(argv[arg_no++]);
  float       alpha = (float) atof(argv[arg_no++]);
  uint32_t    recall_k = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    L = (uint32_t) atoi(argv[arg_no++]);
  if (L >= recall_k)
      ::Lvec.push_back(L);

  std::cout << "Assigning parameters" << std::endl;
  params[std::string("n_iters")] = n_iters;
  params[std::string("range")] = range;
  params[std::string("recall_k")] = recall_k;

  // hard-coded params
  params[std::string("disk_search_node_cache_count")] = 100;
  params[std::string("disk_search_nthreads")] = 16;
  params[std::string("beam_width")] = 4;
  params[std::string("L")] = L;
  mem_alpha = alpha;
  merge_alpha = alpha;
  params[std::string("merge_maxc")] = (uint32_t) (range * 2.5);
  ::save_index_as_one_file = single_file;

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
  std::cout << "Exiting\n";
  return 0;
}
