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

int merge_level;
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
void merge_kernel(lsmidx::LSMVectorIndex<T, TagT>        &lsm_index,
                   bool                            print_stats = false) {
  
  omp_set_max_active_levels(4);
  if(merge_level == -1){
    lsmidx::WriteOptions opt;
    // lsm_index.Delete(opt, lsmidx::TagSlice<TagT>(2));
    // lsm_index.Delete(opt, lsmidx::TagSlice<TagT>(3));
    // lsm_index.Delete(opt, lsmidx::TagSlice<TagT>(4));
    // lsm_index.Delete(opt, lsmidx::TagSlice<TagT>(5));
    lsm_index.TriggerMergeMemIndex();
  }else{
    lsm_index.TriggerMergeDiskIndex(merge_level);
  }
  
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

  merge_kernel<T>(lsm_index);
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
    "<search_L> ",
    "<level>"
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
              merge_level = atoi(argv[arg_no++]);
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
