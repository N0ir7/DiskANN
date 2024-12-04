#pragma once

#include <climits>
#include <string>
#include <vector>
#include "utils.h"
#include "parameters.h"

#define SECTORS_PER_MERGE (uint64_t) 65536
// max number of points per mem index being merged -- 32M
#define MAX_PTS_PER_MEM_INDEX (uint64_t)(1 << 25)
#define INDEX_OFFSET (uint64_t)(MAX_PTS_PER_MEM_INDEX * 4)
#define MAX_INSERT_THREADS (uint64_t) 18
#define MAX_N_THREADS (uint64_t) 18
#define NUM_INDEX_LOAD_THREADS (uint64_t) 18
#define PER_THREAD_BUF_SIZE (uint64_t)(65536 * 64 * 4)

#define PQ_FLASH_INDEX_MAX_NODES_TO_CACHE 200000

namespace lsmidx{
namespace config {
  const int disk_indexes_level = 2;
  const int mem_indexes_num = 2;
  const int level0_merge_thresh = 16000000;
  const int num_levels = 2;
  const int leveln_merge_thresh[2] = {1, INT_MAX}; // static todo: 确定阈值
  const std::string mem_index_name = "mem_index";
  const std::vector<std::string> leveln_index_names = {"level1", "level2"};
}  // namespace config
typedef struct BuildOptions {
  diskann::Parameters params;
  diskann::Metric dist_metric;
  bool is_single_file_index;
  size_t dimension;
} BuildOptions;

typedef struct SearchOptions {
  uint64_t K;
  uint64_t search_L;
  uint64_t beamwidth;
} SearchOptions;

typedef struct WriteOptions {

} WriteOptions;
}