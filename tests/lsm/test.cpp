#include <vector>
#include <string>
#include <limits>
#include "aux_utils.h"
#include "math_utils.h"
#include "utils.h"
#include "index.h"
#include "lsm/merger/util/index_data_iterator.h"
#include "linux_aligned_file_reader.h"
#include "gp/graph_partitioner.h"
#include "gp/greedy_allocation_relayout.h"
#include "gp/neighbor_frequency_relayout.h"
#include "gp/in_degree_rank_relayout.h"
int main(int argc, char** argv) {
  std::string disk_index_prefix1 = "/home/hlqiu/index/lsmidx_merge_insert_test/level1_0";
  std::string out_disk_index_prefix1 = "/home/hlqiu/index/test_redistribute/level1_0";

  lsmidx::PartitionOption opt;

  std::shared_ptr<lsmidx::InDegreeRankRelayout> idrr = std::make_shared<lsmidx::InDegreeRankRelayout>(disk_index_prefix1+"_disk.index");
  idrr->gm->graph_degree_statistic();
  idrr->partition_statistic("raw", 0, 0);
  idrr->GraphPartition(opt);
  std::shared_ptr<lsmidx::GreedyAllocationRelayout> gar = std::make_shared<lsmidx::GreedyAllocationRelayout>(idrr);
  
  gar->GraphPartition(opt);
  lsmidx::NeighborFrequencyRelayout nfr(gar);
  opt.nfr_rounds = 10;
  nfr.GraphPartition(opt);
  // nfr.partition_statistic();
  // nfr.RedistributeIndex<float, uint32_t>(disk_index_prefix1, out_disk_index_prefix1);
}