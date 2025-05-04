#pragma once
#include <vector>
#include <mutex>
#include "utils.h"
#include "gp/graph_partitioner.h"

namespace lsmidx{
class GreedyAllocationRelayout : public GraphPartitioner{
public:
  GreedyAllocationRelayout(std::shared_ptr<GraphPartitioner> other):GraphPartitioner(other){};
  void GraphPartition(PartitionOption option) override;
  unsigned DecideInsertPid(unsigned cur_nid);
};
}; // namesp lsmidx
