#pragma once
#include <vector>
#include <mutex>
#include "utils.h"
#include "gp/graph_partitioner.h"

namespace lsmidx{
class InDegreeRankRelayout : public GraphPartitioner{
public:
  InDegreeRankRelayout(std::string index_file):GraphPartitioner(index_file){};
  void GraphPartition(PartitionOption option) override;
  void generate_visit_order_by_in_degree(std::vector<unsigned>& visit_order, _u64 npts,std::vector<std::vector<unsigned>>& reverse_graph);
};
}; // namesp lsmidx
