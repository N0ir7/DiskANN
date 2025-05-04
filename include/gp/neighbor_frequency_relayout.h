#pragma once
#include <vector>
#include <mutex>
#include "utils.h"
#include "gp/graph_partitioner.h"

namespace lsmidx{
struct NFRStat{
  double ivf_time = 0;
  unsigned select_free = 0;
  uint64_t select_nums = 0;
  void Aggregate(NFRStat& other){
    this->ivf_time = other.ivf_time;
    this->select_free = other.select_free;
    this->select_nums = other.select_nums;
  }
};
class NeighborFrequencyRelayout : public GraphPartitioner{
public:
  NeighborFrequencyRelayout(std::shared_ptr<GraphPartitioner> other):GraphPartitioner(other){};
  void GraphPartition(PartitionOption option) override;
  void Init(std::shared_ptr<GraphPartitioner> other);
private:
  void GraphPartitionWithLDG(NFRStat* stat = nullptr);
  unsigned SelectPartition(unsigned i, NFRStat* stat = nullptr);
  unsigned Sync(unsigned i, NFRStat* stat = nullptr);
};
}; // namesp lsmidx
