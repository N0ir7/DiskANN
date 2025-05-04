#pragma once
#include <vector>
#include <mutex>
#include <omp.h>
#include <list>
#include "utils.h"
#include "gp/graph_manager.h"
#include "concurrent_queue.h"
#include "lsm/options.h"
#include "aligned_file_reader.h"
#include "pq_flash_index.h"
#include "lsm/merger/util/sector_buffer_pool.h"

inline void generate_visit_order_seq(std::vector<unsigned>& visit_order, _u64 npts){
  visit_order.resize(npts);
  std::iota(visit_order.begin(), visit_order.end(), 0);
  return;
}
inline void generate_visit_order_random(std::vector<unsigned>& visit_order, _u64 npts){

  generate_visit_order_seq(visit_order, npts);
  auto rng = std::default_random_engine{};
  std::shuffle(visit_order.begin(), visit_order.end(), rng);
  return;
}
namespace lsmidx{

struct PartitionOption{
  std::string save_path = "";
  /**
   * used for GAR
  */
  int lock_nums = 0;
  /**
   * used for NFR
  */
  int nfr_rounds = 5;
  /**
   * used for BNS
  */
  int bns_rounds = 5;
};
class GraphPartitioner{
  friend class NeighborFrequencyRelayout;
  friend class GreedyAllocationRelayout;
  friend class BlockNeighborSwap;
public:
  GraphPartitioner(std::shared_ptr<GraphManager> gm);
  GraphPartitioner(std::string index_file);
  GraphPartitioner(std::shared_ptr<GraphPartitioner> other);
  virtual void GraphPartition(PartitionOption option) = 0;
  void partition_statistic(std::string algo, int round, float time_in_ms);
  void re_id2pid();
  void Init();
  void Lock(std::vector<unsigned> & init_stream, int lock_npts = 0, std::unordered_set<unsigned>* vis = nullptr);
  bool InsertPointIntoPartition(unsigned pid, unsigned nid);
  bool InsertPointIntoPartitionRaw(unsigned pid, unsigned nid);
  bool SelectFree(unsigned& pid);
  size_t dim(){
    return gm->_dim;
  }
  _u64 nd(){
    return gm->_nd;
  }
  _u64 width(){
    return gm->_width;
  }
  _u64 ep(){
    return gm->_ep;
  }
  std::vector<_u64> partition_max_size(){
    return this->_partition_max_size;
  }
  _u64 partition_num(){
    return this->_partition_number;
  }
  template<typename T,typename TagT>
  void RedistributeIndex(std::string from_index_prefix, std::string output_index_name);
  /**
   * graph info
  */
  std::shared_ptr<GraphManager> gm;
protected:
  template<typename T>
  void InsertPointIntoNewIndex(std::ofstream& writer, std::vector<unsigned>& id_map, unsigned start_id);
  
  /**
   * partition info
  */
  // the number of partitions
  _u64 _partition_number = 0;
  // partition size threshold
  std::vector<_u64> _partition_max_size;
  /**
   * used for lock
  */
  std::vector<std::unique_ptr<std::mutex>> pmutex;
  std::vector<bool> _lock_nodes;
  std::vector<bool> _lock_pids;
  // each partition set
  std::vector<std::vector<unsigned>> _partition;
  std::unordered_map<unsigned, unsigned> id2pid;
  /**
   * Process information record variable
  */
  std::string index_file;
  _u64 nnodes_per_sector;
  uint32_t max_node_len;
  uint32_t data_dim;
  /**
   * other tool util
  */
  std::unique_ptr<ReadOnlySectorBufferPool> bp;
};
}; // namesp lsmidx