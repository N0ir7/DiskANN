#include "gp/greedy_allocation_relayout.h"
#include <unordered_map>
#include <unordered_set>
namespace lsmidx{

unsigned GreedyAllocationRelayout::DecideInsertPid(unsigned cur_nid){
  unsigned pid = 0;
  auto iter = id2pid.find(cur_nid);
  if(iter != id2pid.end()){
    pid = iter->second;
  }else{
    SelectFree(pid);
  }
  return pid;
}
void GreedyAllocationRelayout::GraphPartition(PartitionOption option){
  std::vector<unsigned> init_stream;
  for (unsigned i = 0; i < _partition.size(); i++) {
    for (unsigned j = 0; j < _partition[i].size(); j++) {
      init_stream.emplace_back(_partition[i][j]);
    }
  }
  this->_partition.clear();
  this->_partition.resize(_partition_number);
  std::unordered_set<unsigned> vis;
  // auto& full_graph = gm->full_graph;
  auto& reverse_graph = gm->reverse_graph;

  int lock_nums = option.lock_nums;
  // 生成访问顺序队列
  auto qs = std::chrono::high_resolution_clock::now();
  /**
   * 按照访问顺序进行GAR算法分配
  */
  int num = 0;
  for (auto cur_nid : init_stream) {
    if (vis.count(cur_nid)) {
      continue;  // has insert into partition
    }
    unsigned cur_pid = DecideInsertPid(cur_nid);
    // try insert cur pt into a partition
    for(;cur_pid < this->_partition_number; cur_pid++){
      if(this->InsertPointIntoPartitionRaw(cur_pid, cur_nid)){
        vis.insert(cur_nid);
        break;
      }
    }
    if(cur_pid == this->_partition_number){
      if(SelectFree(cur_pid)){
        this->InsertPointIntoPartitionRaw(cur_pid, cur_nid);
        vis.insert(cur_nid);
      }else{
        std::cout << "Error: partition_number is not enough while allocate " << cur_nid << std::endl;
        return;
      }
    }
    
    // try insert its in neighbor into the same paretition
    for (unsigned neighbor : reverse_graph[cur_nid]) {
      if (vis.count(neighbor)){
        continue;
      }
      unsigned neighbor_pid = DecideInsertPid(neighbor);
      if(neighbor_pid < cur_pid){
        num++;
        neighbor_pid = cur_pid;
      }
      for(;neighbor_pid < this->_partition_number; neighbor_pid++){
        if(this->InsertPointIntoPartition(neighbor_pid, neighbor)){
          vis.insert(neighbor);
          break;
        }
      }
      if(neighbor_pid == this->_partition_number){
        if(SelectFree(neighbor_pid)){
          this->InsertPointIntoPartition(neighbor_pid, neighbor);
          vis.insert(neighbor);
        }else{
          std::cout << "Error: partition_number is not enough while allocate " << neighbor << std::endl;
          return;
        }
      }
    }
  }
  this->re_id2pid();
  auto qe = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = qe - qs;
  partition_statistic("gar",0, diff.count() * 1000);
  // Lock top lock_nums point in init_stream
  this->Lock(init_stream, lock_nums, &vis);

  std::cout << "GAR over."<<num << std::endl;
}
};