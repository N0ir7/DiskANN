#include "gp/in_degree_rank_relayout.h"
#include <unordered_map>
#include <unordered_set>
namespace lsmidx{
void InDegreeRankRelayout::generate_visit_order_by_in_degree(std::vector<unsigned>& visit_order, _u64 npts,std::vector<std::vector<unsigned>>& reverse_graph){
  std::vector<unsigned> in_degrees(npts);
  for (size_t i = 0; i < npts; ++i) {
    in_degrees[i] = reverse_graph[i].size();
  }
  generate_visit_order_seq(visit_order, npts);
  /**
   * 根据每个节点的入度，对节点编号进行升序排序。
   * 入度较大的节点排在前面，入度较小的节点排在后面。
   * 排序结果存入 visit_order 中，visit_order[i] 表示第 i 个被访问的节点编号。
   */
  std::sort(visit_order.begin(), visit_order.end(), [&](unsigned a, unsigned b) {
    return in_degrees[a] > in_degrees[b];
  });
  return;
}
void InDegreeRankRelayout::GraphPartition(PartitionOption option){
  this->_partition.clear();
  this->_partition.resize(_partition_number);
  std::unordered_set<unsigned> vis;
  std::vector<unsigned> init_stream;
  auto& full_graph = gm->full_graph;

  int lock_nums = option.lock_nums;
  // 生成访问顺序队列
  generate_visit_order_by_in_degree(init_stream, this->nd(), gm->reverse_graph);
  auto qs = std::chrono::high_resolution_clock::now();
  /**
   * 按照访问顺序进行IDRR算法分配
  */
 unsigned cur_pid = 0;
  for (auto cur_nid : init_stream) {
    if (vis.count(cur_nid)) {
      continue;  // has insert into partition
    }
    while(!this->InsertPointIntoPartition(cur_pid, cur_nid)){
      cur_pid++;
    }
    if(cur_pid == this->_partition_number){
      std::cout << "Error: partition_number is not enough while allocate " << cur_nid << std::endl;
      return;
    }
    vis.insert(cur_nid);
  }
  auto qe = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = qe - qs;
  partition_statistic("idrr",0, diff.count() * 1000);
  // Lock top lock_nums point in init_stream
  this->Lock(init_stream, lock_nums, &vis);

  std::cout << "IDRR over." << std::endl;
}
};