#include "gp/neighbor_frequency_relayout.h"
#include <unordered_map>
#include <unordered_set>
namespace lsmidx{
void NeighborFrequencyRelayout::Init(std::shared_ptr<GraphPartitioner> other){
  this->gm = other->gm;
  this->_partition_number = other->_partition_number;
  this->_partition_max_size = other->_partition_max_size;
  this->pmutex = std::move(other->pmutex);
  this->_lock_nodes = std::move(other->_lock_nodes);
  this->_lock_pids = std::move(other->_lock_pids);
  this->id2pid = std::move(other->id2pid);
}
void NeighborFrequencyRelayout::GraphPartition(PartitionOption option){
  int& rounds = option.nfr_rounds;
  std::string& save_path = option.save_path;
  NFRStat* stats = new NFRStat[rounds];
  for (int cur_round = 0; cur_round < rounds; cur_round++) {
    GraphPartitionWithLDG(stats + cur_round);

    /**
     * output some log info
    */
    std::cout << "select free: " << (double)stats[cur_round].select_free / _partition_number << std::endl;
    std::cout << "ivf time: " << stats[cur_round].ivf_time << " ;round: " << cur_round + 1 << std::endl;
    partition_statistic("nfr", cur_round, stats[cur_round].ivf_time * 1000);
  }
  NFRStat total;
  for(int i = 0; i < rounds; i++){
    total.Aggregate(stats[i]);
  }
  std::cout << "select pid nums:" << total.select_nums << " total select free: " << total.select_free << std::endl;
  std::cout << "total ivf time: " << total.ivf_time << std::endl;

  std::cout << "NFR over." << std::endl;
  delete[] stats;
}
void NeighborFrequencyRelayout::GraphPartitionWithLDG(NFRStat* stat){
    std::vector<unsigned> init_stream;

    // 生成随机访问队列
    generate_visit_order_random(init_stream, nd());

    // 清空之前的分区信息
    this->_partition.clear();
    this->_partition.resize(_partition_number);

    /**
     * 遵循NFR算法进行分区
    */
    auto qs = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 1) num_threads(16)
    for (unsigned i = 0; i < nd(); i++) {
      size_t n = init_stream[i];
      if (_lock_nodes[n]) continue;
      Sync(n, stat);
      if(i >0 && i % 10000 == 0){
        std::cout<< "partition: " << i << '/' << nd() << std::endl;
      }
    }
    this->re_id2pid();
    auto qe = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = qe - qs;
    if(stat){
      stat->ivf_time += diff.count();
    }
}
unsigned NeighborFrequencyRelayout::Sync(unsigned nid, NFRStat* stat) {
  unsigned pid = SelectPartition(nid, stat);
  pmutex[pid]->lock();

  while (_partition[pid].size() == _partition_max_size[pid]) {
    pmutex[pid]->unlock();
    pid = SelectPartition(nid, stat);
    pmutex[pid]->lock();
  }
  this->InsertPointIntoPartition(pid, nid);
  pmutex[pid]->unlock();

  return pid;
}

unsigned NeighborFrequencyRelayout::SelectPartition(unsigned i, NFRStat* stat) {
    if(stat){
      #pragma omp atomic
      stat->select_nums++;
    }
    auto& full_graph =  gm->full_graph;
    auto& reverse_graph =  gm->reverse_graph;
    
    unsigned res = std::numeric_limits<unsigned int>::max();
    std::unordered_map<unsigned, unsigned> pcount_direct;

    unsigned max_pid_in_direct = 0;
    std::unordered_map<unsigned, unsigned> pcount_reverse;
    unsigned min_pid_in_reverse = _partition_number-1;
    // 统计正向边邻居的partition分布
    for (auto n : full_graph[i]) {
      unsigned pid = id2pid[n];
      if (pid == std::numeric_limits<unsigned int>::max()) continue;
      pcount_direct[pid] = pcount_direct[pid] + 1;
      if(pid > max_pid_in_direct){ // 统计最大正向邻居
        max_pid_in_direct = pid;
      }
    }
    // 统计反向边邻居的partition分布
    for (auto n : reverse_graph[i]) {
      unsigned pid = id2pid[n];
      if (pid == std::numeric_limits<unsigned int>::max()) continue;
      pcount_reverse[pid] = pcount_reverse[pid] + 1;
      if(pid < min_pid_in_reverse){ // 统计最大反向邻居
        min_pid_in_reverse = pid;
      }
    }
    /**
     * 我们希望点在所有正向邻居的后面，在所有反向邻居的前面
    */
    if(max_pid_in_direct <= min_pid_in_reverse){
      /**
       * 如果最大正向邻居 <= 最小反向邻居，那么是可以取到最优解的, 也就是在[max_pid_in_direct, min_pid_in_reverse]之间取
      */
      float maxn = 0.0;
      for(unsigned possible_pid = max_pid_in_direct; possible_pid <= min_pid_in_reverse; possible_pid++){
        double s = _partition[possible_pid].size();
        float cnt = (1 - s / _partition_max_size[possible_pid]);
        if (cnt > maxn && _partition[possible_pid].size() < _partition_max_size[possible_pid]) {
          res = possible_pid;
          maxn = cnt;
        }
      }
    }else{
      /**
       * 如果最大正向邻居 > 最小反向邻居，那么取不到最优解，就只能在[min_pid_in_reverse, max_pid_in_direct]这之间去做取舍
      */
      float maxn = 0.0, direct_weight = 0.0, reverse_weight = 0.0;
      // 统计反向邻居在min_pid_in_reverse之后的数量
      for(unsigned p = min_pid_in_reverse; p < _partition_number; p++){
        auto iter = pcount_direct.find(p);
        if(iter == pcount_direct.end()){
          continue;
        }
        reverse_weight += iter->second;
      }
      // 统计正向邻居在min_pid_in_reverse之前的数量
      for(unsigned p = 0; p < min_pid_in_reverse; p++){
        auto iter = pcount_reverse.find(p);
        if(iter == pcount_reverse.end()){
          continue;
        }
        direct_weight += iter->second;
      }
      // 分别计算这个[min_pid_in_reverse, max_pid_in_direct]区间范围内各个取值的权重，取最大
      for(unsigned possible_pid = min_pid_in_reverse; possible_pid <= max_pid_in_direct; possible_pid++){
        // 增加direct权重
        auto iter1 = pcount_direct.find(possible_pid);
        if(iter1 != pcount_direct.end()){
          direct_weight += iter1->second;
        }
        double s = _partition[possible_pid].size();
        float cnt = (direct_weight + reverse_weight) * (1 - s / _partition_max_size[possible_pid]);
        if (cnt > maxn && _partition[possible_pid].size() < _partition_max_size[possible_pid]) {
          res = possible_pid;
          maxn = cnt;
        }
        // 减少reverse权重
        auto iter2 = pcount_reverse.find(possible_pid);
        if(iter2 != pcount_reverse.end()){
          reverse_weight -= iter2->second;
        }
      }
    }
    pcount_direct.clear();
    pcount_reverse.clear();
    // 如果没有找到合适的分区，则找一个空闲分区
    if (res == std::numeric_limits<unsigned int>::max()) {
      if(stat){
        #pragma omp atomic
        stat->select_free++;
      }
      SelectFree(res);
    }
    return res;
  }
};