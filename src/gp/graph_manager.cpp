#include "gp/graph_manager.h"
#include "lsm/options.h"
#include <map>
// #define SECTORS_PER_MERGE (uint64_t) 4096
namespace lsmidx{
GraphManager::GraphManager(std::string index_file, std::vector<std::vector<unsigned>>& partition, std::unordered_map<unsigned, unsigned>& id2pid){
  load_disk_index<float>(index_file, partition, id2pid);
}
template <typename T>
void GraphManager::load_disk_index(std::string index_name, std::vector<std::vector<unsigned>>& partition, std::unordered_map<unsigned, unsigned>& id2pid) {
  std::cout << "loading disk index file: " << index_name << "... " << std::endl;
  std::ifstream in;
  
  _u64 expected_npts;
  auto metas = get_disk_index_meta(index_name);

  expected_npts = metas[0];
  _nd = expected_npts;
  _dim = metas[1];

  _max_node_len = metas[3];
  _nnodes_per_sector = metas[4];
  _nnodes_per_partition = _nnodes_per_sector * SECTORS_PER_MERGE;
  _partition_number = ROUND_UP(_nd, _nnodes_per_partition) / _nnodes_per_partition;

  char * merge_buf = nullptr;
  diskann::alloc_aligned((void **) &merge_buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);
  full_graph.resize(_nd);
  partition.resize(_partition_number);
  _u64 des = 0;
  for (unsigned i = 0; i < _partition_number; i++) {
    in.open(index_name, std::ios::binary);
    in.seekg(SECTOR_LEN, std::ios::beg);
    in.read(merge_buf, SECTORS_PER_MERGE * SECTOR_LEN);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(16)
    for (_u32 k = 0; k < SECTORS_PER_MERGE; k++) {
      char *sector_buf = merge_buf + (k * SECTOR_LEN);
      for (unsigned j = 0; j < _nnodes_per_sector; j++) {
        unsigned cur_node_id = i * _nnodes_per_partition + k * _nnodes_per_sector + j;
        if((cur_node_id) >= _nd){
          continue;
        }
        char * node_buf = sector_buf + j * _max_node_len;
        unsigned &nnbr = *(unsigned *)(node_buf + _dim * sizeof(T));
        unsigned *nhood_buf = (unsigned *)(node_buf + (_dim * sizeof(T)) + sizeof(unsigned));
        std::vector<unsigned> tmp(nnbr);
        des += nnbr;
        memcpy((char *)tmp.data(), nhood_buf, nnbr * sizeof(unsigned));
        full_graph[cur_node_id].assign(tmp.begin(), tmp.end());
        
        #pragma omp critical
        {
          partition[i].emplace_back(cur_node_id);
          id2pid[cur_node_id] = i;
        }
      }
    }
  }
  in.close();
  reverse_graph.resize(_nd);
  std::vector<std::mutex> ms(_nd);
#pragma omp parallel for shared(reverse_graph, full_graph)
  for (unsigned i = 0; i < _nd; i++) {
    for (unsigned j = 0; j < full_graph[i].size(); j++) {
      std::lock_guard<std::mutex> lock(ms[full_graph[i][j]]);
      reverse_graph[full_graph[i][j]].emplace_back(i);
    }
  }
  std::cout << "avg degree: " << (double)des / _nd << std::endl;
  diskann::aligned_free(merge_buf);
  std::cout << "_nd: " << _nd << " _dim:" << _dim << " _nnodes_per_sector:" << _nnodes_per_sector << " pn:" << _partition_number << std::endl;
  std::cout << "load index over." << std::endl;
}
void GraphManager::graph_degree_statistic() {
  size_t n = full_graph.size();
  std::vector<unsigned> out_degrees(n), in_degrees(n);

  // 收集出度和入度
  for (size_t i = 0; i < n; ++i) {
    out_degrees[i] = full_graph[i].size();
    in_degrees[i] = reverse_graph[i].size();
  }

  // 计算平均和最大值
  auto calc_stats = [](const std::vector<unsigned>& degrees,
                       const std::string& name) {
    std::vector<unsigned> sorted = degrees;
    std::sort(sorted.begin(), sorted.end());

    unsigned max_degree = sorted.back();
    unsigned min_degree = sorted.front();
    double avg_degree = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();

    auto get_percentile = [&](double p) {
      size_t idx = static_cast<size_t>(p * sorted.size());
      if (idx >= sorted.size()) idx = sorted.size() - 1;
      return sorted[idx];
    };

    std::cout << "=== " << name << " degree statistics ===" << std::endl;
    std::cout << "Max: " << max_degree << std::endl;
    std::cout << "Min: " << min_degree << std::endl;
    std::cout << "Average: " << avg_degree << std::endl;
    std::cout << "P10: " << get_percentile(0.10) << std::endl;
    std::cout << "P30: " << get_percentile(0.30) << std::endl;
    std::cout << "P50: " << get_percentile(0.50) << std::endl;
    std::cout << "P90: " << get_percentile(0.90) << std::endl;
    std::cout << "P99: " << get_percentile(0.99) << std::endl;
    std::cout << std::endl;
  };

  // 输出入度和出度统计
  calc_stats(in_degrees, "In");
  calc_stats(out_degrees, "Out");
}
}