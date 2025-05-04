#pragma once
#include <vector>
#include "utils.h"

namespace lsmidx{
inline size_t get_file_size(const std::string &fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    size_t end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    std::cout << "Could not open file: " << fname << std::endl;
    return 0;
  }
}
inline std::vector<_u64> get_disk_index_meta(const std::string &path) {
  std::ifstream fin(path, std::ios::binary);

  int meta_n, meta_dim;
  std::vector<_u64> metas;

  fin.read((char *)(&meta_n), sizeof(int));
  fin.read((char *)(&meta_dim), sizeof(int));

  metas.resize(meta_n);
  fin.read((char *)(metas.data()), sizeof(_u64) * meta_n);
  // }
  fin.close();
  return metas;
}
class GraphManager{
public:
  GraphManager(std::string index_file, std::vector<std::vector<unsigned>>& partition, std::unordered_map<unsigned, unsigned>& id2pid);
  template <typename T>
  void load_disk_index(std::string index_name, std::vector<std::vector<unsigned>>& _partition, std::unordered_map<unsigned, unsigned>& id2pid);
  void graph_degree_statistic();
  /**
   * graph info
  */
  // vector dimension
  size_t _dim;
  // vector number
  _u64 _nd;
  _u64 _max_node_len;
  _u64 _nnodes_per_sector;
  _u64 _partition_number;
  _u64 _nnodes_per_partition;
  // max out-degree
  unsigned _width;
  // seed vertex id
  unsigned _ep;
  // neighbor list
  // std::vector<std::vector<unsigned>> direct_graph;
  std::vector<std::vector<unsigned>> full_graph;
  std::vector<std::vector<unsigned>> reverse_graph;
  /**
   * other tool util
  */
  
};
}; // namesp lsmidx