#include "gp/graph_partitioner.h"
#include "linux_aligned_file_reader.h"
#include <map>
#include <iomanip>

namespace lsmidx{
GraphPartitioner::GraphPartitioner(std::string index_file):index_file(index_file){
  this->gm = std::make_shared<GraphManager>(index_file, _partition, id2pid);
  this->Init();
}
GraphPartitioner::GraphPartitioner(std::shared_ptr<GraphPartitioner> other){
  this->gm = other->gm;
  this->_partition_number = other->_partition_number;
  this->_partition_max_size = other->_partition_max_size;
  this->pmutex = std::move(other->pmutex);
  this->_lock_nodes = std::move(other->_lock_nodes);
  this->_lock_pids = std::move(other->_lock_pids);
  this->_partition = std::move(other->_partition);
  this->id2pid = std::move(other->id2pid);
  this->nnodes_per_sector = this->gm->_nnodes_per_sector;
  this->max_node_len = this->gm->_max_node_len;
  this->data_dim = this->gm->_dim;
  this->bp = std::move(other->bp);
}
template<typename T>
void GraphPartitioner::InsertPointIntoNewIndex( std::ofstream& writer, std::vector<unsigned>& id_map, unsigned start_id){
  diskann::DiskNode<T> node = bp->get_node<T>(start_id);
  _u64 nnbrs = node.nnbrs;
  char *node_disk_buf = (char *)node.coords;
  unsigned *node_nbrs = node.nbrs;
  for (_u64 m = 0; m < nnbrs; ++m) { // 更新所有邻居的ID
    unsigned id = node_nbrs[m];
    node_nbrs[m] = id_map[id];
  }
  /**
   * 将点写入到新索引文件中
  */
 assert(node_disk_buf != nullptr);
  writer.write(node_disk_buf, max_node_len);
  bp->unpin(start_id);
}
template<typename T,typename TagT>
void GraphPartitioner::RedistributeIndex(std::string from_index_prefix, std::string output_index_prefix){
  std::string from_pq_compressed_file = from_index_prefix + "_pq_compressed.bin";
  std::string from_index_name = from_index_prefix + "_disk.index";
  std::string from_tags_file = from_index_name + ".tags";

  std::string output_index_name = output_index_prefix + "_disk.index";
  std::string output_pq_compressed_file = output_index_prefix + "_pq_compressed.bin";
  std::string output_tags_file = output_index_name + ".tags";
  // this->reader->open(from_index_name, false, false);
  if(!this->bp){
    this->bp = std::make_unique<ReadOnlySectorBufferPool>();
  }
  this->bp->InitIndexReader(from_index_name);
  this->bp->RegisterReaderThread();
  std::ofstream data_writer(output_index_name, std::ios::binary);
  assert(data_writer.is_open());
  int npts = this->nd();
  /**
   * 建立老id至新id的映射
   */
  unsigned new_pt_id = 0;
  std::vector<unsigned> new_ids(npts);
  for(int i = 0; i < _partition.size(); i++){
    for(int j = 0; j < _partition[i].size(); j++){
      new_ids[_partition[i][j]] = new_pt_id++;
    }
  }
  if(new_pt_id != npts){
    std::cout << "Not all points are properly allocated! With new_pt_id = " << new_pt_id << ", while total npts = " << npts << std::endl;
  }
  /**
   * 重写disk file
  */
  int sector_num = 0, k = 0;
  for(int i = 0; i < _partition.size(); i++){
    for(int j = 0; j < _partition[i].size(); j++){
      unsigned old_id = _partition[i][j];
      if(k % gm->_nnodes_per_sector == 0){
        sector_num++;
        data_writer.seekp(sector_num * (uint64_t)SECTOR_LEN, std::ios::beg);
        k = 0;
        if(sector_num >0 && sector_num % 1000 == 0){
          std::cout<< "npts: " << sector_num * gm->_nnodes_per_sector<< '/' << nd() << std::endl;
        }
      }
      InsertPointIntoNewIndex<T>(data_writer, new_ids, old_id);
      k++;
    }
  }
  this->bp->reader->deregister_all_threads();
  // this->reader->close();
  // 复制数据文件的元信息
  char * buf = nullptr;
  diskann::alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
  std::ifstream srcFile(from_index_name, std::ios::binary);
  srcFile.seekg(0, std::ios::beg);
  srcFile.read(buf, SECTOR_LEN);
  assert(data_writer.is_open());
  data_writer.seekp(0, std::ios::beg);
  data_writer.write(buf, SECTOR_LEN);
  data_writer.close();
  srcFile.close();
  diskann::aligned_free(buf);
  /**
   * 重写pq_compressed file
  */
  uint8_t* pq_data;
  size_t pq_nchunks;
  size_t pq_num;
  diskann::load_bin<uint8_t>(from_pq_compressed_file, pq_data, pq_num, pq_nchunks);
  uint8_t* new_pq_data = nullptr;
  diskann::alloc_aligned((void **) &new_pq_data, npts * pq_nchunks * sizeof(uint8_t), pq_nchunks);
  for(int i = 0; i < npts; i++){
    uint8_t* cur_dest_ptr = new_pq_data + new_ids[i] * pq_nchunks;
    uint8_t* cur_src_ptr = pq_data + i * pq_nchunks;
    memcpy(cur_dest_ptr, cur_src_ptr, pq_nchunks * sizeof(uint8_t));
  }
  diskann::save_bin<uint8_t>(output_pq_compressed_file, new_pq_data, npts, pq_nchunks);
  diskann::aligned_free(new_pq_data);
  delete[] pq_data;
  /**
   * 重写tag文件
  */
  TagT* tag_data;
  uint64_t tag_num, tag_dim;
  diskann::load_bin<TagT>(from_tags_file, tag_data, tag_num, tag_dim);
  TagT* new_tag_data = nullptr;
  diskann::alloc_aligned((void **) &new_tag_data, npts * sizeof(TagT), sizeof(TagT));
  for(int i = 0; i < npts; i++){
    TagT* cur_dest_ptr = new_tag_data + new_ids[i];
    TagT* cur_src_ptr = tag_data + i;
    memcpy(cur_dest_ptr, cur_src_ptr, sizeof(TagT));
  }
  diskann::save_bin<TagT>(output_tags_file, new_tag_data, npts, 1);
  diskann::aligned_free(new_tag_data);
  delete[] tag_data;
}
GraphPartitioner::GraphPartitioner(std::shared_ptr<GraphManager> gm):gm(gm){
  this->Init();
}
void GraphPartitioner::Init(){
  _u64 npts = this->nd();
  /**
   * Initialize partition related variables
  */
  this->_partition_max_size.resize(this->_partition.size());
  for(int i = 0; i < this->_partition.size(); i++){
    this->_partition_max_size[i] = this->_partition[i].size();
  }
  this->_partition_number = this->gm->_partition_number;
  this->nnodes_per_sector = this->gm->_nnodes_per_sector;
  this->max_node_len = this->gm->_max_node_len;
  this->data_dim = this->gm->_dim;
  for (unsigned i = 0; i < _partition_number; i++) {
    this->pmutex.push_back(std::make_unique<std::mutex>());
  }
  this->_lock_pids.clear();
  this->_lock_pids.resize(_partition_number, false);
  /**
   * Initialize npt related variables
  */
  this->_lock_nodes.clear();
  this->_lock_nodes.resize(npts, false);
}
void GraphPartitioner::Lock(std::vector<unsigned> & init_stream, int lock_npts, std::unordered_set<unsigned>* vis){
  if(lock_npts == 0){
    return;
  }
  /**
   * 将访问顺序的前lock_npts进行锁定
  */
  for(size_t i = 0; i < init_stream.size(); i++){
    if(lock_npts <= 0){
      break;
    }
    if(vis && !vis->count(i)){
      std::cout << "Error: a init lock point "<< i <<" is not allocated" << std::endl;
      return;
    }
    // 锁定对应点及partition
    _lock_nodes[i] = true;
    auto pid = id2pid[i];
    _lock_pids[pid] = true;
    lock_npts--;
  }
  // 锁定对应partition上的其他点
  int locked_partition_num = 0;
  int locked_nodes_num = 0;
  for (unsigned i = 0; i < _partition_number; i++) {
    if (!_lock_pids[i]) break;
    for (unsigned s : _partition[i]) {
      _lock_nodes[s] = true;
    }
    locked_nodes_num += _partition[i].size();
    locked_partition_num++;
  }
  std::cout << "finally, it locks partition nums: " << locked_partition_num << " locks nodes num: " << locked_nodes_num << std::endl;
}
bool GraphPartitioner::InsertPointIntoPartition(unsigned pid, unsigned nid){
  if(_partition[pid].size() >= _partition_max_size[pid]){
    return false;
  }
  _partition[pid].emplace_back(nid);
  id2pid[nid] = pid;
  return true;
}
bool GraphPartitioner::InsertPointIntoPartitionRaw(unsigned pid, unsigned nid){
  if(_partition[pid].size() >= _partition_max_size[pid]){
    return false;
  }
  _partition[pid].emplace_back(nid);
  return true;
}
bool GraphPartitioner::SelectFree(unsigned& pid){
  for(int i = 0; i < this->_partition.size(); i++){
    if(this->_partition[i].size() < this->_partition_max_size[i]){
      pid = i;
      return true;
    }
  }
  return false;
}
void print_ror_stats(std::vector<float>& tmp, int partition_no, int round, std::string algo, float time_in_ms) {
  std::sort(tmp.begin(), tmp.end());

  float min_val = tmp.front();
  float max_val = tmp.back();
  float avg_val = std::accumulate(tmp.begin(), tmp.end(), 0.0f) / tmp.size();

  auto get_percentile = [&](float percentile) {
    size_t idx = static_cast<size_t>(percentile * tmp.size());
    if (idx >= tmp.size()) idx = tmp.size() - 1;
    return tmp[idx];
  };

  std::cout << "\n[ algo: " << algo << ", round: " << round <<", partition no: " << partition_no << "]" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Min ROR: " << min_val << std::endl;
  std::cout << "Max ROR: " << max_val << std::endl;
  std::cout << "Average ROR: " << avg_val << std::endl;

  std::cout << "ROR Percentiles:" << std::endl;
  std::cout << "  0.10: " << get_percentile(0.10f) << std::endl;
  std::cout << "  0.30: " << get_percentile(0.30f) << std::endl;
  std::cout << "  0.50: " << get_percentile(0.50f) << std::endl;
  std::cout << "  0.90: " << get_percentile(0.90f) << std::endl;
  std::cout << "  0.95: " << get_percentile(0.95f) << std::endl;
  std::cout << "  0.99: " << get_percentile(0.99f) << std::endl;
  /**
   * Output csv
  */
  std::string log_prefix = "/home/hlqiu/LSMIndex/log/lsmidx_merge_insert_delete";
  std::string log_file_path = log_prefix + "_redistribute.csv";
  bool is_new_file = !file_exists(log_file_path);  // 判断是否是新文件

  std::ofstream log_file(log_file_path, std::ios::app);
  if (log_file.is_open()) {
      // 只有新文件才写入表头
      if (is_new_file) {
          log_file << "Algo,round,partition no,cost time,Min,Max,Avg,p10,p30,p50,p90,p95,p99" << std::endl;
      }

      // 追加数据
      log_file << algo << "," << round << "," << partition_no << "," << time_in_ms
      << "," << min_val << "," << max_val << "," << avg_val
      << "," << get_percentile(0.10f) << "," << get_percentile(0.30f) << "," << get_percentile(0.50f)
      << "," << get_percentile(0.90f) << "," << get_percentile(0.95f) << "," << get_percentile(0.99f) << std::endl;

      log_file.close();
  } else {
      std::cerr << "Failed to open delete log file!" << std::endl;
  }
}
/**
 * count the id overlap according to the graph partitioning
 */
void GraphPartitioner::partition_statistic(std::string algo, int round, float time_in_ms) {
  auto npts = this->nd();
  auto& reverse_graph = gm->reverse_graph;
  std::vector<float> relaxed_overlap_ratio(npts, 0);

  std::unordered_set<unsigned> cur_set; // 该分区中所有邻居节点的集合
  for (int i = _partition_number-1; i >= 0; i--) {
    for (size_t j = 0; j < _partition[i].size(); j++) {
      cur_set.insert(_partition[i][j]);
    }
    #pragma omp parallel for schedule(dynamic, 1) num_threads(16)
    for (size_t j = 0; j < _partition[i].size(); j++) {
      unsigned id = _partition[i][j];
      if(reverse_graph[id].size() == 0){
        relaxed_overlap_ratio[id] = 1;
        continue;
      }
      for (unsigned &x : reverse_graph[id]) {
        if(cur_set.find(x) == cur_set.end()){
          assert(id2pid[x]<id2pid[id]);
          continue;
        }
        relaxed_overlap_ratio[id]++;
      }
      relaxed_overlap_ratio[id] /= reverse_graph[id].size();
    }
    // 针对当前 partition 提取对应 ROR 并打印
    std::vector<float> partition_ror;
    for (unsigned id : _partition[i]) {
      partition_ror.push_back(relaxed_overlap_ratio[id]);
    }
    print_ror_stats(partition_ror, i, round, algo, time_in_ms);
  }
  /**
   * 输出一些统计信息
  */
  // 最后输出整体统计信息
  print_ror_stats(relaxed_overlap_ratio, -1, round, algo, time_in_ms);
}
void GraphPartitioner::re_id2pid() {
  id2pid.clear();
  for (unsigned i = 0; i < _partition_number; i++) {
    for (unsigned j = 0; j < _partition[i].size(); j++) {
      id2pid[_partition[i][j]] = i;
    }
  }
}
template void GraphPartitioner::RedistributeIndex<float, uint32_t>(std::string, std::string);
template void GraphPartitioner::RedistributeIndex<uint8_t, uint32_t>(std::string, std::string);
template void GraphPartitioner::RedistributeIndex<int8_t, uint32_t>(std::string, std::string);
template void GraphPartitioner::RedistributeIndex<float, int64_t>(std::string, std::string);
template void GraphPartitioner::RedistributeIndex<uint8_t, int64_t>(std::string, std::string);
template void GraphPartitioner::RedistributeIndex<int8_t, int64_t>(std::string, std::string);
template void GraphPartitioner::RedistributeIndex<float, uint64_t>(std::string, std::string);
template void GraphPartitioner::RedistributeIndex<uint8_t, uint64_t>(std::string, std::string);
template void GraphPartitioner::RedistributeIndex<int8_t, uint64_t>(std::string, std::string);
}