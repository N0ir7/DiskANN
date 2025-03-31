#include "lsm/level/level_index.h"
#include "timer.h"
namespace lsmidx{
  template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::GetActiveTags(tsl::robin_set<TagT>& active_tags){
    tsl::robin_set<TagT> tags;
    this->index->get_active_tags(tags);
    for(auto& tag : tags){
        active_tags.insert(tag);
    }
    return;
}
template<typename T, typename TagT>
int PQFlashIndexProxy<T, TagT>::GetCurrentNumPoints(){
    // auto lock = this->GetReadLock();
    return this->index->return_nd();
}
template<typename T, typename TagT>
PQFlashIndexProxy<T, TagT>::PQFlashIndexProxy(diskann::Metric dist_metric, std::string working_dir, int idx, std::shared_ptr<AlignedFileReader> &reader, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_disk, int level, bool is_single_file_index, int num_threads):LevelIndex<T, TagT>(IndexType::ON_DISK_DISKANN, paras_disk, merge_thresh, level, dims, is_single_file_index, dist_metric),reader(reader){
    // 初始化基本参数
    this->index_prefix = working_dir + '/' + lsmidx::config::leveln_index_names[level] + "_" + std::to_string(idx);

    // 初始化索引
    this->index = std::make_shared<diskann::PQFlashIndex<T, TagT>>(dist_metric, reader, is_single_file_index, true/*enable tags*/);
    
    // 加载数据
    this->LoadIndex(this->index_prefix, num_threads);
}
template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::ReportQueryInfo(int query_num){
  diskann::cout << "【" << this->GetIndexPrefix()<< " 】";
  if(query_num <=0){
    diskann::cout << stat.ToString() << std::endl;
  }else{
    diskann::cout << stat.ToString(query_num) << std::endl;
  }
}
template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, diskann::QueryStats * stats){
  // auto lock = this->GetReadLock();
  if(this->index->return_nd() == 0){
    return;
  }
  
  diskann::Timer timer;
  uint64_t search_L = options.search_L;
  uint64_t beamwidth = options.beamwidth;
  uint64_t k = options.K;
  std::vector<float> disk_result_dists(k);
  std::vector<TagT> disk_result_tags(k);
  diskann::QueryStats tmp;
  // this->index->cached_beam_search(query, k, search_L, disk_result_tags.data(), disk_result_dists.data(), beamwidth, stats);
  this->index->cached_beam_search(query, k, search_L, disk_result_tags.data(), disk_result_dists.data(), beamwidth, &tmp);
  stats->Aggregate(tmp);
  stat.Aggregate(tmp);
  for(unsigned i = 0; i < disk_result_tags.size(); i++){
    res.emplace_back(disk_result_tags[i], disk_result_dists[i]);
  }
}
template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::ReloadIndex(const std::string &disk_index_prefix){
  // std::string disk_index_data_path = disk_index_prefix + "_disk.index";
  // std::string disk_pq_coord_path = disk_index_prefix + "_pq_compressed.bin";
  // std::string disk_tag_path = disk_index_data_path + ".tags";
  // 加载数据
  this->ClearIndex();
    
  this->LoadIndex(disk_index_prefix, 6);
    
  // this->index->reload_index(disk_index_data_path, disk_pq_coord_path, disk_tag_path);
  // this->delete_tag_set.Load(disk_index_prefix);
}
template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::LoadIndex(const std::string &disk_index_prefix, int num_threads){
  if(file_exists(disk_index_prefix + "_disk.index")){
    int res = this->index->load(disk_index_prefix.c_str(), num_threads);
    if(res != 0){
        diskann::cout << "Failed to load disk index" << std::endl;
        exit(-1);
    }
    uint32_t node_cache_count = 1 + (uint32_t) round(this->index->return_nd() * 0.01);
    node_cache_count = node_cache_count > PQ_FLASH_INDEX_MAX_NODES_TO_CACHE
                           ? PQ_FLASH_INDEX_MAX_NODES_TO_CACHE
                           : node_cache_count;
    std::vector<uint32_t> cache_node_list;

    // 生成要缓存的 BFS 层级节点列表，并加载这些缓存节点
    // 节点的邻居信息存储到nhood_cache中，并将节点的坐标存储到coord_cache
    this->index->cache_bfs_levels(node_cache_count,
                                       cache_node_list);
    this->index->load_cache_list(cache_node_list);
  }
  this->delete_tag_set.Load(disk_index_prefix);
}
template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::ClearIndex(){
    // auto lock = this->GetWriteLock();
    // auto delete_lock = this->GetWriteDeleteLock();
    // 进行clear
    this->index.reset();
    this->index = std::make_shared<diskann::PQFlashIndex<T, TagT>>(this->dist_metric, this->reader, this->is_single_file_index, true/*enable tags*/);
    this->delete_tag_set.Clear();
}
template<typename T, typename TagT>
std::shared_ptr<diskann::PQFlashIndex<T, TagT>> PQFlashIndexProxy<T, TagT>::GetIndex(){
    return index;
}
template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::GetMedoids(std::vector<TagT>& medoid_vec){
  auto pair = this->index->get_medoid();
  auto medoid_ptr = pair.first;
  auto medoid_num = pair.second;
  for(size_t i = 0; i < medoid_num; i++){
    medoid_vec.emplace_back(medoid_ptr[i]);
  }
}
template<typename T, typename TagT>
std::string PQFlashIndexProxy<T, TagT>::ReportIndexInfo(){
    std::ostringstream oss;
    tsl::robin_set<TagT> tags;
    this->index->get_active_tags(tags);
    oss << '[';
    oss << "prefix: " << this->GetIndexPrefix() << "; ";
    oss << "Number of points: " << this->index->return_nd() << "; ";
    oss << "Active tags: " << tags.size() << "; ";
    oss << ']';
    return oss.str();
}
// template class instantiations
template class PQFlashIndexProxy<float, uint32_t>;
template class PQFlashIndexProxy<uint8_t, uint32_t>;
template class PQFlashIndexProxy<int8_t, uint32_t>;
template class PQFlashIndexProxy<float, int64_t>;
template class PQFlashIndexProxy<uint8_t, int64_t>;
template class PQFlashIndexProxy<int8_t, int64_t>;
template class PQFlashIndexProxy<float, uint64_t>;
template class PQFlashIndexProxy<uint8_t, uint64_t>;
template class PQFlashIndexProxy<int8_t, uint64_t>;
} // namespace lsmidx