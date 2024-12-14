#include "lsm/level_index.h"


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
PQFlashIndexProxy<T, TagT>::PQFlashIndexProxy(diskann::Metric dist_metric, std::string working_dir, std::shared_ptr<AlignedFileReader> &reader, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_disk, int level, bool is_single_file_index, int num_threads):LevelIndex<T, TagT>(IndexType::ON_DISK_DISKANN, paras_disk, merge_thresh, level, dims, is_single_file_index, dist_metric){
    // 初始化基本参数
    this->index_prefix = working_dir + '/' + lsmidx::config::leveln_index_names[level-1];

    // 初始化索引
    this->index = std::make_shared<diskann::PQFlashIndex<T, TagT>>(dist_metric, reader, is_single_file_index, true/*enable tags*/);
    
    // 加载数据
    if(file_exists(this->index_prefix + "_disk.index")){
        int res = this->index->load(this->index_prefix.c_str(), num_threads);
        if(res != 0){
            diskann::cout << "Failed to load disk index" << std::endl;
            exit(-1);
        }
    }
}

template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, diskann::QueryStats * stats){

  uint64_t search_L = options.search_L;
  uint64_t beamwidth = options.beamwidth;
  uint64_t k = options.K;
  std::vector<float> disk_result_dists(k);
  std::vector<TagT> disk_result_tags(k);
  this->index->cached_beam_search(query, k, search_L, disk_result_tags.data(), disk_result_dists.data(), beamwidth, stats);

  for(unsigned i = 0; i < disk_result_tags.size(); i++){
      res.emplace_back(disk_result_tags[i], disk_result_dists[i]);
  }
}
template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::ReloadIndex(const std::string &disk_index_prefix){
    std::string disk_index_data_path = disk_index_prefix + "_disk.index";
    std::string disk_pq_coord_path = disk_index_prefix + "_pq_compressed.bin";
    std::string disk_tag_path = disk_index_data_path + ".tags";
    this->index->reload_index(disk_index_data_path, disk_pq_coord_path, disk_tag_path);
}

template<typename T, typename TagT>
void InMemIndexProxy<T, TagT>::GetActiveTags(tsl::robin_set<TagT>& active_tags){
    tsl::robin_set<TagT> tags;
    for(auto& index : this->indexes){
        index->get_active_tags(tags);
        for(auto& tag : tags){
            active_tags.insert(tag);
        }
    }
    return;
}
template<typename T, typename TagT>
InMemIndexProxy<T, TagT>::InMemIndexProxy(diskann::Metric dist_metric, std::string working_dir, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_mem, bool is_single_file_index):LevelIndex<T, TagT>(IndexType::IN_MEM_DISKANN, paras_mem, merge_thresh, 0, dims, is_single_file_index, dist_metric), switching_disk_prefixes(false){
    // 初始化基本参数
    this->index_prefix = working_dir + '/' + lsmidx::config::mem_index_name;
    int num = lsmidx::config::mem_indexes_num;
    // 初始化index以及对应的状态数组
    for(int i = 0; i<num ; i++){
        this->indexes.emplace_back(std::make_shared<diskann::Index<T, TagT>>(
            dist_metric, dims, merge_thresh * 2, true/*dynamic index*/, is_single_file_index, true /*enable tags*/
        ));
    }
    this->active_states.reserve(num);
    this->index_clearing_states.reserve(num);
    this->clear_locks.reserve(num);
    for(int i = 0; i < num; i++){
        this->active_states.emplace_back(std::make_unique<std::atomic_bool>(false));
        this->index_clearing_states.emplace_back(std::make_unique<std::atomic_bool>(false));
        this->clear_locks.emplace_back(std::make_unique<std::shared_timed_mutex>());
    }
    /**
     * 尝试加载数据
    */
    for (int i = 0; i < num ; i++){
        std::string prefix = this->index_prefix + "_" + std::to_string(i);
        if(file_exists(prefix)){
            this->indexes[i]->load(prefix.c_str());
        }
    }
    // 将第一个mem_index激活
    bool expected_active = false;
    this->active_states[0]->compare_exchange_strong(expected_active, true);
    this->current = 0;
}
template<typename T, typename TagT>
void InMemIndexProxy<T, TagT>::KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, [[maybe_unused]] diskann::QueryStats * stats){
  uint64_t search_L = options.search_L;
  uint64_t k = options.K;
  //check each memory index - if non empty and not being currently cleared - search and get top K active tags 
  for(size_t i = 0;i<this->indexes.size();i++){
    if(this->index_clearing_states[this->current]->load() == true){
        continue;
    }
    std::shared_lock<std::shared_timed_mutex> lock(*(this->clear_locks[i]));
    std::shared_ptr<diskann::Index<T, TagT>>& index = this->indexes[i];
    if(index->get_num_points() > 0){
        index->search(query, (uint32_t)k, (uint32_t)search_L, res);
    }
  }

}

template<typename T, typename TagT>
void InMemIndexProxy<T, TagT>::LazyDelete(TagT tag){
    std::shared_ptr<diskann::Index<T, TagT>>& index = this->indexes[this->current];
    index->lazy_delete(tag);
}

template<typename T, typename TagT>
int InMemIndexProxy<T, TagT>::Put([[maybe_unused]] const WriteOptions& options, const VecSlice<T>& key, const TagT& tag){
    const T* point = key.data();

    // 确认目前所指向的index是否在工作
    if(this->active_states[this->current]->load() == false){
        diskann::cout << "Active index indicated as mem_index_"<< this->current << "but it cannot accept insertions" << std::endl;
        return -1;
    }

    // 拿到目前指向的index并进行插入
    std::shared_ptr<diskann::Index<T, TagT>>& index = this->indexes[this->current];
    if(index->get_num_points() < index->return_max_points()){
        if(index->insert_point(point, *this->paras.get(), tag) != 0){
            diskann::cout << "Could not insert point with tag " << tag << std::endl;
            return -3;
        }
        return 0;
    }else{
        diskann::cout << "Capacity exceeded" << std::endl;
    }
    return 0;
}
template<typename T, typename TagT>
int InMemIndexProxy<T, TagT>::Switch(){
    // 获取下一个活动index
    int next_idx = this->GetNextSwitchIdx();

    // 如果说下一个index不为空，则进行clear
    if(this->active_states[next_idx]->load() == false){
        if(this->indexes[next_idx]->get_num_points()>0){
            diskann::cout << "Initialised new index for mem_index_" << next_idx << std::endl;
            this->ClearIndex(next_idx);
        }
    }

    // 将下一个活动index从非激活态设置为激活态
    bool expected_active = false;
    this->active_states[next_idx]->compare_exchange_strong(expected_active, true);

    // 将原本激活的index设置为非激活
    expected_active  = true;
    this->active_states[this->current]->compare_exchange_strong(expected_active, false);

    // 切换当前指向的index
    int prev_idx = this->current;
    this->current = next_idx;

    return prev_idx;
}
template<typename T, typename TagT>
std::string InMemIndexProxy<T, TagT>::SaveIndex(size_t idx){
    std::string save_path;
    if(idx>=this->indexes.size()){
        return save_path;
    }
    if(this->active_states[idx]->load() == true){
        return save_path;
    }
    save_path = this->index_prefix + "_" + std::to_string(idx);
    this->indexes[idx]->save(save_path.c_str());

    return save_path;
}
template<typename T, typename TagT>
void InMemIndexProxy<T, TagT>::ClearIndex(size_t idx){
    if(idx>=this->indexes.size()){
        return;
    }
    if(this->active_states[idx]->load() == true){
        return;
    }
    // 先修改对应的clear状态
    bool expected_clearing = false;
    this->index_clearing_states[idx]->compare_exchange_strong(expected_clearing, true);
    std::unique_lock<std::shared_timed_mutex> lock(*(this->clear_locks[idx]));

    // 进行clear
    this->indexes[idx].reset();
    this->indexes[idx] = std::make_shared<diskann::Index<T, TagT>>(this->dist_metric, this->dimension, this->merge_thresh * 2 , 1, this->is_single_file_index, 1);
    // 再把clear状态修改回去
    expected_clearing = true;
    this->index_clearing_states[idx]->compare_exchange_strong(expected_clearing, false);

}
template<typename T, typename TagT>
int InMemIndexProxy<T, TagT>::GetCurrentNumPoints(){
    return this->indexes[this->current]->get_num_points();
}

template<typename T, typename TagT>
int InMemIndexProxy<T, TagT>::GetNextSwitchIdx(){
  return (this->current + 1) % this->indexes.size();
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

  // template class instantiations
  template class InMemIndexProxy<float, uint32_t>;
  template class InMemIndexProxy<uint8_t, uint32_t>;
  template class InMemIndexProxy<int8_t, uint32_t>;
  template class InMemIndexProxy<float, int64_t>;
  template class InMemIndexProxy<uint8_t, int64_t>;
  template class InMemIndexProxy<int8_t, int64_t>;
  template class InMemIndexProxy<float, uint64_t>;
  template class InMemIndexProxy<uint8_t, uint64_t>;
  template class InMemIndexProxy<int8_t, uint64_t>;
} // namespace lsmidx