#include "lsm/level_index.h"


namespace lsmidx{

template<typename T, typename TagT>
PQFlashIndexProxy<T, TagT>::PQFlashIndexProxy(diskann::Metric dist_metric, std::string working_dir, std::shared_ptr<AlignedFileReader> &reader, size_t dims, size_t merge_thresh, diskann::Parameters& paras_disk, int level, bool is_single_file_index, int num_threads){
    // 初始化基本参数
    this->type = ON_DISK_DISKANN;
    this->paras = paras_disk;
    this->merge_thresh = merge_thresh;
    this->level = level;
    this->dimension = dims;
    this->is_single_file_index = is_single_file_index;
    this->dist_metric = dist_metric;
    // 初始化索引
    this->index = std::make_shared<diskann::PQFlashIndex<T, TagT>>(dist_metric, reader, is_single_file_index, true/*enable tags*/);
    this->index_prefix = working_dir + '/' + lsmidx::config::leveln_index_names[level-1];
    // 加载数据
    int res = this->index->load(this->index_prefix.c_str(), num_threads);
    if(res != 0){
        diskann::cout << "Failed to load disk index" << std::endl;
        exit(-1);
    }
}

template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options){

  uint64_t search_L = options.search_L;
  uint64_t beamwidth = options.beamwidth;
  uint64_t k = options.K;
  std::vector<float> disk_result_dists(k);
  std::vector<TagT> disk_result_tags(k);
  this->cached_beam_search(query, k, search_L, disk_result_tags.data(), disk_result_dists.data(), beamwidth);

  for(unsigned i = 0; i < disk_result_tags.size(); i++){
      res.emplace_back(disk_result_tags[i], disk_result_dists[i]);
  }
}
template<typename T, typename TagT>
void PQFlashIndexProxy<T, TagT>::reload_index(const std::string &disk_index_prefix){
    std::string disk_index_data_path = disk_index_prefix + "_disk.index";
    std::string disk_pq_coord_path = disk_index_prefix + "_pq_compressed.bin";
    std::string disk_tag_path = disk_index_data_path + ".tags";
    this->index->reload_index(disk_index_data_path, disk_pq_coord_path, disk_tag_path);
}

template<typename T, typename TagT>
InMemIndexProxy<T, TagT>::InMemIndexProxy(diskann::Metric dist_metric, std::string working_dir, size_t dims, size_t merge_thresh, diskann::Parameters& paras_mem, bool is_single_file_index){
    // 初始化基本参数
    this->type = IN_MEM_DISKANN;
    this->paras = paras_mem;
    this->index_prefix = working_dir + '/' + lsmidx::config::mem_index_name;
    this->merge_thresh = merge_thresh;
    this->level = 0;
    this->dimension = dims;
    this->is_single_file_index = is_single_file_index;
    this->dist_metric = dist_metric;
    int num = lsmidx::config::mem_indexes_num;
    // 初始化index以及对应的状态数组
    for(int i = 0; i<num ; i++){
        this->indexes.emplace_back(std::make_shared<diskann::Index<T, TagT>>(
            dist_metric, dims, merge_thresh, true/*dynamic index*/, is_single_file_index, true /*enable tags*/
        ));
    }
    active_states.resize(num, std::atomic_bool(false));
    index_clearing_states.resize(num, std::atomic_bool(false));
    clear_locks.resize(num);

    // 将第一个mem_index激活
    bool expected_active = false;
    active_states[0].compare_exchange_strong(expected_active, true);
}
template<typename T, typename TagT>
void InMemIndexProxy<T, TagT>::KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options){
  uint64_t search_L = options.search_L;
  uint64_t k = options.K;
  //check each memory index - if non empty and not being currently cleared - search and get top K active tags 
  for(int i = 0;i<this->indexes.size();i++){
    if(this->index_clearing_states[this>current].load() == true){
        continue;
    }
    std::shared_lock<std::shared_timed_mutex> lock(this->clear_locks[i]);
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
int InMemIndexProxy<T, TagT>::Put(const WriteOptions& options, const VecSlice<T>& key, const TagT& tag){
    const T* point = key.data();

    // 确认目前所指向的index是否在工作
    if(this->active_states[this>current].load() == false){
        diskann::cout << "Active index indicated as mem_index_"<< this->current << "but it cannot accept insertions" << std::endl;
        return -1;
    }

    // 拿到目前指向的index并进行插入
    std::shared_ptr<diskann::Index<T, TagT>>& index = this->indexes[this->current];
    if(index->get_num_points() < index->return_max_points()){
        if(index->insert_point(point, this->paras, tag) != 0){
            diskann::cout << "Could not insert point with tag " << tag << std::endl;
            return -3;
        }
        return 0;
    }else{
        diskann::cout << "Capacity exceeded" << std::endl;
    }

}
template<typename T, typename TagT>
int InMemIndexProxy<T, TagT>::Switch(){
    // 获取下一个活动index
    int next_idx = this->GetNextSwitchIdx();

    // 如果说下一个index不为空，则进行clear
    if(this->active_states[next_idx].load() == false){
        if(this->indexes[next_idx]->get_num_points()>0){
            diskann::cout << "Initialised new index for mem_index_" << next_idx << std::endl;
            this->ClearIndex(next_idx);
        }
    }

    // 将下一个活动index从非激活态设置为激活态
    bool expected_active = false;
    this->active_states[next_idx].compare_exchange_strong(expected_active, true);

    // 将原本激活的index设置为非激活
    expected_active  = true;
    this->active_states[this->current].compare_exchange_strong(expected_active, false);

    // 切换当前指向的index
    int prev_idx = this->current;
    this->current = next_idx;

    return prev_idx;
}
template<typename T, typename TagT>
std::string InMemIndexProxy<T, TagT>::SaveIndex(int idx){
    std::string save_path;
    if(idx>=this->indexes.size()){
        return save_path;
    }
    if(this->active_states[idx].load() == true){
        return save_path;
    }
    save_path = this->index_prefix + "_" + idx;
    this->indexes[idx]->save(save_path);

    return save_path;
}
template<typename T, typename TagT>
void InMemIndexProxy<T, TagT>::ClearIndex(int idx){
    if(idx>=this->indexes.size()){
        return;
    }
    if(this->active_states[idx].load() == true){
        return;
    }
    // 先修改对应的clear状态
    bool expected_clearing = false;
    this->index_clearing_states[idx].compare_exchange_strong(expected_clearing, true);
    std::unique_lock<std::shared_timed_mutex> lock(this->clear_locks[idx]);

    // 进行clear
    this->indexes[idx].reset();
    this->indexes[idx] = std::make_shared<diskann::Index<T, TagT>>(this->dist_metric, this->dimension, this->merge_thresh * 2 , 1, this->is_single_file_index, 1);
    // 再把clear状态修改回去
    expected_clearing = true;
    this->index_clearing_states[idx].compare_exchange_strong(expected_clearing, false);

}
template<typename T, typename TagT>
int InMemIndexProxy<T, TagT>::GetCurrentNumPoints(){
    return this->indexes[this->current]->get_num_points();
}

template<typename T, typename TagT>
int InMemIndexProxy<T, TagT>::GetNextSwitchIdx(){
  return (this->current + 1) % this->indexes.size();
}

} // namespace lsmidx