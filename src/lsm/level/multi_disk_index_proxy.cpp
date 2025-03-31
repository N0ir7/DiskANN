#include "lsm/level/level_index.h"
#include "linux_aligned_file_reader.h"
namespace lsmidx{
  template<typename T, typename TagT>
void MultiPQFlashIndexProxy<T, TagT>::GetActiveTags(tsl::robin_set<TagT>& active_tags){
    // auto delete_lock = this->GetReadDeleteLock();
    size_t size = this->indexes.size();
    // acquire locks
    std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
    std::vector<int> lock_idx_vec;
    for(size_t i = 0; i < size; i++){
        if(this->active_states[i]->load() == false){
            continue;
        }
        lock_vec.emplace_back(this->GetSubReadLock(i));
        lock_idx_vec.emplace_back(i);
    }
    // get result
    std::vector<tsl::robin_set<TagT>> local_res;
    local_res.resize(size);
    for(auto i : lock_idx_vec){
        this->indexes[i]->GetActiveTags(local_res[i]);
    }
    tsl::robin_set<TagT> filter_res;
    for(auto i : lock_idx_vec){
        if(local_res[i].empty()){
            continue;
        }
        // first filter delete set
        this->indexes[i]->FilterDeletedTags(filter_res);
        // then add cur res
        filter_res.insert(local_res[i].begin(), local_res[i].end());
    }
    active_tags.insert(filter_res.begin(), filter_res.end());
    return;
}
template<typename T, typename TagT>
void MultiPQFlashIndexProxy<T, TagT>::SetReader(){
    int num = lsmidx::config::level0_indexes_num;
    this->readers.reserve(num);
    for(int i = 0; i < num ; i++){
        #ifdef _WINDOWS
        #ifndef USE_BING_INFRA
            this->readers.emplace_back(new WindowsAlignedFileReader());
        #else
           this->readers.emplace_back(new diskann::BingAlignedFileReader());
        #endif
        #else
            this->readers.emplace_back(new LinuxAlignedFileReader());
        #endif
    }
}
template<typename T, typename TagT>
int MultiPQFlashIndexProxy<T, TagT>::GetCurrentNumPoints(){
    int sum = 0;
    size_t size = this->indexes.size();
    std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
    std::vector<int> lock_idx_vec;
    for(size_t i = 0; i < size; i++){
        if(this->active_states[i]->load() == false){
            continue;
        }
        lock_vec.emplace_back(this->GetSubReadLock(i));
        lock_idx_vec.emplace_back(i);
    }
    for(auto i : lock_idx_vec){
        sum += this->indexes[i]->GetCurrentNumPoints();
    }
    return sum;
}
template<typename T, typename TagT>
MultiPQFlashIndexProxy<T, TagT>::MultiPQFlashIndexProxy(diskann::Metric dist_metric, std::string working_dir, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_disk, int level, bool is_single_file_index, int num_threads):MultiIndex<T, TagT>(IndexType::ON_DISK_DISKANN, paras_disk, merge_thresh, level, dims, is_single_file_index, dist_metric){
    this->SetReader();
    this->num_threads_for_search = num_threads;
    // 初始化基本参数
    this->index_prefix = working_dir + '/' + lsmidx::config::leveln_index_names[0];
    int num = lsmidx::config::level0_indexes_num;
    // 初始化索引
    // this->locks.reserve(num);
    this->active_states.reserve(num);
    for(int i = 0; i < num; i++){
        /**
         * 初始化各种状态量
        */
        this->indexes.emplace_back(std::make_shared<lsmidx::PQFlashIndexProxy<T, TagT>>(dist_metric, working_dir, i, readers[i], dims, merge_thresh, paras_disk, 0, is_single_file_index, num_threads));
        // this->locks.emplace_back(std::make_unique<std::shared_mutex>());
        this->active_states.emplace_back(std::make_unique<std::atomic_bool>(false));

        // 激活
        std::string local_index_prefix = this->index_prefix + "_" + std::to_string(i);
        if(file_exists(local_index_prefix + "_disk.index")){
            bool expected_active = false;
            this->active_states[i]->compare_exchange_strong(expected_active, true);
        }else{
            this->free_slots.push(i);
        }
    }
}

template<typename T, typename TagT>
int MultiPQFlashIndexProxy<T, TagT>::GetFreeIndexSlot(){
    std::unique_lock<std::shared_mutex>local_lock(this->free_slots_lock);
    while(this->free_slots.empty()){
      return -1;
    }
    int slot = this->free_slots.front();
    this->free_slots.pop();
    return slot;
}
template<typename T, typename TagT>
std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> MultiPQFlashIndexProxy<T, TagT>::GetNonFreeIndexList(){
    std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> res;
    for(size_t i = 0; i < this->indexes.size(); i++){
        if(this->active_states[i]->load() == false){
            continue;
        }
        res.emplace_back(this->indexes[i]);
    }
    return res;
}
template<typename T, typename TagT>
void MultiPQFlashIndexProxy<T, TagT>::KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, diskann::QueryStats * stats){

  size_t size = this->indexes.size();
  std::vector<std::vector<diskann::Neighbor_Tag<TagT>>> local_res;
  local_res.resize(size);
//   auto delete_lock = this->GetReadDeleteLock();
  std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
  std::vector<int> lock_idx_vec;
  lock_vec.reserve(size);
  for(size_t i = 0; i < size; i++){
      if(this->active_states[i]->load() == false){
          continue;
      }
      lock_vec.emplace_back(this->GetSubReadLock(i));
      lock_idx_vec.emplace_back(i);
  }
  
  options.search_L = 15;
  //check each disk index - if non empty - search and get top K active tags
  #pragma omp parallel for schedule(dynamic, 1) num_threads(size) 
  for(auto i : lock_idx_vec){
    this->indexes[i]->KNNQuery(query, local_res[i], options, stats);
  }

  // each component filter delete set younger than itself
  std::vector<diskann::Neighbor_Tag<TagT>> filter_res;
  for(auto i : lock_idx_vec){
    if(local_res[i].empty()){
        continue;
    }
    // first filter delete set
    this->indexes[i]->FilterDeletedTags(filter_res);
    // then add cur res
    filter_res.insert(filter_res.end(), local_res[i].begin(), local_res[i].end());
  }
  res.insert(res.end(), filter_res.begin(), filter_res.end());
}
template<typename T, typename TagT>
void MultiPQFlashIndexProxy<T, TagT>::ReloadIndex(const std::string &disk_index_prefix, size_t idx){
    if(idx >= this->indexes.size()){
        return;
    }
    if(this->active_states[idx]->load() == true){
        return;
    }
    this->GetSubWriteLock(idx);
    this->indexes[idx]->LoadIndex(disk_index_prefix);
    this->RefreshDeleteTagSet();
    bool expected_active = false;
    this->active_states[idx]->compare_exchange_strong(expected_active, true);
}

template<typename T, typename TagT>
void MultiPQFlashIndexProxy<T, TagT>::ClearSubIndex(size_t idx){
    // 进行clear
     if(idx >= this->indexes.size()){
        return;
    }
    this->GetSubWriteLock(idx);
    this->indexes[idx]->ClearIndex();
    this->AddFreeSlot(idx);

    bool expected_active = true;
    this->active_states[idx]->compare_exchange_strong(expected_active, false);
}
template<typename T, typename TagT>
void MultiPQFlashIndexProxy<T, TagT>::ClearIndex(){
    size_t size = this->indexes.size();
    for(size_t i = 0; i < size; i++){
      this->ClearSubIndex(i);
    }
    this->delete_tag_set.Clear();
}
template<typename T, typename TagT>
void MultiPQFlashIndexProxy<T, TagT>::BatchLazyDeleteIndex(std::vector<int>& remove_idx_vec){
  for(auto i : remove_idx_vec){
    this->active_states[i]->store(false);
  }
}
template<typename T, typename TagT>
void MultiPQFlashIndexProxy<T, TagT>::RefreshDeleteTagSet(){
    size_t size = this->indexes.size();
    std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
    std::vector<int> lock_idx_vec;
    lock_vec.reserve(size);
    for(size_t i = 0; i < size; i++){
        if(this->active_states[i]->load() == false){
            continue;
        }
        lock_vec.emplace_back(this->GetSubReadLock(i));
        lock_idx_vec.emplace_back(i);
    }
    tsl::robin_set<TagT> union_set;
    for(auto i : lock_idx_vec){
        this->indexes[i]->Union(union_set);
    }
    this->delete_tag_set.Swap(union_set);
}
template<typename T, typename TagT>
void MultiPQFlashIndexProxy<T, TagT>::AddFreeSlot(int idx){
    std::unique_lock<std::shared_mutex>local_lock(this->free_slots_lock);
    diskann::cout<< "Add free slots " << idx << " in multi disk"<< std::endl;
    this->free_slots.push(idx);
    return;
}
template<typename T, typename TagT>
void MultiPQFlashIndexProxy<T, TagT>::GetMedoids(std::vector<TagT>& medoid_vec){
    size_t size = this->indexes.size();
    for(size_t i = 0; i < size; i++){
        if(this->active_states[i]->load() == false){
            continue;
        }
        this->indexes[i]->GetMedoids(medoid_vec);
    }
}
template<typename T, typename TagT>
std::unique_lock<std::shared_mutex> MultiPQFlashIndexProxy<T, TagT>::GetSubWriteLock(size_t idx){
    if (idx >= this->indexes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return this->indexes[idx]->GetWriteLock();
}
template<typename T, typename TagT>
std::shared_lock<std::shared_mutex> MultiPQFlashIndexProxy<T, TagT>::GetSubReadLock(size_t idx){
    if (idx >= this->indexes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return this->indexes[idx]->GetReadLock();
}
template<typename T, typename TagT>
std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> MultiPQFlashIndexProxy<T, TagT>::GetIndexProxy(size_t idx){
    if (idx >= indexes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return this->indexes[idx];
}
template<typename T, typename TagT>
std::shared_ptr<diskann::PQFlashIndex<T, TagT>> MultiPQFlashIndexProxy<T, TagT>::GetIndex(size_t idx){
    if (idx >= indexes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return this->indexes[idx]->GetIndex();
}
template<typename T, typename TagT>
std::string MultiPQFlashIndexProxy<T, TagT>::ReportIndexInfo(){
    std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
    size_t size = this->indexes.size();
    std::vector<int> lock_idx_vec;
    lock_vec.reserve(size);
    for(size_t i = 0; i < size; i++){
        if(this->active_states[i]->load() == false){
            continue;
        }
        lock_vec.emplace_back(this->GetSubReadLock(i));
        lock_idx_vec.emplace_back(i);
    }
    std::ostringstream oss;
    oss << '[';
    for(auto i : lock_idx_vec){
        oss << "sub index " << i <<" :" << this->indexes[i]->ReportIndexInfo() << "; ";
    }
    oss << ']';
    return oss.str();
}
// template class instantiations
template class MultiPQFlashIndexProxy<float, uint32_t>;
template class MultiPQFlashIndexProxy<uint8_t, uint32_t>;
template class MultiPQFlashIndexProxy<int8_t, uint32_t>;
template class MultiPQFlashIndexProxy<float, int64_t>;
template class MultiPQFlashIndexProxy<uint8_t, int64_t>;
template class MultiPQFlashIndexProxy<int8_t, int64_t>;
template class MultiPQFlashIndexProxy<float, uint64_t>;
template class MultiPQFlashIndexProxy<uint8_t, uint64_t>;
template class MultiPQFlashIndexProxy<int8_t, uint64_t>;

} // namespace lsmidx