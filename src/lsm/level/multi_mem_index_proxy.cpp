#include "lsm/level/level_index.h"

namespace lsmidx{
  template<typename T, typename TagT>
void MultiInMemIndexProxy<T, TagT>::GetActiveTags(tsl::robin_set<TagT>& active_tags){
    size_t size = this->indexes.size();
    // acquire locks
    auto cur_lock = GetCurrentMutexReadLock();
    std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
    for(size_t i = 0; i < size; i++){
        lock_vec.emplace_back(this->GetSubReadLock(i));
    }
    std::vector<tsl::robin_set<TagT>> local_res;
    local_res.resize(size);
    for(size_t i = 0; i < size; i++){
        this->indexes[i]->GetActiveTags(local_res[i]);
    }
    tsl::robin_set<TagT> filter_res;
    
    for(int i = GetNextSwitchIdx(); i != this->current; i = (i+1) % size){
        // first filter delete set
        this->indexes[i]->FilterDeletedTags(filter_res);
        // then add cur res
        filter_res.insert(local_res[i].begin(), local_res[i].end());
    }
    this->indexes[this->current]->FilterDeletedTags(filter_res);
    filter_res.insert(local_res[this->current].begin(), local_res[this->current].end());
    active_tags.insert(filter_res.begin(), filter_res.end());
    return;
}
template<typename T, typename TagT>
MultiInMemIndexProxy<T, TagT>::MultiInMemIndexProxy(diskann::Metric dist_metric, std::string working_dir, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_mem, bool is_single_file_index):MultiIndex<T, TagT>(IndexType::IN_MEM_DISKANN, paras_mem, merge_thresh, 0, dims, is_single_file_index, dist_metric){
    // 初始化基本参数
    this->index_prefix = working_dir + '/' + lsmidx::config::mem_index_name;
    int num = lsmidx::config::mem_indexes_num;
    // 初始化index以及对应的状态数组
    for(int i = 0; i<num ; i++){
        this->indexes.emplace_back(std::make_shared<lsmidx::InMemIndexProxy<T, TagT>>(
            dist_metric, working_dir, i, dims, merge_thresh, paras_mem, is_single_file_index
        ));
    }
    this->active_states.reserve(num);
    for(int i = 0; i < num; i++){
        this->active_states.emplace_back(std::make_unique<std::atomic_bool>(false));
        // this->index_clearing_states.emplace_back(std::make_unique<std::atomic_bool>(false));
        // this->clear_locks.emplace_back(std::make_unique<std::shared_mutex>());
    }
    
    // 将第一个mem_index激活
    bool expected_active = false;
    this->active_states[0]->compare_exchange_strong(expected_active, true);
    this->current = 0;
}
template<typename T, typename TagT>
void MultiInMemIndexProxy<T, TagT>::KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options,[[maybe_unused]] std::vector<lsmidx::TagDeleter<TagT>*> exclude_set_list, [[maybe_unused]] diskann::QueryStats * stats){
  size_t size = this->indexes.size();
  std::vector<std::vector<diskann::Neighbor_Tag<TagT>>> local_res;
  local_res.resize(size);
  auto cur_lock = GetCurrentMutexReadLock();
  std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
  for(size_t i = 0; i < size; i++){
    lock_vec.emplace_back(this->GetSubReadLock(i));
  }
  std::vector<lsmidx::TagDeleter<TagT>*> empty_exclude_set_list;
  //check each memory index - if non empty and not being currently cleared - search and get top K active tags 
  for(size_t i = 0;i<size;i++){
    std::shared_ptr<lsmidx::InMemIndexProxy<T, TagT>>& index = this->indexes[i];
    index->KNNQuery(query, local_res[i], options, empty_exclude_set_list);
  }
  // each component filter delete set younger than itself
  
  std::vector<diskann::Neighbor_Tag<TagT>> filter_res;
  for(int i = GetNextSwitchIdx(); i != this->current; i = (i+1) % size){
    // first filter delete set
    this->indexes[i]->FilterDeletedTags(filter_res);
    // then add cur res
    filter_res.insert(filter_res.end(), local_res[i].begin(), local_res[i].end());
  }
  this->indexes[this->current]->FilterDeletedTags(filter_res);
  filter_res.insert(filter_res.end(), local_res[this->current].begin(), local_res[this->current].end());
   
  res.insert(res.end(), filter_res.begin(), filter_res.end());
}

template<typename T, typename TagT>
int MultiInMemIndexProxy<T, TagT>::LazyDelete(TagT tag){
    auto cur_lock = this->GetCurrentMutexReadLock();
    // 确认目前所指向的index是否在工作
    if(this->active_states[this->current]->load() == false){
        diskann::cout << "Active index indicated as mem_index_"<< this->current << "but it cannot accept insertions" << std::endl;
        return -1;
    }
    auto read_lock = this->GetSubReadLock(this->current);
    this->delete_tag_set.Insert(tag);
    this->indexes[this->current]->LazyDelete(tag);
    return 0;
}

template<typename T, typename TagT>
int MultiInMemIndexProxy<T, TagT>::Put([[maybe_unused]] const WriteOptions& options, const VecSlice<T>& key, const TagT& tag, diskann::InsertStats * stats){
    auto s = std::chrono::high_resolution_clock::now();
    auto cur_lock = this->GetCurrentMutexReadLock();
    auto e = std::chrono::high_resolution_clock::now();
    
    // 确认目前所指向的index是否在工作
    if(this->active_states[this->current]->load() == false){
        diskann::cout << "Active index indicated as mem_index_"<< this->current << "but it cannot accept insertions" << std::endl;
        return -1;
    }
    auto s2 = std::chrono::high_resolution_clock::now();
    auto read_lock = this->GetSubReadLock(this->current);
    auto e2 = std::chrono::high_resolution_clock::now();
    // 拿到目前指向的index并进行插入
    auto res = this->indexes[this->current]->Put(options, key, tag, stats);
    auto e3 = std::chrono::high_resolution_clock::now();
    if(stats){
        std::chrono::duration<double> diff = e - s;
        std::chrono::duration<double> diff2 = e2 - s2;
        std::chrono::duration<double> diff3 = e3 - e2;
        stats->get_cur_mutex_read_lock_time += diff.count();
        stats->get_cur_index_read_lock_time += diff2.count();
        stats->insert_time += diff3.count();
    }
    return res;
}
template<typename T, typename TagT>
int MultiInMemIndexProxy<T, TagT>::Switch(){
    auto cur_lock = this->GetCurrentMutexWriteLock();
    // 获取下一个活动index
    int next_idx = this->GetNextSwitchIdx();

    // 如果说下一个index不为空，则进行clear
    if(this->active_states[next_idx]->load() == false){
        this->ClearSubIndex(next_idx);
    }
    // 拿到当前和下一个将切换的index的写锁
    // in case of dead lock
    int left = this->current<next_idx?this->current:next_idx;
    int right = this->current<next_idx?next_idx:this->current;
    auto lock1 = this->GetSubReadLock(left);
    auto lock2 = this->GetSubReadLock(right);
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
std::string MultiInMemIndexProxy<T, TagT>::SaveIndex(size_t idx){
    std::string save_path;
    if(idx>=this->indexes.size()){
        return save_path;
    }
    if(this->active_states[idx]->load() == true){
        return save_path;
    }
    auto read_lock = this->GetSubReadLock(idx);
    save_path = this->indexes[idx]->SaveIndex();

    return save_path;
}
template<typename T, typename TagT>
void MultiInMemIndexProxy<T, TagT>::ClearSubIndex(size_t idx){
    if(idx>=this->indexes.size()){
        return;
    }
    if(this->active_states[idx]->load() == true){
        return;
    }
    {
        auto write_lock = this->GetSubWriteLock(idx);
        // 进行clear
        diskann::cout<<"clear sub index "<< idx << "in multi mem"<<std::endl;
        this->indexes[idx]->ClearIndex();
    }
    this->RefreshDeleteTagSet();

}
template<typename T, typename TagT>
void MultiInMemIndexProxy<T, TagT>::ClearIndex(){
    size_t size = this->indexes.size();
    for(size_t i = 0; i < size; i++){
      this->ClearSubIndex(i);
    }
    this->delete_tag_set.Clear();
}

template<typename T, typename TagT>
int MultiInMemIndexProxy<T, TagT>::GetCurrentNumPoints(){
  int sum = 0;
  // acquire locks
  size_t size = this->indexes.size();
  std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
  for(size_t i = 0; i < size; i++){
      lock_vec.emplace_back(this->GetSubReadLock(i));
  }
  for(size_t i = 0;i<this->indexes.size();i++){
    sum += this->indexes[i]->GetCurrentNumPoints();
  }
  return sum;
}
template<typename T, typename TagT>
int MultiInMemIndexProxy<T, TagT>::GetNumPointsOfCur(){
  // acquire locks
  auto cur_lock = this->GetCurrentMutexReadLock();
  auto lock = this->GetSubReadLock(this->current);
  return this->indexes[this->current]->GetCurrentNumPoints();
}
template<typename T, typename TagT>
void MultiInMemIndexProxy<T, TagT>::RefreshDeleteTagSet(){
    std::vector<std::vector<diskann::Neighbor_Tag<TagT>>> local_res;
    size_t size = this->indexes.size();
    local_res.reserve(size);
    std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
    for(size_t i = 0; i < size; i++){
        lock_vec.emplace_back(this->GetSubReadLock(i));
    }
    tsl::robin_set<TagT> union_set;
    for(size_t i = 0; i < size; i++){
        this->indexes[i]->Union(union_set);
    }
    this->delete_tag_set.Swap(union_set);
}
template<typename T, typename TagT>
void MultiInMemIndexProxy<T, TagT>::RecalculateInsertMemIndexEntryPoint(){
    auto cur_lock = this->GetCurrentMutexReadLock();
    auto lock = this->GetSubReadLock(this->current);
    this->indexes[this->current]->GetIndex()->recalculate_entry_point();
}
template<typename T, typename TagT>
int MultiInMemIndexProxy<T, TagT>::GetNextSwitchIdx(){
  return (this->current + 1) % this->indexes.size();
}
template<typename T, typename TagT>
std::unique_lock<std::shared_mutex> MultiInMemIndexProxy<T, TagT>::GetSubWriteLock(size_t idx){
    if (idx >= this->indexes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return this->indexes[idx]->GetWriteLock();
}
template<typename T, typename TagT>
std::shared_lock<std::shared_mutex> MultiInMemIndexProxy<T, TagT>::GetSubReadLock(size_t idx){
    if (idx >= this->indexes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return this->indexes[idx]->GetReadLock();
}
template<typename T, typename TagT>
std::shared_lock<std::shared_mutex> MultiInMemIndexProxy<T, TagT>::GetCurrentMutexReadLock(){
    
    return std::shared_lock<std::shared_mutex>(this->current_mutex);
}
template<typename T, typename TagT>
std::unique_lock<std::shared_mutex> MultiInMemIndexProxy<T, TagT>::GetCurrentMutexWriteLock(){
    return std::unique_lock<std::shared_mutex>(this->current_mutex);
}
template<typename T, typename TagT>
std::shared_ptr<InMemIndexProxy<T, TagT>> MultiInMemIndexProxy<T, TagT>::GetIndexProxy(size_t idx){
    if (idx >= indexes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return this->indexes[idx];
}
template<typename T, typename TagT>
std::shared_ptr<diskann::Index<T, TagT>> MultiInMemIndexProxy<T, TagT>::GetIndex(size_t idx){
    if (idx >= indexes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return this->indexes[idx]->GetIndex();
}
template<typename T, typename TagT>
std::string MultiInMemIndexProxy<T, TagT>::ReportIndexInfo(){
    std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
    size_t size = this->indexes.size();
    for(size_t i = 0; i < size; i++){
        lock_vec.emplace_back(this->GetSubReadLock(i));
    }
    std::ostringstream oss;
    oss << '[';
    for(size_t i = 0; i < size; i++){
        oss << "sub index " << i <<" :" << this->indexes[i]->ReportIndexInfo() << "; ";
    }
    oss << ']';
    return oss.str();
}
// template class instantiations
template class MultiInMemIndexProxy<float, uint32_t>;
template class MultiInMemIndexProxy<uint8_t, uint32_t>;
template class MultiInMemIndexProxy<int8_t, uint32_t>;
template class MultiInMemIndexProxy<float, int64_t>;
template class MultiInMemIndexProxy<uint8_t, int64_t>;
template class MultiInMemIndexProxy<int8_t, int64_t>;
template class MultiInMemIndexProxy<float, uint64_t>;
template class MultiInMemIndexProxy<uint8_t, uint64_t>;
template class MultiInMemIndexProxy<int8_t, uint64_t>;
} // namespace lsmidx