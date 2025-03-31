#include "lsm/tag_deleter.h"

namespace lsmidx{
template<typename TagT>
MultiTagDeleter<TagT>::MultiTagDeleter(int num):check_switch_delete(false){
    // 初始化tag_sets及对应的状态数组
    this->deletion_tag_sets.resize(num);
    this->active_states.reserve(num);
    for(int i = 0; i < num; i++){
        this->active_states.emplace_back(std::make_unique<std::atomic_bool>(false));
    }
    // 将第一个tag_set激活
    bool expected_active = false;
    this->active_states[0]->compare_exchange_strong(expected_active, true);
    this->current = 0;
}
template<typename TagT>
bool MultiTagDeleter<TagT>::Insert(TagT tag){
  if(this->active_states[this->current]->load() == false){
    diskann::cout << "Active deletion set indicated as deletion_tag_set"<<this->current<<" but it cannot accept deletions" << std::endl;
    return false;
  }
  tsl::robin_set<TagT>& deletion_set = this->deletion_tag_sets[this->current];
  deletion_set.insert(tag);
  return true;
}

template<typename TagT>
bool MultiTagDeleter<TagT>::IsDelete(TagT tag){
  for(size_t i = 0;i< this->deletion_tag_sets.size();i++){
    tsl::robin_set<TagT>& deletion_set = this->deletion_tag_sets[i];
    if(deletion_set.find(tag) != deletion_set.end()){
      return true;
    }
  }
  return false;
}
template<typename TagT>
int MultiTagDeleter<TagT>::GetNextSwitchIdx(){
  return (this->current + 1) % this->deletion_tag_sets.size();
}

template<typename TagT>
std::vector<TagT>* MultiTagDeleter<TagT>::Switch(){

  // 使用原子操作检查并切换删除集合的活动状态标志 check_switch_delete，避免并发修改
  bool expected_value = false;
  this->check_switch_delete.compare_exchange_strong(expected_value, true);

  // 获取下一个活动set
  int next_idx = this->GetNextSwitchIdx();

  /**
   * 将下一个活动set从非激活态设置为激活态，并将其原本内容清空
  */
  bool expected_active = false;
  if(this->active_states[next_idx]->compare_exchange_strong(expected_active, true)) {
    this->deletion_tag_sets[next_idx].clear();
    diskann::cout << "Cleared deletion_tag_set_"<<next_idx<<" - ready to accept new points" << std::endl;
  } else{
    diskann::cout << "Failed to clear _deletion_set_1" << std::endl;
  }

  // 将原本激活的set设置为非激活
  expected_active  = true;
  this->active_states[this->current]->compare_exchange_strong(expected_active, false);
  
  // 将原本的set中的tag返回
  tsl::robin_set<TagT>& cur_set = this->deletion_tag_sets[this->current];
  std::vector<TagT> * del_vec = new std::vector<TagT>(cur_set.size());
  size_t i = 0;
  for(auto iter : cur_set){
    (*del_vec)[i] = iter;
    i++;
  }

  // 切换当前指向的删除集合
  this->current = next_idx;
  
  // 恢复 check_switch_delete 为 false，允许下次切换
  expected_value = true;
  check_switch_delete.compare_exchange_strong(expected_value, false);

  return del_vec;
}
template<typename TagT>
void MultiTagDeleter<TagT>::FilterDeletedTags(tsl::robin_set<TagT>& active_tags){
  for(auto& delete_set: this->deletion_tag_sets){
      for(TagT tag: delete_set){
          active_tags.erase(tag);
      }
  }
  return;
}
template class MultiTagDeleter<uint32_t>;
template class MultiTagDeleter<int64_t>;
template class MultiTagDeleter<uint64_t>;

template<typename TagT>
bool TagDeleter<TagT>::Insert(TagT tag){
  auto write_lock = this->GetWriteLock();
  if(this->delete_set.count(tag)){
    return false;
  }
  this->delete_set.insert(tag);
  return true;
}
template<typename TagT>
void TagDeleter<TagT>::FilterDeletedTags(tsl::robin_set<TagT>& active_tags){
  auto read_lock = this->GetReadLock();
  for(TagT tag: this->delete_set){
    active_tags.erase(tag);
  }
  return;
}
template<typename TagT>
bool TagDeleter<TagT>::IsDelete(TagT tag){
  auto read_lock = this->GetReadLock();
  if(this->delete_set.find(tag) != this->delete_set.end()){
    return true;
  }
  return false;
}
template<typename TagT>
_u64 TagDeleter<TagT>::Save(std::string index_prefix){
  auto read_lock = this->GetReadLock();
  if (this->delete_set.size() == 0) {
    return 0;
  }
  std::string out_disk_index_path = index_prefix + ".del";
  std::unique_ptr<TagT[]> delete_list =
      std::make_unique<TagT[]>(this->delete_set.size());
  _u32 i = 0;
  for (auto &del : delete_set) {
    delete_list[i++] = del;
  }
  read_lock.unlock();
  return diskann::save_bin<TagT>(out_disk_index_path, delete_list.get(), this->delete_set.size(), 1);
}
template<typename TagT>
_u64 TagDeleter<TagT>::Load(std::string index_prefix){
  auto write_lock = this->GetWriteLock();
  if(!this->delete_set.empty()){
    this->delete_set.clear();
  }
  
  std::string in_disk_index_path = index_prefix + ".del";
  if(!file_exists(in_disk_index_path)){
    return 0;
  }
  std::unique_ptr<TagT[]> delete_list;
  _u64                    npts, ndim;
  diskann::load_bin<TagT>(in_disk_index_path, delete_list, npts, ndim);

  for (size_t i = 0; i < npts; i++) {
    this->delete_set.insert(delete_list[i]);
  }
  return npts;
}
template<typename TagT>
std::unique_lock<std::shared_mutex> TagDeleter<TagT>::GetWriteLock(){
  return std::unique_lock<std::shared_mutex>(lock);
}
template<typename TagT>
std::shared_lock<std::shared_mutex> TagDeleter<TagT>::GetReadLock(){
  return std::shared_lock<std::shared_mutex>(lock);
}
template<typename TagT>
void TagDeleter<TagT>::Swap(tsl::robin_set<TagT>& delete_set){
  auto write_lock = this->GetWriteLock();
  this->delete_set = std::move(delete_set);
}
template<typename TagT>
bool TagDeleter<TagT>::IsEmpty(){
  auto read_lock = this->GetReadLock();
  return this->delete_set.empty();
}
template<typename TagT>
void TagDeleter<TagT>::Clear(){
  auto write_lock = this->GetWriteLock();
  this->delete_set.clear();
}
template<typename TagT>
void TagDeleter<TagT>::Union(tsl::robin_set<TagT>& union_delete_set){
  auto read_lock = this->GetReadLock();
  union_delete_set.insert(this->delete_set.begin(), this->delete_set.end());
}
template<typename TagT>
void TagDeleter<TagT>::Union(TagDeleter<TagT>& union_delete_set){
  auto read_lock = this->GetReadLock();
  for(auto tag : this->delete_set){
    union_delete_set.Insert(tag);
  }
}
template class TagDeleter<uint32_t>;
template class TagDeleter<int64_t>;
template class TagDeleter<uint64_t>;

} // namesp lsmidx