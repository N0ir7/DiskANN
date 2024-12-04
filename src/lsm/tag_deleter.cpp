#include "lsm/tag_deleter.h"

namespace lsmidx{

template<typename TagT>
bool TagDeleter<TagT>::Insert(TagT tag){
  if(this->active_states[this->current].load() == false){
    diskann::cout << "Active deletion set indicated as deletion_tag_set"<<this->current<<" but it cannot accept deletions" << std::endl;
    return false;
  }
  tsl::robin_set<TagT>& deletion_set = this->deletion_tag_sets[this->current];
  deletion_set.insert(tag);
  return true;
}

template<typename TagT>
bool TagDeleter<TagT>::IsDelete(TagT tag){
  for(int i = 0;i< this->deletion_tag_sets.size();i++){
    tsl::robin_set<TagT>& deletion_set = this->deletion_tag_sets[i];
    if(deletion_set.find(tag) != deletion_set.end()){
      return true;
    }
  }
  return false;
}
template<typename TagT>
int TagDeleter<TagT>::GetNextSwitchIdx(){
  return (this->current + 1) % this->deletion_tag_sets.size();
}

template<typename TagT>
std::vector<TagT>* TagDeleter<TagT>::Switch(){

  // 使用原子操作检查并切换删除集合的活动状态标志 check_switch_delete，避免并发修改
  bool expected_value = false;
  this->check_switch_delete.compare_exchange_strong(expected_value, true);

  // 获取下一个活动set
  int next_idx = this->GetNextSwitchIdx();

  /**
   * 将下一个活动set从非激活态设置为激活态，并将其原本内容清空
  */
  bool expected_active = false;
  if(this->active_states[next_idx].compare_exchange_strong(expected_active, true)) {
    this->deletion_tag_sets[next_idx].clear();
    diskann::cout << "Cleared deletion_tag_set_"<<next_idx<<" - ready to accept new points" << std::endl;
  } else{
    diskann::cout << "Failed to clear _deletion_set_1" << std::endl;
  }

  // 将原本激活的set设置为非激活
  expected_active  = true;
  this->active_states[this->current].compare_exchange_strong(expected_active, false);
  
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

} // namesp lsmidx