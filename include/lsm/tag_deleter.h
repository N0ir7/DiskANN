#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include "lsm/options.h"
#include "lsm/slice.h"
#include "Neighbor_Tag.h"
#include "index.h"
#include "tsl/robin_set.h"

namespace lsmidx{
template<typename TagT = uint32_t>
class MultiTagDeleter{
public:
  MultiTagDeleter(int num);
  bool Insert(TagT tag);
  bool IsDelete(TagT tag);
  std::vector<TagT>* Switch();
  void FilterDeletedTags(tsl::robin_set<TagT>& active_tags);
private:
  int GetNextSwitchIdx();
  int current = 0;
  std::vector<tsl::robin_set<TagT>> deletion_tag_sets;
  std::vector<std::unique_ptr<std::atomic_bool>> active_states;
  std::atomic_bool check_switch_delete; 
};

template<typename TagT = uint32_t>
class TagDeleter{
public:
  bool Insert(TagT tag);
  bool IsDelete(TagT tag);
  void FilterDeletedTags(tsl::robin_set<TagT>& active_tags);
  _u64 Save(std::string index_prefix);
  _u64 Load(std::string index_prefix);
  std::unique_lock<std::shared_mutex> GetWriteLock();
  std::shared_lock<std::shared_mutex> GetReadLock();
  bool IsEmpty();
  void Union(tsl::robin_set<TagT>& union_delete_set);
  void Union(TagDeleter<TagT>& union_delete_set);
  void Clear();
  void Swap(tsl::robin_set<TagT>& delete_set);
private:
  tsl::robin_set<TagT> delete_set;
  std::shared_mutex lock;
};
} // namesp lsmidx