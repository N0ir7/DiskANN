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
class TagDeleter{
public:
  bool Insert(TagT tag);
  bool IsDelete(TagT tag);
  std::vector<TagT>* Switch();
private:
  int GetNextSwitchIdx();
  int current = 0;
  std::vector<tsl::robin_set<TagT>> deletion_tag_sets;
  std::vector<std::atomic_bool> active_states;
  std::atomic_bool check_switch_delete = false; 
};

} // namesp lsmidx