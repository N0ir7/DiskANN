#pragma once
#include <vector>
#include <mutex>
#include <list>
#include "utils.h"
#include "concurrent_queue.h"
#include "lsm/options.h"
#include "aligned_file_reader.h"
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "gp/graph_manager.h"
#define BUFFER_POOL_BLOCK_NUMS 65536
// sector # on disk where node_id is present
#define NODE_SECTOR_NO(node_id) (((_u64) (node_id)) / nnodes_per_sector + 1)
// obtains region of sector containing node
#define OFFSET_TO_NODE(sector_buf, node_id) \
  ((char *) sector_buf + (((_u64) node_id) % nnodes_per_sector) * max_node_len)
// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + data_dim * sizeof(T))

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *) (node_buf)

namespace lsmidx{
class ReadOnlySectorBufferPool {
public:
  ReadOnlySectorBufferPool();
  void InitIndexReader(std::string index_name);
  ~ReadOnlySectorBufferPool();

  bool get_sector(int sector_id, char*& out_ptr);
  template<typename T>
  diskann::DiskNode<T> get_node(int node_id);
  void unpin_internal(int slot);
  void unpin(char* sector_ptr);
  void unpin(int node_id);
  void RegisterReaderThread();
  /**
   * 统计信息
  */
  int cache_miss = 0;
  std::shared_ptr<AlignedFileReader> reader;
private:
  char* buffer_;
  std::mutex mu_;

  std::unordered_map<int, int> sector_id_to_slot_; // sector_id -> slot
  std::vector<int> pin_counts_;                    // slot -> pin count
  std::vector<int> slot_sector_ids_;               // slot -> sector_id

  std::list<int> lru_list_;                        // LRU 顺序（前：最新，后：最老）
  std::unordered_map<int, std::list<int>::iterator> slot_lru_iter_; // slot -> lru iterator
  
  _u64 npts;
  _u64 max_node_len;
  _u64 nnodes_per_sector;
  size_t data_dim;
  void move_to_front(int slot) {
      auto it = slot_lru_iter_.find(slot);
      if (it != slot_lru_iter_.end()) {
          lru_list_.erase(it->second);
      }
      lru_list_.push_front(slot);
      slot_lru_iter_[slot] = lru_list_.begin();
  }

  int find_free_or_evict_lru_slot() {
      for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
          int slot = *it;
          if (pin_counts_[slot] == 0) {
              lru_list_.erase(std::next(it).base());
              slot_lru_iter_.erase(slot);
              return slot;
          }
      }

      throw std::runtime_error("No free or evictable sector slot available");
  }
};
}