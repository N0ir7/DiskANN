#include "lsm/merger/util/sector_buffer_pool.h"

namespace lsmidx{
  ReadOnlySectorBufferPool::ReadOnlySectorBufferPool() {
      diskann::alloc_aligned((void**)&buffer_, BUFFER_POOL_BLOCK_NUMS * SECTOR_LEN, SECTOR_LEN);
      if (!buffer_) {
          throw std::runtime_error("Failed to allocate aligned buffer");
      }
      pin_counts_.resize(BUFFER_POOL_BLOCK_NUMS, 0);
      slot_sector_ids_.resize(BUFFER_POOL_BLOCK_NUMS, -1);

      // 初始化 LRU：所有 slot 视为“未使用”，加入链尾
      for (int i = 0; i < BUFFER_POOL_BLOCK_NUMS; ++i) {
          lru_list_.push_back(i);
          slot_lru_iter_[i] = std::prev(lru_list_.end());
      }
  }
  void ReadOnlySectorBufferPool::InitIndexReader(std::string index_name){
    if(this->reader){
        this->reader.reset();
    }
    this->reader = std::make_shared<LinuxAlignedFileReader>();
    // this->reader->register_thread();
    this->reader->open(index_name, false, false);
    auto metas = get_disk_index_meta(index_name);
    npts = metas[0];
    data_dim = metas[1];
    max_node_len = metas[3];
    nnodes_per_sector = metas[4];
  }
  void ReadOnlySectorBufferPool::RegisterReaderThread(){
    if(this->reader){
        this->reader->register_thread();
    }
  }
  ReadOnlySectorBufferPool::~ReadOnlySectorBufferPool() {
      if (buffer_) {
          diskann::aligned_free(buffer_);
          buffer_ = nullptr;
      }
      if(reader){
        reader->close();
        reader.reset();
      }
  }

  bool ReadOnlySectorBufferPool::get_sector(int sector_id, char*& out_ptr) {
      std::lock_guard<std::mutex> lock(mu_);

      // 命中：直接返回
      auto it = sector_id_to_slot_.find(sector_id);
      if (it != sector_id_to_slot_.end()) {
          int slot = it->second;
          pin_counts_[slot]++;
          move_to_front(slot);  // 更新 LRU 顺序
          out_ptr = buffer_ + slot * SECTOR_LEN;
          return true;
      }
      cache_miss++;
      // 未命中：找可用 slot（优先空闲的），否则使用 LRU 尾部
      int slot = find_free_or_evict_lru_slot();

      // 如果该 slot 原来映射某个 sector_id，先移除
      int old_sector_id = slot_sector_ids_[slot];
      if (old_sector_id != -1) {
          sector_id_to_slot_.erase(old_sector_id);
      }

      // 分配新的
      sector_id_to_slot_[sector_id] = slot;
      slot_sector_ids_[slot] = sector_id;
      pin_counts_[slot] = 1;
      move_to_front(slot);  // 新使用，放在 LRU 前部

      out_ptr = buffer_ + slot * SECTOR_LEN;
      return false;
  }
  template<typename T>
  diskann::DiskNode<T> ReadOnlySectorBufferPool::get_node(int node_id) {
      _u64 start_sector = NODE_SECTOR_NO(((size_t) node_id));
      _u64 start_off = start_sector * SECTOR_LEN;
      char * sector_scratch = nullptr;
      bool cached = get_sector(start_sector, sector_scratch);
      if(!cached){
        IOContext &ctx = this->reader->get_ctx();
        std::vector<AlignedRead> read_reqs;
        read_reqs.emplace_back(
            start_off, SECTOR_LEN,
            sector_scratch);
        this->reader->read(read_reqs, ctx, false);
      }
      assert(sector_scratch != nullptr);
      char *node_buf =
            OFFSET_TO_NODE(sector_scratch, node_id);
      return diskann::DiskNode<T>(node_id, OFFSET_TO_NODE_COORDS(node_buf), OFFSET_TO_NODE_NHOOD(node_buf));
  }
  void ReadOnlySectorBufferPool::unpin_internal(int slot) {
      if (pin_counts_[slot] == 0) {
          throw std::runtime_error("Unpin called on unpinned slot");
      }
      pin_counts_[slot]--;
      // 不调整 LRU 顺序
  }
  void ReadOnlySectorBufferPool::unpin(char* sector_ptr) {
      int offset = sector_ptr - buffer_;
      if (offset < 0 || offset % SECTOR_LEN != 0 || offset >= BUFFER_POOL_BLOCK_NUMS * SECTOR_LEN) {
          throw std::runtime_error("Invalid sector pointer in unpin");
      }
      int slot = offset / SECTOR_LEN;
      std::lock_guard<std::mutex> lock(mu_);
      unpin_internal(slot);
  }
  void ReadOnlySectorBufferPool::unpin(int node_id){
    _u64 start_sector = NODE_SECTOR_NO(((size_t) node_id));
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = sector_id_to_slot_.find(start_sector);
    if(iter == sector_id_to_slot_.end()){
        throw std::runtime_error("Invalid node id in unpin");
    }
    auto slot = iter->second;
    unpin_internal(slot);
  }
template diskann::DiskNode<float> ReadOnlySectorBufferPool::get_node<float>(int);
template diskann::DiskNode<uint8_t> ReadOnlySectorBufferPool::get_node<uint8_t>(int);
template diskann::DiskNode<int8_t> ReadOnlySectorBufferPool::get_node<int8_t>(int);
};