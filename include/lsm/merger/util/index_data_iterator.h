#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include "pq_flash_index.h"
#include "concurrent_queue.h"
#include "lsm/level/level_index.h"

namespace lsmidx
{
struct DiskIndexFileMeta{
  std::string vector_path;
  std::string data_path;
  std::string tag_path;
  std::string pq_coords_path;
  std::string pq_table_path;
  std::string medoids_file_path;
  std::string centroids_file_path;
  std::string delete_list_path;
  std::string index_prefix_path;
  bool is_single_file;
  DiskIndexFileMeta(){};
  DiskIndexFileMeta(std::string data_path,std::string tag_path,std::string pq_coords_path,std::string pq_table_path)
  :data_path(data_path),tag_path(tag_path),pq_coords_path(pq_coords_path),pq_table_path(pq_table_path){};

  DiskIndexFileMeta(std::string iprefix,bool is_single_file):is_single_file(is_single_file){
    index_prefix_path = iprefix;
    if(is_single_file){
      pq_table_path = iprefix;
      pq_coords_path = iprefix;
      data_path = iprefix;
    }else{
      pq_table_path = iprefix + "_pq_pivots.bin";
      pq_coords_path = iprefix + "_pq_compressed.bin";
      data_path = iprefix + "_disk.index";
      tag_path = iprefix + "_disk.index.tags";
      medoids_file_path = data_path + "_medoids.bin";
      centroids_file_path = data_path + "_centroids.bin";
      delete_list_path = iprefix + ".del";
    }
  }
};

template<typename T,typename TagT>
class DiskIndexDataIterator{
  public:
    ~DiskIndexDataIterator();
    DiskIndexDataIterator(DiskIndexFileMeta index_file_meta,
                          std::shared_ptr<diskann::PQFlashIndex<T, TagT>> index)
                          :index_file_meta_(index_file_meta)
                          ,index_(index){};
    // 移动构造函数
    DiskIndexDataIterator(DiskIndexDataIterator&& other) noexcept
        : index_file_meta_(std::move(other.index_file_meta_)),
          output_index_file_meta_(std::move(other.output_index_file_meta_)),
          index_(std::move(other.index_)),
          disk_nodes_(std::move(other.disk_nodes_)),
          local_offset_(other.local_offset_),
          cur_start_id_(other.cur_start_id_),
          next_start_id_(other.next_start_id_),
          node_need_flush_back_(other.node_need_flush_back_),
          pq_need_flush_back_(other.pq_need_flush_back_),
          tag_need_flush_back_(other.tag_need_flush_back_),
          read_only_(other.read_only_),
          read_write_same_file_(other.read_write_same_file_),
          buf_(other.buf_) {
        other.buf_ = nullptr; // 防止悬空指针
    }

    // 禁用复制构造函数
    DiskIndexDataIterator(const DiskIndexDataIterator&) = delete;

    // 移动赋值运算符（如果需要）
    DiskIndexDataIterator& operator=(DiskIndexDataIterator&& other) noexcept {
        if (this != &other) {
            index_file_meta_ = std::move(other.index_file_meta_);
            output_index_file_meta_ = std::move(other.output_index_file_meta_);
            index_ = std::move(other.index_);
            disk_nodes_ = std::move(other.disk_nodes_);
            local_offset_ = other.local_offset_;
            cur_start_id_ = other.cur_start_id_;
            next_start_id_ = other.next_start_id_;
            node_need_flush_back_ = other.node_need_flush_back_;
            pq_need_flush_back_ = other.pq_need_flush_back_;
            tag_need_flush_back_ = other.tag_need_flush_back_;
            read_only_ = other.read_only_;
            read_write_same_file_ = other.read_write_same_file_;
            buf_ = other.buf_;
            other.buf_ = nullptr;
        }
        return *this;
    }
    bool GetNode(diskann::DiskNode<T>& node, unsigned node_id);
    void Init(bool read_only, int sectors_per_batch, DiskIndexFileMeta* output_index_file_meta = nullptr);
    std::tuple<diskann::DiskNode<T> *, uint8_t *, TagT*> Next();
    std::tuple<diskann::DiskNode<T>*, uint8_t *, TagT*> SeekNode(unsigned id);
    void SeekBatch(unsigned node_id);
    std::tuple<std::vector<diskann::DiskNode<T>> *,uint8_t *, TagT*> NextBatch();
    bool HasNext();
    bool HasNextBatch();
    void NotifyFlushBack();
    void TryFlushBack();
    void NotifyNodeFlushBack();
    void NotifyPQCoordFlushBack();
    void NotifyTagFlushBack();
    double GetIOTime(){
      return io_time;
    }
    int GetRandomRead(){
      return random_read_4k;
    }
    int GetRandomWrite(){
      return random_write_4k;
    }
    int GetSeqRead(){
      return seq_read_4k;
    }
    int GetSeqWrite(){
      return seq_write_4k;
    }
  private:
    DiskIndexFileMeta index_file_meta_;
    DiskIndexFileMeta output_index_file_meta_;
    std::shared_ptr<diskann::PQFlashIndex<T, TagT>> index_;
    std::vector<diskann::DiskNode<T>> disk_nodes_;
    uint32_t local_offset_ = 0;
    uint32_t cur_start_id_ = 0;
    uint32_t next_start_id_ = 0;
    bool node_need_flush_back_ = false;
    bool pq_need_flush_back_ = false;
    bool tag_need_flush_back_ = false;
    bool read_only_ = true; 
    bool read_write_same_file_ = true; 
    char * buf_ = nullptr;
    int sectors_per_batch;
    void NodeFlushBack();
    void PQCoordFlushBack();
    void TagFlushBack();
    void DumpToDisk(const uint32_t start_id,
                    const char *   buf,
                    const uint32_t n_sectors,
                    std::string data_path);
    // int sum = 0;
    double io_time = 0;
    int random_read_4k = 0;
    int seq_read_4k = 0;
    int random_write_4k = 0;
    int seq_write_4k = 0;
};
template<typename T,typename TagT>
class MultiDiskIndexDataIterator{
  public:
    MultiDiskIndexDataIterator(std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> indexes, std::vector<tsl::robin_set<TagT>>* ptr):indexes(std::move(indexes)), deleted_tags_vec(ptr){};

    std::tuple<diskann::DiskNode<T> *, uint8_t *, TagT*> Next();
    bool HasNext();
    bool HasNextBatch();
    std::tuple<std::vector<diskann::DiskNode<T>> *,uint8_t *, TagT*> NextBatch();
    tsl::robin_set<TagT>* GetCurDeleteTagSet();
    int GetCurIndexFrozenPoint();
    void Init();
    double GetIOTime(){
      if(iter){
        return io_time + iter->GetIOTime();
      }
      return io_time;
    }
    int GetRandomRead(){
      if(iter){
        return random_read_4k + iter->GetRandomRead();
      }
      return random_read_4k;
    }
    int GetRandomWrite(){
      if(iter){
        return random_write_4k + iter->GetRandomWrite();
      }
      return random_write_4k;
    }
    int GetSeqRead(){
      if(iter){
        return seq_read_4k + iter->GetSeqRead();
      }
      return seq_read_4k;
    }
    int GetSeqWrite(){
      if(iter){
        return seq_write_4k + iter->GetSeqWrite();
      }
      return seq_write_4k;
    }
  private:
    std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> indexes;
    std::vector<tsl::robin_set<TagT>>* deleted_tags_vec;
    std::shared_ptr<DiskIndexDataIterator<T, TagT>> iter;
    int cur = 0;
    double io_time = 0;
    int random_read_4k = 0;
    int seq_read_4k = 0;
    int random_write_4k = 0;
    int seq_write_4k = 0;
};
} // namespace lsmidx
