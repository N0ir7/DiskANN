#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include "pq_flash_index.h"
#include "concurrent_queue.h"

namespace lsmidx
{
struct DiskIndexFileMeta{
  std::string data_path;
  std::string tag_path;
  std::string pq_coords_path;
  std::string pq_table_path;
  std::string medoids_file_path;
  std::string centroids_file_path;
  std::string index_prefix_path;
  bool is_single_file;
  DiskIndexFileMeta(){};
  DiskIndexFileMeta(std::string data_path,std::string tag_path,std::string pq_coords_path,std::string pq_table_path)
  :data_path(data_path),tag_path(tag_path),pq_coords_path(pq_coords_path),pq_table_path(pq_coords_path){};

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
          output_writer_(std::move(other.output_writer_)),
          index_(std::move(other.index_)),
          disk_nodes_(std::move(other.disk_nodes_)),
          local_offset_(other.local_offset_),
          cur_start_id_(other.cur_start_id_),
          next_start_id_(other.next_start_id_),
          node_need_flush_back_(other.node_need_flush_back_),
          pq_need_flush_back_(other.pq_need_flush_back_),
          tag_need_flush_back_(other.tag_need_flush_back_),
          buf_(other.buf_) {
        other.buf_ = nullptr; // 防止悬空指针
    }

    // 禁用复制构造函数
    DiskIndexDataIterator(const DiskIndexDataIterator&) = delete;

    // 移动赋值运算符（如果需要）
    DiskIndexDataIterator& operator=(DiskIndexDataIterator&& other) noexcept {
        if (this != &other) {
            index_file_meta_ = std::move(other.index_file_meta_);
            output_writer_ = std::move(other.output_writer_);
            index_ = std::move(other.index_);
            disk_nodes_ = std::move(other.disk_nodes_);
            local_offset_ = other.local_offset_;
            cur_start_id_ = other.cur_start_id_;
            next_start_id_ = other.next_start_id_;
            node_need_flush_back_ = other.node_need_flush_back_;
            pq_need_flush_back_ = other.pq_need_flush_back_;
            tag_need_flush_back_ = other.tag_need_flush_back_;
            buf_ = other.buf_;
            other.buf_ = nullptr;
        }
        return *this;
    }
    
    void Init(bool read_only, std::string output_data_path = "");
    std::tuple<diskann::DiskNode<T> *, uint8_t *, TagT*> Next();
    std::tuple<std::vector<diskann::DiskNode<T>> *,uint8_t *, TagT*> NextBatch();
    bool HasNext();
    bool HasNextBatch();
    void NotifyFlushBack();
    void TryFlushBack();
    void NotifyNodeFlushBack();
    void NotifyPQCoordFlushBack();
    void NotifyTagFlushBack();
  private:
    DiskIndexFileMeta index_file_meta_;
    std::unique_ptr<std::ofstream> output_writer_;
    std::shared_ptr<diskann::PQFlashIndex<T, TagT>> index_;
    std::vector<diskann::DiskNode<T>> disk_nodes_;
    uint32_t local_offset_ = 0;
    uint32_t cur_start_id_ = 0;
    uint32_t next_start_id_ = 0;
    bool node_need_flush_back_ = false;
    bool pq_need_flush_back_ = false;
    bool tag_need_flush_back_ = false;
    char * buf_ = nullptr;
    void NodeFlushBack();
    void PQCoordFlushBack();
    void TagFlushBack();
    void DumpToDisk(const uint32_t start_id,
                    const char *   buf,
                    const uint32_t n_sectors,
                    std::ofstream & output_writer);
};
} // namespace lsmidx
