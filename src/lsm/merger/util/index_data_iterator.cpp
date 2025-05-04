#include "lsm/merger/util/index_data_iterator.h"
#include "lsm/options.h"
#include "utils.h"

namespace lsmidx
{
template<typename T, typename TagT>
DiskIndexDataIterator<T, TagT>::~DiskIndexDataIterator(){
  TryFlushBack();
  diskann::aligned_free((void *) buf_);
  // diskann::cout << "Deconstruct a Disk Iterator of" << this->index_file_meta_.data_path <<"("<<this->sum<<")"<<std::endl;
}
template<typename T, typename TagT>
bool DiskIndexDataIterator<T, TagT>::GetNode(diskann::DiskNode<T>& node, unsigned node_id){
  if(node_id < this->cur_start_id_ || node_id >= this->next_start_id_){
    return false;
  }
  unsigned offset = node_id - this->cur_start_id_;
  node = this->disk_nodes_[offset];
  return true;
}
template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::Init(bool read_only, int sectors_per_batch, DiskIndexFileMeta* output_index_file_meta){

  // 初始化数据路径
  this->output_index_file_meta_ = this->index_file_meta_;
  if(output_index_file_meta != nullptr){
    this->output_index_file_meta_ = *output_index_file_meta;
    this->read_write_same_file_ = false;
  }
  this->sectors_per_batch = sectors_per_batch;
  diskann::cout << "Init a Disk Iterator of" << this->index_file_meta_.data_path << "; Alloc a buffer, size: "<< sectors_per_batch * SECTOR_LEN/1024/1024<<"MB; ";
  /**
   * 如果需要write，还需要初始化一个writer
  */
  if(!read_only){
    this->read_only_ = read_only;
    diskann::cout << "output data path: " 
                  << this->output_index_file_meta_.data_path
                  <<" ;output tag path: "
                  << this->output_index_file_meta_.tag_path
                  <<" ;outpt pq_compressed path: "
                  << this->output_index_file_meta_.pq_coords_path;
  }
  diskann::cout<<std::endl;
  // 分配一个读取缓冲区
  diskann::alloc_aligned((void **) &this->buf_, sectors_per_batch * SECTOR_LEN, SECTOR_LEN);
}
template<typename T, typename TagT>
std::tuple<diskann::DiskNode<T>*, uint8_t *, TagT*> DiskIndexDataIterator<T, TagT>::Next(){
  /**
   * 当前batch还没有读完，则返回当前batch的数据
  */
  if(this->local_offset_< this->disk_nodes_.size()){
    auto res = this->index_->get_pq_config();
    uint64_t pq_nchunks = res.second;
    uint8_t * pq_data = res.first;
    uint64_t cur_offset = (uint64_t) (this->cur_start_id_ + this->local_offset_);
    const uint64_t pq_offset = cur_offset * pq_nchunks;
    TagT* tag = &this->index_->get_tags()[cur_offset];

    return {&this->disk_nodes_[this->local_offset_++], pq_data + pq_offset,tag};
  }
  /**
   * 读取下一个batch的数据
  */
  NextBatch();

  return Next();
}
template<typename T, typename TagT>
std::tuple<diskann::DiskNode<T>*, uint8_t *, TagT*> DiskIndexDataIterator<T, TagT>::SeekNode(unsigned node_id){
  assert(read_write_same_file_);
  /**
   * 当前batch还没有读完，则返回当前batch的数据
  */
  if(node_id >= this->cur_start_id_ && node_id < this->next_start_id_){
    auto res = this->index_->get_pq_config();
    uint64_t pq_nchunks = res.second;
    uint8_t * pq_data = res.first;
    const uint64_t pq_offset = node_id * pq_nchunks;
    TagT* tag = &this->index_->get_tags()[node_id];
    return {&this->disk_nodes_[node_id - this->cur_start_id_], pq_data + pq_offset,tag};
  }
  /**
   * 读取下一个batch的数据
  */
  SeekBatch(node_id);

  return SeekNode(node_id);
}
template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::SeekBatch(unsigned node_id){
  // 如果前一个batch需要写回，则进行写回
  if(node_need_flush_back_){
    NodeFlushBack();
  }
  /**
   * 读取下一个batch的数据
  */
  this->disk_nodes_.clear();
  this->cur_start_id_ = (node_id / this->index_->nnodes_per_sector) * this->index_->nnodes_per_sector;
  this->local_offset_ = node_id - this->cur_start_id_;
  memset(this->buf_, 0, sectors_per_batch * SECTOR_LEN);
  auto s = std::chrono::high_resolution_clock::now();
  this->next_start_id_ = this->index_->merge_read(this->disk_nodes_, this->cur_start_id_,
                                                sectors_per_batch, this->buf_);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  this->io_time += diff.count();
  this->random_read_4k += 1;
  this->seq_read_4k += sectors_per_batch - 1;
  return;
}
template<typename T, typename TagT>
std::tuple<std::vector<diskann::DiskNode<T>>*,uint8_t *, TagT*> DiskIndexDataIterator<T, TagT>::NextBatch(){
  // 如果前一个batch需要写回，则进行写回
  if(node_need_flush_back_){
    NodeFlushBack();
  }
  /**
   * 读取下一个batch的数据
  */
  this->disk_nodes_.clear();
  this->local_offset_ = 0;
  this->cur_start_id_ = this->next_start_id_; 
  memset(this->buf_, 0, sectors_per_batch * SECTOR_LEN);
  auto s = std::chrono::high_resolution_clock::now();
  this->next_start_id_ = this->index_->merge_read(this->disk_nodes_, this->cur_start_id_,
                                                sectors_per_batch, this->buf_);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  this->io_time += diff.count();
  this->random_read_4k += 1;
  this->seq_read_4k += sectors_per_batch - 1;
  /**
   * 同时获取下一个batch对应的PQ坐标信息和tag信息
  */
  auto res = this->index_->get_pq_config();
  uint64_t pq_nchunks = res.second;
  uint8_t * pq_data = res.first;
  uint64_t cur_offset = (uint64_t) this->cur_start_id_;
  const uint64_t pq_offset = cur_offset * pq_nchunks;
  TagT* tag = &this->index_->get_tags()[cur_offset];
  diskann::cout << "read a batch from " << index_file_meta_.data_path<<"; patch size: "<< this->disk_nodes_.size()<<" nodes"<< std::endl;
  // this->sum += this->disk_nodes_.size();
  // 如果输出文件与输入文件不同，则无论是否为脏都需要写回
  if(!read_write_same_file_){
    this->NotifyFlushBack();
  }
  return {&this->disk_nodes_, pq_data + pq_offset, tag};
}

template<typename T, typename TagT>
bool DiskIndexDataIterator<T, TagT>::HasNext(){
  if(this->local_offset_< this->disk_nodes_.size()){
    return true;
  }
  return this->next_start_id_ < this->index_->return_nd();
}

template<typename T, typename TagT>
bool DiskIndexDataIterator<T, TagT>::HasNextBatch(){
  return this->next_start_id_ < this->index_->return_nd();
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::NotifyFlushBack(){
  if(read_only_){
    return;
  }
  if(!node_need_flush_back_){
    this->node_need_flush_back_ = true;
  }
  if(!pq_need_flush_back_){
    this->pq_need_flush_back_ = true;
  }
  if(!tag_need_flush_back_){
    this->tag_need_flush_back_ = true;
  }
}
template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::NotifyPQCoordFlushBack(){
  if(read_only_){
    return;
  }
  if(!pq_need_flush_back_){
    this->pq_need_flush_back_ = true;
  }
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::NotifyNodeFlushBack(){
  if(read_only_){
    return;
  }
  if(!node_need_flush_back_){
    this->node_need_flush_back_ = true;
  }
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::NotifyTagFlushBack(){
  if(read_only_){
    return;
  }
  if(!tag_need_flush_back_){
    this->tag_need_flush_back_ = true;
  }
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::TryFlushBack(){
  if(read_only_){
    return;
  }
  if(node_need_flush_back_){
    NodeFlushBack();
  }
  if(pq_need_flush_back_){
    PQCoordFlushBack();
  }
  if(tag_need_flush_back_){
    TagFlushBack();
  }
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::NodeFlushBack(){
  assert(this->node_need_flush_back_);
  if(read_only_){
    return;
  }
  auto s = std::chrono::high_resolution_clock::now();
  this->DumpToDisk(this->cur_start_id_,this->buf_,sectors_per_batch,this->output_index_file_meta_.data_path);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  this->io_time += diff.count();
  this->random_write_4k += 1;
  this->seq_write_4k += sectors_per_batch - 1;
  this->node_need_flush_back_ = false;
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::PQCoordFlushBack(){
  assert(this->pq_need_flush_back_);
  diskann::cout << "Dumping full compressed PQ vectors from memory.\n";
  
  auto res = this->index_->get_pq_config();
  uint64_t pq_nchunks = res.second;
  uint8_t * pq_data = res.first;

  auto s = std::chrono::high_resolution_clock::now();
  uint64_t bytes_written = diskann::save_bin<uint8_t>(this->output_index_file_meta_.pq_coords_path, 
                            pq_data,
                            (uint64_t) this->index_->return_nd(),
                            pq_nchunks,
                            0);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  int sectors = (bytes_written + SECTOR_LEN - 1) / SECTOR_LEN;
  this->io_time += diff.count();
  this->random_write_4k += 1;
  this->seq_write_4k += sectors - 1;
  this->pq_need_flush_back_ = false;
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::TagFlushBack(){
  assert(this->tag_need_flush_back_);
  diskann::cout << "Dumping Tags Data from memory.\n";
  
  TagT* tag_data = this->index_->get_tags();
  auto s = std::chrono::high_resolution_clock::now();
  uint64_t bytes_written = diskann::save_bin<TagT>(this->output_index_file_meta_.tag_path, 
                    tag_data, 
                    (uint64_t) this->index_->return_nd(), 
                    1,
                    0);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  int sectors = (bytes_written + SECTOR_LEN - 1) / SECTOR_LEN;
  this->io_time += diff.count();
  this->random_write_4k += 1;
  this->seq_write_4k += sectors - 1;
  this->tag_need_flush_back_ = false;
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::DumpToDisk(const uint32_t start_id,
                                            const char *   buf,
                                            const uint32_t n_sectors,
                                            std::string data_path) {
  assert(start_id % this->index_->nnodes_per_sector == 0);
  uint32_t start_sector = (start_id / this->index_->nnodes_per_sector) + 1; // 第一个sector是元信息，+1
  uint64_t start_off = start_sector * (uint64_t) SECTOR_LEN;

  // 为了保持Iterator的功能专一性，我们不在Iterator内进行文件头信息的更新
  std::ofstream output_writer;
  open_file_to_write(output_writer, data_path);
  // seek fp
  output_writer.seekp(start_off, std::ios::beg);

  // dump
   uint64_t write_sector = ROUND_UP(this->index_->return_nd() - start_id, this->index_->nnodes_per_sector) / this->index_->nnodes_per_sector;
  uint64_t bytes_to_write = std::min(write_sector * SECTOR_LEN, (uint64_t)n_sectors * SECTOR_LEN);
  output_writer.write(buf, bytes_to_write);

  // 报错
  uint64_t nb_written =
      (uint64_t) output_writer.tellp() - (uint64_t) start_off;
  output_writer.close();
  if (nb_written != bytes_to_write) {
    std::stringstream sstream;
    sstream << "ERROR!!! Wrote " << nb_written << " bytes to disk instead of "
            << ((uint64_t) n_sectors) * SECTOR_LEN;
    diskann::cerr << sstream.str() << std::endl;
    throw diskann::ANNException(sstream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  }
}

template class DiskIndexDataIterator<float, uint32_t>;
template class DiskIndexDataIterator<uint8_t, uint32_t>;
template class DiskIndexDataIterator<int8_t, uint32_t>;
template class DiskIndexDataIterator<float, int64_t>;
template class DiskIndexDataIterator<uint8_t, int64_t>;
template class DiskIndexDataIterator<int8_t, int64_t>;
template class DiskIndexDataIterator<float, uint64_t>;
template class DiskIndexDataIterator<uint8_t, uint64_t>;
template class DiskIndexDataIterator<int8_t, uint64_t>;

} // namespace lsmidx
