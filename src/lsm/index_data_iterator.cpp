#include "lsm/index_data_iterator.h"

namespace lsmidx
{
template<typename T, typename TagT>
DiskIndexDataIterator<T, TagT>::~DiskIndexDataIterator(){
  TryFlushBack();
  aligned_free((void *) buf_);
  if(output_writer_){
    output_writer_->close();
  }
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::Init(bool read_only,std::string output_data_path){
  std::string output_path = this->index_file_meta_.data_path;
  if(!output_data_path.empty()){
    output_path = output_data_path;
  }
  if(!read_only){
    output_writer_ = std::make_unique<std::ofstream>(output_path, std::ios::out | std::ios::binary);
    // TODO: 写元信息
    output_writer_->seekp(SECTOR_LEN, std::ios::beg);
  }
  alloc_aligned((void **) &this->buf_, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);
}
template<typename T, typename TagT>
std::tuple<diskann::DiskNode<T>*, uint8_t *, TagT*> DiskIndexDataIterator<T, TagT>::Next(){
  if(this->local_offset_< this->disk_nodes_.size()){
    auto res = this->index_->get_pq_config();
    uint64_t pq_nchunks = res.second;
    uint8_t * pq_data = res.first;
    uint64_t cur_offset = (uint64_t) (this->cur_start_id_ + this->local_offset_);
    const uint64_t pq_offset = cur_offset * pq_nchunks;
    TagT* tag = &this->index_->get_tags()[cur_offset];

    return {&this->disk_nodes_[this->local_offset_++], pq_data + pq_offset,tag};
  }
  if(node_need_flush_back_){
    NodeFlushBack();
  }
  this->disk_nodes_.clear();
  this->local_offset_ = 0;
  this->cur_start_id_ = this->next_start_id_; 
  this->next_start_id_ = this->index_->merge_read(this->disk_nodes_, this->cur_start_id_,
                                                SECTORS_PER_MERGE, this->buf_);
  return Next();
}

template<typename T, typename TagT>
std::tuple<std::vector<diskann::DiskNode<T>>*,uint8_t *, TagT*> NextBatch(){
  if(node_need_flush_back_){
    NodeFlushBack();
  }

  this->disk_nodes_.clear();
  this->local_offset_ = 0;
  this->cur_start_id_ = this->next_start_id_; 
  this->next_start_id_ = this->index_->merge_read(this->disk_nodes_, this->cur_start_id_,
                                                SECTORS_PER_MERGE, this->buf_);
  auto res = this->index_->get_pq_config();
  uint64_t pq_nchunks = res.second;
  uint8_t * pq_data = res.first;
  uint64_t cur_offset = (uint64_t) this->cur_start_id_;
  const uint64_t pq_offset = cur_offset * pq_nchunks;
  TagT* tag = &this->index_->get_tags()[cur_offset];
  return {&this->disk_nodes_, pq_data + pq_offset, tag};
}

template<typename T, typename TagT>
bool DiskIndexDataIterator<T, TagT>::HasNext(){
  if(this->local_offset_< this->disk_nodes.size()){
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
  if(!output_writer_){
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
  if(!output_writer){
    return;
  }
  if(!pq_need_flush_back_){
    this->pq_need_flush_back_ = true;
  }
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::NotifyNodeFlushBack(){
  if(!output_writer_){
    return;
  }
  if(!node_need_flush_back_){
    this->node_need_flush_back_ = true;
  }
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::NotifyTagFlushBack(){
  if(!output_writer_){
    return;
  }
  if(!tag_need_flush_back_){
    this->tag_need_flush_back_ = true;
  }
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::TryFlushBack(){
  if(!output_writer_){
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
  if(!output_writer_){
    return;
  }
  this->DumpToDisk(this->cur_start_id_,buf,SECTORS_PER_MERGE,this->output_writer_.get());
  output_writer.flush();
  this->node_need_flush_back_ = false;
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::PQCoordFlushBack(){
  assert(this->pq_need_flush_back_);
  diskann::cout << "Dumping full compressed PQ vectors from memory.\n";
  
  auto res = this->index_->get_pq_config();
  uint64_t pq_nchunks = res.second;
  uint8_t * pq_data = res.first;
  // re-open PQ writer
  // std::ofstream pq_writer(this->index_file_meta_.pq_coords_path, std::ios::out | std::ios::binary);
  
  // pq_writer.seekp(2 * sizeof(uint32_t), std::ios::beg);
  // // write all (old + new) PQ data to disk; no need to modify header
  // pq_writer.write((char *) pq_data,
  //                 ((uint64_t) this->index_->return_nd() * (uint64_t) pq_nchunks));
  // pq_writer.close();
  diskann::save_bin<uint8_t>(this->index_file_meta_.pq_coords_path, 
                            pq_data,
                            (uint64_t) this->index_->return_nd(),
                            pq_nchunks,
                            0);
  this->pq_need_flush_back_ = false;
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::TagFlushBack(){
  assert(this->pq_need_flush_back_);
  diskann::cout << "Dumping full compressed PQ vectors from memory.\n";
  
  TagT* tag_data = this->index_->get_tags();
  // re-open PQ writer
  // std::ofstream tag_writer(this->index_file_meta_.tag_path, std::ios::out | std::ios::binary);
  // tag_writer.seekp(2 * sizeof(uint32_t), std::ios::beg);
  // // write all (old + new) PQ data to disk; no need to modify header
  // tag_writer.write((char *) tag_data,
  //                 ((uint64_t) new_max_pts * sizeof(TagT)));
  // tag_writer.close();
  diskann::save_bin<TagT>(this->index_file_meta_.tag_path, 
                    tag_data, 
                    (uint64_t) this->index_->return_nd(), 
                    1,
                    0);
  this->tag_need_flush_back_ = false;
}

template<typename T, typename TagT>
void DiskIndexDataIterator<T, TagT>::DumpToDisk(const uint32_t start_id,
                                            const char *   buf,
                                            const uint32_t n_sectors,
                                            std::ofstream & output_writer) {
  assert(start_id % this->nnodes_per_sector == 0);
  uint32_t start_sector = (start_id / this->index_->nnodes_per_sector) + 1;
  uint64_t start_off = start_sector * (uint64_t) SECTOR_LEN;

  // seek fp
  output_writer.seekp(start_off, std::ios::beg);

  // dump
  output_writer.write(buf, (uint64_t) n_sectors * (uint64_t) SECTOR_LEN);

  uint64_t nb_written =
      (uint64_t) output_writer.tellp() - (uint64_t) start_off;
  if (nb_written != (uint64_t) n_sectors * (uint64_t) SECTOR_LEN) {
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
