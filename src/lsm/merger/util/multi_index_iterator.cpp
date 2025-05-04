#include "lsm/merger/util/index_data_iterator.h"
#include "lsm/options.h"
#include "utils.h"

namespace lsmidx
{
  template<typename T,typename TagT>
void MultiDiskIndexDataIterator<T, TagT>::Init(){
  if(this->cur >= indexes.size()){
    return;
  }
  auto ptr = this->indexes[this->cur];
  iter = std::make_shared<DiskIndexDataIterator<T, TagT>>(DiskIndexFileMeta(ptr->GetIndexPrefix(), false),ptr->GetIndex());
  iter->Init(true, SECTORS_PER_MERGE);
}
template<typename T,typename TagT>
std::tuple<diskann::DiskNode<T> *, uint8_t *, TagT*> MultiDiskIndexDataIterator<T, TagT>::Next(){
  if(iter && iter->HasNext()){
    return iter->Next();
  }
  auto ptr = this->indexes[++this->cur];
  iter = std::make_shared<DiskIndexDataIterator<T, TagT>>(DiskIndexFileMeta(ptr->GetIndexPrefix(), false),ptr->GetIndex());
  iter->Init(true, SECTORS_PER_MERGE);
  return Next();
}
template<typename T,typename TagT>
bool MultiDiskIndexDataIterator<T, TagT>::HasNext(){
  if(iter && iter->HasNext()){
    return true;
  }

  return cur < indexes.size()-1;
}
template<typename T,typename TagT>
std::tuple<std::vector<diskann::DiskNode<T>> *,uint8_t *, TagT*> MultiDiskIndexDataIterator<T, TagT>::NextBatch(){
  if(iter && iter->HasNextBatch()){
    return iter->NextBatch();
  }
  /**
   * 更换指向下一个索引的迭代器
  */
  this->io_time += iter->GetIOTime();
  this->random_read_4k += iter->GetRandomRead();
  this->seq_read_4k += iter->GetSeqRead();
  this->random_write_4k += iter->GetRandomWrite();
  this->seq_write_4k += iter->GetSeqWrite();
  auto ptr = this->indexes[++this->cur];
  iter = std::make_shared<DiskIndexDataIterator<T, TagT>>(DiskIndexFileMeta(ptr->GetIndexPrefix(), false),ptr->GetIndex());
  iter->Init(true, SECTORS_PER_MERGE);
  return NextBatch();
}
template<typename T,typename TagT>
tsl::robin_set<TagT>* MultiDiskIndexDataIterator<T, TagT>::GetCurDeleteTagSet(){
  if(cur == indexes.size()-1 || cur >= indexes.size()){
    return nullptr;
  }
  return &((*deleted_tags_vec)[cur+1]);
}
template<typename T,typename TagT>
int MultiDiskIndexDataIterator<T, TagT>::GetCurIndexFrozenPoint(){
  return this->indexes[this->cur]->GetIndex()->return_frozen_location();
}
template<typename T,typename TagT>
bool MultiDiskIndexDataIterator<T, TagT>::HasNextBatch(){
  if(iter->HasNextBatch()){
    return true;
  }

  return cur < indexes.size()-1;
}
template class MultiDiskIndexDataIterator<float, uint32_t>;
template class MultiDiskIndexDataIterator<uint8_t, uint32_t>;
template class MultiDiskIndexDataIterator<int8_t, uint32_t>;
template class MultiDiskIndexDataIterator<float, int64_t>;
template class MultiDiskIndexDataIterator<uint8_t, int64_t>;
template class MultiDiskIndexDataIterator<int8_t, int64_t>;
template class MultiDiskIndexDataIterator<float, uint64_t>;
template class MultiDiskIndexDataIterator<uint8_t, uint64_t>;
template class MultiDiskIndexDataIterator<int8_t, uint64_t>;
}