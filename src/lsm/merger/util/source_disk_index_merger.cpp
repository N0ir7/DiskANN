#include "lsm/merger/util/disk_index_merger.h"

namespace lsmidx{
template<typename T, typename TagT>
SourceDiskIndexMerger<T, TagT>::SourceDiskIndexMerger(std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> &src_indexes):indexes(src_indexes){
  tsl::robin_set<TagT> tmp_delete_set;
  size_t size = this->indexes.size();
  this->deleted_tags_vec.resize(size);
  this->total_num = 0;
  for(int i = size - 1; i >= 0; i--){
    { // 读取tag_list，统计有效的向量数
      std::shared_ptr<diskann::PQFlashIndex<T, TagT>> index = this->indexes[i]->GetIndex();
      tsl::robin_set<TagT> tag_list;
      index->get_active_tags(tag_list);
      size_t tag_num = tag_list.size();
      if(i == 0){
        this->total_num += tag_num;
      }else{
        for(auto tag : tag_list){
          if(tmp_delete_set.count(tag)){
            continue;
          }
          this->total_num++;
        }
      }
    }
    { // 读取delete_list，并进行累计，得到每个index的需要删除的tag
      this->indexes[i]->delete_tag_set.Union(tmp_delete_set);
      this->deleted_tags_vec[i] = tmp_delete_set;
    }
  }
  
}
template<typename T, typename TagT>
MultiDiskIndexDataIterator<T, TagT> SourceDiskIndexMerger<T, TagT>::GetIterator(){
  return MultiDiskIndexDataIterator<T,TagT>(this->indexes, &deleted_tags_vec);
}

template<typename T, typename TagT>
uint32_t SourceDiskIndexMerger<T, TagT>::GetNumPoints(){ 
  return total_num;
}
template<typename T, typename TagT>
tsl::robin_set<TagT>* SourceDiskIndexMerger<T, TagT>::GetDeleteTagSet(int idx){ 
  return &deleted_tags_vec[idx];
}
template<typename T, typename TagT>
tsl::robin_set<TagT>& SourceDiskIndexMerger<T, TagT>::GetDeleteTagSet(){
  return deleted_tags_vec[0];
}

template class SourceDiskIndexMerger<float, uint32_t>;
template class SourceDiskIndexMerger<uint8_t, uint32_t>;
template class SourceDiskIndexMerger<int8_t, uint32_t>;
template class SourceDiskIndexMerger<float, int64_t>;
template class SourceDiskIndexMerger<uint8_t, int64_t>;
template class SourceDiskIndexMerger<int8_t, int64_t>;
template class SourceDiskIndexMerger<float, uint64_t>;
template class SourceDiskIndexMerger<uint8_t, uint64_t>;
template class SourceDiskIndexMerger<int8_t, uint64_t>;
}