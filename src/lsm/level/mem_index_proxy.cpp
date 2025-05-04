#include "lsm/level/level_index.h"
#include "timer.h"
namespace lsmidx{
  template<typename T, typename TagT>
void InMemIndexProxy<T, TagT>::GetActiveTags(tsl::robin_set<TagT>& active_tags){
    tsl::robin_set<TagT> tags;
    this->index->get_active_tags(tags);
    for(auto& tag : tags){
        active_tags.insert(tag);
    }
    return;
}
template<typename T, typename TagT>
InMemIndexProxy<T, TagT>::InMemIndexProxy(diskann::Metric dist_metric, std::string working_dir, int idx,size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_mem, bool is_single_file_index):LevelIndex<T, TagT>(IndexType::IN_MEM_DISKANN, paras_mem, merge_thresh, 0, dims, is_single_file_index, dist_metric){
    // 初始化基本参数
    this->index_prefix = working_dir + '/' + lsmidx::config::mem_index_name + "_" + std::to_string(idx);

    // 初始化index
    this->index = std::make_shared<diskann::Index<T, TagT>>(
        dist_metric, dims, merge_thresh * 2, true/*dynamic index*/, is_single_file_index, true /*enable tags*/
    );

    /**
     * 尝试加载数据
    */
    if(file_exists(this->index_prefix)){
        this->index->load(this->index_prefix.c_str());
    }else{
        this->index->enable_delete();
    }
}
template<typename T, typename TagT>
void InMemIndexProxy<T, TagT>::KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, std::vector<lsmidx::TagDeleter<TagT>*> exclude_set_list, [[maybe_unused]] diskann::QueryStats * stats){
  uint64_t search_L = options.search_L;
  uint64_t k = options.K;
  std::vector<diskann::Neighbor_Tag<TagT>> local_res;
  if(index->get_num_points() > 0){
    index->search(query, (uint32_t)k, (uint32_t)search_L, local_res);
  }
  for(auto& nbr : local_res){
    bool deleted = false;
    for(auto& exclude_set : exclude_set_list){
        if(exclude_set->IsDelete(nbr.tag)){
            deleted = true;
            break;
        }
    }
    if(deleted){
        continue;
    }
    res.emplace_back(nbr);
  }
}

template<typename T, typename TagT>
int InMemIndexProxy<T, TagT>::LazyDelete(TagT tag){
    this->delete_tag_set.Insert(tag);
    this->index->lazy_delete(tag);
    return 0;
}

template<typename T, typename TagT>
int InMemIndexProxy<T, TagT>::Put([[maybe_unused]] const WriteOptions& options, const VecSlice<T>& key, const TagT& tag, diskann::InsertStats * stats){
    const T* point = key.data();

    if(this->index->get_num_points() >= this->index->return_max_points()){
        diskann::cout << "Capacity exceeded" << std::endl;
        return -2;
    }
    if(index->insert_point(point, *this->paras.get(), tag, stats) != 0){
        diskann::cout << "Could not insert point with tag " << tag << std::endl;
        return -3;
    }
    return 0;
}

template<typename T, typename TagT>
std::string InMemIndexProxy<T, TagT>::SaveIndex(){
    this->index->save(this->index_prefix.c_str());
    this->delete_tag_set.Save(this->index_prefix);
    return this->index_prefix;
}
template<typename T, typename TagT>
void InMemIndexProxy<T, TagT>::ClearIndex(){
    this->index->clear_index();
    this->index->enable_delete();
    this->delete_tag_set.Clear();
}
template<typename T, typename TagT>
int InMemIndexProxy<T, TagT>::GetCurrentNumPoints(){
    return this->index->get_num_points();
}
template<typename T, typename TagT>
std::shared_ptr<diskann::Index<T, TagT>> InMemIndexProxy<T, TagT>::GetIndex(){
    return this->index;
}
template<typename T, typename TagT>
void InMemIndexProxy<T, TagT>::Consolidate(){
    this->index->consolidate_for_flush(*(this->GetParameter()));
}
template<typename T, typename TagT>
std::string InMemIndexProxy<T, TagT>::ReportIndexInfo(){
    std::ostringstream oss;
    oss << '[';
    oss << "prefix: " << this->GetIndexPrefix() << "; ";
    oss << this->index->status_str();
    oss << ']';
    return oss.str();
}
// template class instantiations
template class InMemIndexProxy<float, uint32_t>;
template class InMemIndexProxy<uint8_t, uint32_t>;
template class InMemIndexProxy<int8_t, uint32_t>;
template class InMemIndexProxy<float, int64_t>;
template class InMemIndexProxy<uint8_t, int64_t>;
template class InMemIndexProxy<int8_t, int64_t>;
template class InMemIndexProxy<float, uint64_t>;
template class InMemIndexProxy<uint8_t, uint64_t>;
template class InMemIndexProxy<int8_t, uint64_t>;
} // namespace lsmidx