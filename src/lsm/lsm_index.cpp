#include "lsm/lsm_index.h"
#include "Neighbor_Tag.h"
#include "index.h"
#include "pq_flash_index.h"
#include "lsm/level0_merger.h"
#include "lsm/leveln_merger.h"
#include "timer.h"
#include "utils.h"
#include <omp.h>
namespace lsmidx
{
void SetMemIndexBuildParams(const diskann::Parameters& parameters, std::shared_ptr<diskann::Parameters>& mem_params){
    // 设置内存索引参数
    mem_params->Set<unsigned>("L", parameters.Get<unsigned>("L_mem"));
    mem_params->Set<unsigned>("R", parameters.Get<unsigned>("R_mem"));
    mem_params->Set<unsigned>("C", parameters.Get<unsigned>("C"));
    mem_params->Set<float>("alpha", parameters.Get<float>("alpha_mem"));
    mem_params->Set<unsigned>("num_rnds", 2);
    mem_params->Set<bool>("saturate_graph", 0);
}
void SetDiskIndexBuildParams(const diskann::Parameters& parameters, std::shared_ptr<diskann::Parameters>& disk_params){
    // 设置内存索引参数
    disk_params->Set<unsigned>("L", parameters.Get<unsigned>("L_disk"));
    disk_params->Set<unsigned>("R", parameters.Get<unsigned>("R_disk"));
    disk_params->Set<unsigned>("C", parameters.Get<unsigned>("C"));
    disk_params->Set<float>("alpha", parameters.Get<float>("alpha_disk"));
    disk_params->Set<unsigned>("num_rnds", 2);
    disk_params->Set<bool>("saturate_graph", 0);
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::SetSeachParams(const diskann::Parameters& parameters){
    // BeamSearch的宽度
    this->beamwidth = parameters.Get<uint32_t>("beamwidth");
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::SetSystemParams(const BuildOptions& options){
    const diskann::Parameters& params = options.params;
    // 搜索线程数
    this->num_search_threads = params.Get<_u32>("num_search_threads");
    // 缓存节点数量
    this->num_nodes_to_cache = params.Get<_u32>("nodes_to_cache");

    this->dimension = options.dimension;
    this->is_single_file_index = options.is_single_file_index;
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::SetDistanceFunction(diskann::Distance<T>* dist, diskann::Metric dist_metric){
    
    this->dist_comp = dist;
    
    this->dist_metric = dist_metric;
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::SetReader(){
    int num = lsmidx::config::num_levels;
    this->readers.reserve(num);
    for(int i = 0; i < num ; i++){
        #ifdef _WINDOWS
        #ifndef USE_BING_INFRA
            this->readers.emplace_back(new WindowsAlignedFileReader());
        #else
           this->readers.emplace_back(new diskann::BingAlignedFileReader());
        #endif
        #else
            this->readers.emplace_back(new LinuxAlignedFileReader());
        #endif
    }
    
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::GetActiveTags(tsl::robin_set<TagT>& active_tags){
    this->mem_index->GetActiveTags(active_tags);
    for(auto& disk_index : disk_indexes){
        disk_index->GetActiveTags(active_tags);
    }
    this->global_in_mem_delete_tag_set->FilterDeletedTags(active_tags);
    return;
}
template<typename T, typename TagT>
LSMVectorIndex<T, TagT>::~LSMVectorIndex() {
    for(auto iter : this->deleted_tags_vector){
        delete iter;
    }
}

template<typename T, typename TagT>
LSMVectorIndex<T, TagT>::LSMVectorIndex(const BuildOptions& options, const std::string& working_dir, const std::string& index_name, diskann::Distance<T>* dist):switching_disk(false), check_switch_index(false){
    // diskann::Parameters paras_mem;
    // diskann::Parameters paras_disk;
    this->paras_mem = std::make_shared<diskann::Parameters>();
    this->paras_disk = std::make_shared<diskann::Parameters>();
    const diskann::Parameters& params = options.params;
    // 设置内存索引参数
    SetMemIndexBuildParams(params, this->paras_mem);
    // 设置磁盘索引参数
    SetDiskIndexBuildParams(params, this->paras_disk);
    // 设置其他参数
    SetSeachParams(params);
    SetSystemParams(options);
    SetDistanceFunction(dist, options.dist_metric);
    SetReader();
    // 初始化文件路径
    this->working_dir = working_dir;
    this->index_name = index_name;
    /**
     * 初始化各层索引
     */

    // 初始化内存索引
    mem_index = std::make_shared<InMemIndexProxy<T, TagT>>(this->dist_metric, working_dir+'/'+index_name, this->dimension, lsmidx::config::level0_merge_thresh, paras_mem, this->is_single_file_index);

    // 初始化in_memory_delete_tag_set
    this->global_in_mem_delete_tag_set = std::make_unique<TagDeleter<TagT>>(lsmidx::config::mem_indexes_num);

    // 初始化磁盘各层索引
    for(int cur_level = 1; cur_level <= lsmidx::config::num_levels ; cur_level++){
        const int* arr = lsmidx::config::leveln_merge_thresh;
        int thresh = arr[cur_level-1];
        disk_indexes.emplace_back(std::make_shared<PQFlashIndexProxy<T, TagT>>(this->dist_metric, working_dir+'/'+index_name, this->readers[cur_level-1], this->dimension, thresh, paras_disk, cur_level, this->is_single_file_index,this->num_search_threads));
    }
    this->disk_locks.reserve(this->disk_indexes.size());
    for(size_t i = 0; i < this->disk_indexes.size(); i++){
        this->disk_locks.emplace_back( std::make_unique<std::shared_timed_mutex>());
    }
}
template<typename T, typename TagT>
int LSMVectorIndex<T, TagT>::Put(const WriteOptions& options, const VecSlice<T>& key, const TagSlice<TagT>& value){
    const TagT tag = value.tag();
    while(check_switch_index.load()){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::shared_lock<std::shared_timed_mutex> lock(index_lock);
    mem_index->Put(options, key,tag);
    return -2;
}

template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::Delete(const WriteOptions&, const TagSlice<TagT>& value){
    const TagT tag = value.tag();
    std::unique_lock<std::shared_timed_mutex> lock(delete_lock);

    // 如果global删除成功了，再去删除本地的
    if(this->global_in_mem_delete_tag_set->Insert(tag)){
        this->mem_index->LazyDelete(tag);
    }
}

template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::Search(const SearchOptions& options, const VecSlice<T>& key, TagT* tags, float * distances, diskann::QueryStats * stats){
    std::set<diskann::Neighbor_Tag<TagT>> best;
    size_t size = this->disk_indexes.size() + 1;
    const T* query = key.data();
    const uint64_t K = options.K;

    //search each index and get top K tags
    #pragma omp parallel for schedule(dynamic, 1) num_threads(size)
    for (size_t level = 0; level < size; level++){
        std::vector<diskann::Neighbor_Tag<TagT>> res;
        if(level == 0){ // mem index
            this->mem_index->KNNQuery(query, res, options, stats);
        }else{ // disk index
            size_t idx = level - 1;
            std::shared_lock<std::shared_timed_mutex> lock(*(this->disk_locks[idx]));
            this->disk_indexes[idx]->KNNQuery(query, res, options, stats);
        }
        
        #pragma omp critical
        best.insert(res.begin(),res.end());
    }

    std::vector<diskann::Neighbor_Tag<TagT>> best_vec;
    // 不需要额外的排序，因为set内部本身有序
    for(auto iter : best)
        best_vec.emplace_back(iter);
    
    { //aggregate results, sort and pick top K candidates
        std::shared_lock<std::shared_timed_mutex> lock(delete_lock);
        size_t pos = 0;
        for (auto iter : best_vec) {
            if(!this->global_in_mem_delete_tag_set->IsDelete(iter.tag)) {
                tags[pos] = iter.tag;
                distances[pos] = iter.dist;
                pos++;
                // result.emplace_back(iter.tag, iter.dist);
            }
            if (pos == K)
                break;
        }
    }
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::TriggerMergeMemIndex(){
    if(this->mem_index->GetCurrentNumPoints()<=0){
        return;
    }
    /**
     *  进行切换
     */
    { // 先进行delete_tag_set 切换
        std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
        std::vector<TagT>* delete_tag_list = this->global_in_mem_delete_tag_set->Switch();

        // 保存删除的标签到deleted_tags_vector中
        this->deleted_tags_vector.clear();
        this->deleted_tags_vector.push_back(delete_tag_list);
    }
    int prev_idx = -1;
    { // 再进行index切换
        bool expected_value = false;
        check_switch_index.compare_exchange_strong(expected_value, true);
        std::unique_lock<std::shared_timed_mutex> lock(index_lock);
        prev_idx = this->mem_index->Switch();
        // 恢复 check_switch_index 为 false，允许下次切换
        expected_value = true;
        check_switch_index.compare_exchange_strong(expected_value, false);
    }
    
    if(prev_idx == -1){
        diskann::cout << "can't switch mem_index"<<std::endl;
        return;
    }
    // 将切换前的Index落盘
    std::string save_path = this->mem_index->SaveIndex(prev_idx);
    //start timer
    diskann::Timer timer;
    // 再开始merge  
    MergeMemIndex(save_path);
    diskann::cout << "Merge time : " << timer.elapsed()/1000 << " ms" << std::endl;

    // Merge完以后，将之前的index进行clear
    this->mem_index->ClearIndex(prev_idx);
    
}
template<typename T, typename TagT>
std::unique_ptr<Level0Merger<T, TagT>> LSMVectorIndex<T, TagT>::ConstructLevel0Merger(){
    std::shared_ptr<diskann::Parameters> param = this->disk_indexes[0]->GetParameter();
    uint32_t range = param->Get<unsigned>("R");
    uint32_t l_index = param->Get<unsigned>("L");
    uint32_t maxc = param->Get<unsigned>("C");
    float alpha = param->Get<float>("alpha");
    std::unique_ptr<Level0Merger<T, TagT>> merger = std::make_unique<Level0Merger<T, TagT>>((uint32_t) this->dimension, this->dist_comp, this->dist_metric, (uint32_t) this->beamwidth, range, l_index, alpha, maxc, this->is_single_file_index);

    return merger;
}
void OverwriteOldFile(std::string& old_file, std::string& new_file){
    // Check if the new file exists
    if (!file_exists(new_file)) {
        diskann::cout << "Error: New disk index file does not exist." << std::endl;
        return;
    }

    // Check if the old file exists, if it does, remove it
    if (file_exists(old_file)) {
        delete_file(old_file);
    }

    // Rename the new file to the old file's name (overwrite the old one)
    rename_file(old_file, new_file);

    diskann::cout << "Successfully overwrote " << old_file << " with " << new_file << std::endl;
}
void OverwriteOldIndex(std::string old_disk_index_prefix, std::string new_disk_index_prefix){
    std::vector<std::string> suffixes = {"_disk.index", "_pq_compressed.bin", ".tags", "_pq_pivots.bin","_medoids.bin", "_centroids.bin"};
    for(auto suffix: suffixes){
        std::string old_file = old_disk_index_prefix + suffix;
        std::string new_file = new_disk_index_prefix + suffix;
        OverwriteOldFile(old_file,new_file);
    }
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::MergeMemIndex(std::string mem_index_path){
    std::string in_disk_index_prefix = this->disk_indexes[0]->GetIndexPrefix();
    std::string out_disk_index_prefix = in_disk_index_prefix +"_merge";
    std::string tmp_folder = this->working_dir + "/tmp";
    // 构建一个level0Merger并进行Merge
    std::unique_ptr<Level0Merger<T, TagT>> merger = this->ConstructLevel0Merger();
    
    merger->merge(in_disk_index_prefix.c_str(), {mem_index_path}, out_disk_index_prefix.c_str(), this->deleted_tags_vector, tmp_folder);
    diskann::cout << "Merge done" << std::endl;
    // 进行合并后的磁盘索引的替换
    {
        std::unique_lock<std::shared_timed_mutex> lock(*(this->disk_locks[0]));
        bool expected_value = false;
        if (this->switching_disk.compare_exchange_strong(expected_value, true)) {
            diskann::cout << "Switching to latest merged disk index " << std::endl;
        } else {
            diskann::cout << "Failed to switch" << std::endl;
        }

        // 删除原本的索引，并将新索引重命名
        OverwriteOldIndex(in_disk_index_prefix, out_disk_index_prefix);
        // 重新加载数据
        std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> pqFlashIndexPtr = std::dynamic_pointer_cast<lsmidx::PQFlashIndexProxy<T, TagT>>(this->disk_indexes[0]);
        pqFlashIndexPtr->ReloadIndex(in_disk_index_prefix);
        // 将switching_disk切换回去
        expected_value = true;
        this->switching_disk.compare_exchange_strong(expected_value,false);
    }
}
// template class instantiations
  template class LSMVectorIndex<float, uint32_t>;
  template class LSMVectorIndex<uint8_t, uint32_t>;
  template class LSMVectorIndex<int8_t, uint32_t>;
  template class LSMVectorIndex<float, int64_t>;
  template class LSMVectorIndex<uint8_t, int64_t>;
  template class LSMVectorIndex<int8_t, int64_t>;
  template class LSMVectorIndex<float, uint64_t>;
  template class LSMVectorIndex<uint8_t, uint64_t>;
  template class LSMVectorIndex<int8_t, uint64_t>;
} // namespace lsmidx
