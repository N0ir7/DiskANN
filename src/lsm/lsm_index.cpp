#include "lsm/lsm_index.h"
#include "Neighbor_Tag.h"
#include "index.h"
#include "pq_flash_index.h"
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
    for(int i = 0; i <= num ; i++){
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
    std::vector<tsl::robin_set<TagT>> disk_res;
    tsl::robin_set<TagT> mem_res;
    size_t size = this->disk_indexes.size();
    disk_res.resize(size);
    auto mem_lock = this->GetMemLevelReadLock();
    std::vector<std::shared_lock<std::shared_mutex>> disk_locks;
    disk_locks.reserve(size);
    for (size_t level = 0; level < size; level++){
        disk_locks.emplace_back(this->GetDiskLevelReadLock(level));
    }
    this->mem_index->GetActiveTags(mem_res);
    
    for (size_t level = 0; level < size; level++){
        this->disk_indexes[level]->GetActiveTags(disk_res[level]);
    }

    // filter delete tag point
    tsl::robin_set<TagT> filter_res;
    for (int level = (int)size-1; level >= -1; level--){
        if(level == -1){ // mem index
            this->mem_index->FilterDeletedTags(filter_res);
            filter_res.insert(mem_res.begin(), mem_res.end());
        }else{ // disk index
            this->disk_indexes[level]->FilterDeletedTags(filter_res);
            filter_res.insert(disk_res[level].begin(), disk_res[level].end());
        }
    }
    active_tags.insert(filter_res.begin(), filter_res.end());

    return;
}
template<typename T, typename TagT>
LSMVectorIndex<T, TagT>::~LSMVectorIndex() {
    // for(auto iter : this->deleted_tags_vector){
    //     delete iter;
    // }
}

template<typename T, typename TagT>
LSMVectorIndex<T, TagT>::LSMVectorIndex(const BuildOptions& options, const std::string& working_dir, const std::string& index_name, diskann::Distance<T>* dist){
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
    mem_index = std::make_shared<MultiInMemIndexProxy<T, TagT>>(this->dist_metric, working_dir+'/'+index_name, this->dimension, lsmidx::config::mem_merge_thresh, paras_mem, this->is_single_file_index);

    // 初始化磁盘0层索引
    disk_indexes.emplace_back(std::make_shared<MultiPQFlashIndexProxy<T, TagT>>(this->dist_metric, working_dir+'/'+index_name, this->dimension, lsmidx::config::level0_merge_index_num_thresh, paras_disk, 0, this->is_single_file_index,this->num_search_threads));

    // 初始化磁盘1层以上索引
    for(int cur_level = 1; cur_level <= lsmidx::config::num_levels ; cur_level++){
        const int* arr = lsmidx::config::leveln_merge_thresh;
        int thresh = arr[cur_level-1];
        disk_indexes.emplace_back(std::make_shared<PQFlashIndexProxy<T, TagT>>(this->dist_metric, working_dir+'/'+index_name, 0,  this->readers[cur_level], this->dimension, thresh, paras_disk, cur_level, this->is_single_file_index,this->num_search_threads));
    }
    // 初始化一些状态量
    // this->disk_index_locks.reserve(this->disk_indexes.size());
    // for(size_t i = 0; i < this->disk_indexes.size(); i++){
    //     this->disk_index_locks.emplace_back( std::make_unique<std::shared_mutex>());
    // }
}
template<typename T, typename TagT>
int LSMVectorIndex<T, TagT>::Put(const WriteOptions& options, const VecSlice<T>& key, const TagSlice<TagT>& value, diskann::InsertStats * stats){
    const TagT tag = value.tag();
    // auto s = std::chrono::high_resolution_clock::now();
    // auto lock = this->GetMemLevelReadLock();
    // auto e = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = e - s;
    // stats->get_mem_level_read_lock_time += diff.count();

    // int ret = mem_index->Put(options, key, tag);
    // if(ret == -2){ // 说明当前已经满了，需要进行切换
    //     return -2;
    //     // mem_index->Switch();
    // }

    return mem_index->Put(options, key, tag, stats);
}

template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::Delete(const WriteOptions&, const TagSlice<TagT>& value){
    const TagT tag = value.tag();
    // auto lock = this->GetMemLevelWriteLock();
    this->mem_index->LazyDelete(tag);
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::ReportQueryInfo(int query_num){
    std::shared_ptr<lsmidx::MultiPQFlashIndexProxy<T, TagT>> from_index = std::dynamic_pointer_cast<lsmidx::MultiPQFlashIndexProxy<T, TagT>>(this->disk_indexes[0]);
    int num = lsmidx::config::level0_indexes_num;
    for(int i = 0; i < num; i++){
        from_index->GetIndexProxy(i)->ReportQueryInfo(query_num);
    }
    std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> to_index = std::dynamic_pointer_cast<lsmidx::PQFlashIndexProxy<T, TagT>>(this->disk_indexes[1]);
    to_index->ReportQueryInfo(query_num);
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::ReportIndexInfo(){
    std::ostringstream oss;
    tsl::robin_set<TagT> tags;
    this->GetActiveTags(tags);
    oss << "============LSM Index Report=============="<<std::endl;
    oss << "active tags:" << tags.size() << std::endl;
    oss << "Mem Level:" << this->mem_index->ReportIndexInfo() << std::endl;
    for(int i = 0; i < lsmidx::config::num_levels;i++){
        oss << "Disk Level " << i <<" :" << this->disk_indexes[i]->ReportIndexInfo() << std::endl;
    }
    diskann::cout<< oss.str();
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::Search(const SearchOptions& options, const VecSlice<T>& key, TagT* tags, float * distances, diskann::QueryStats * stats){
    std::set<diskann::Neighbor_Tag<TagT>> best;
    size_t size = this->disk_indexes.size();
    std::vector<std::vector<diskann::Neighbor_Tag<TagT>>> disk_res;
    std::vector<diskann::Neighbor_Tag<TagT>> mem_res; 
    disk_res.resize(size);

    /**
     * 获取每层的读锁
    */
    auto mem_lock = this->GetMemLevelReadLock();
    std::vector<std::shared_lock<std::shared_mutex>> disk_locks;
    disk_locks.reserve(size);
    for (size_t level = 0; level < size; level++){
        disk_locks.emplace_back(this->GetDiskLevelReadLock(level));
    }

    const T* query = key.data();
    const uint64_t K = options.K;

    //search each index and get top K tags
    #pragma omp parallel for schedule(dynamic, 1) num_threads(size + 1)
    for (int level = -1; level < (int)size; level++){
        if(level == -1){ // mem index
            this->mem_index->KNNQuery(query, mem_res, options, stats);
        }else{ // disk index
            this->disk_indexes[level]->KNNQuery(query, disk_res[level], options, stats);
        }
    }
    // filter delete tag point
    std::vector<diskann::Neighbor_Tag<TagT>> filter_res;
    for (int level = (int)size-1; level >= -1; level--){
        if(level == -1){ // mem index
            this->mem_index->FilterDeletedTags(filter_res);
            filter_res.insert(filter_res.end(), mem_res.begin(), mem_res.end());
        }else{ // disk index
            this->disk_indexes[level]->FilterDeletedTags(filter_res);
            filter_res.insert(filter_res.end(), disk_res[level].begin(), disk_res[level].end());
        }
    }
    best.insert(filter_res.begin(), filter_res.end());
    // 不需要额外的排序，因为set内部本身有序
    size_t pos = 0;
    for(auto iter : best){
        tags[pos] = iter.tag;
        distances[pos] = iter.dist;
        pos++;
        if (pos == K)
            break;
    }
}
template<typename T, typename TagT>
std::unique_ptr<MemFlusher<T, TagT>> LSMVectorIndex<T, TagT>::ConstructMemFlusher(){ // TODO

    return std::make_unique<MemFlusher<T, TagT>>((uint32_t) this->dimension, this->dist_comp, this->dist_metric, this->is_single_file_index);
}
template<typename T, typename TagT>
std::unique_ptr<Level0Merger<T, TagT>> LSMVectorIndex<T, TagT>::ConstructLevel0Merger(std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> from_indexes, std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> to_index){ 
    std::shared_ptr<diskann::Parameters> param = this->disk_indexes[0]->GetParameter();
    uint32_t range = param->Get<unsigned>("R");
    uint32_t l_index = param->Get<unsigned>("L");
    uint32_t maxc = param->Get<unsigned>("C");
    float alpha = param->Get<float>("alpha");
    return std::make_unique<Level0Merger<T, TagT>>((uint32_t) this->dimension, this->dist_comp, this->dist_metric, (uint32_t) this->beamwidth, range, l_index, alpha, maxc, this->is_single_file_index,
    from_indexes,
    to_index);
}
template<typename T, typename TagT>
std::unique_ptr<Mem2Level1Merger<T, TagT>> LSMVectorIndex<T, TagT>::ConstructMem2Level1Merger(){
    std::shared_ptr<diskann::Parameters> param = this->disk_indexes[0]->GetParameter();
    uint32_t range = param->Get<unsigned>("R");
    uint32_t l_index = param->Get<unsigned>("L");
    uint32_t maxc = param->Get<unsigned>("C");
    float alpha = param->Get<float>("alpha");

    return std::make_unique<Mem2Level1Merger<T, TagT>>((uint32_t) this->dimension, this->dist_comp, this->dist_metric, (uint32_t) this->beamwidth, range, l_index, alpha, maxc, this->is_single_file_index);
}
void removeOldFile(std::string& old_file){
    // Check if the old file exists, if it does, remove it
    if (file_exists(old_file)) {
        delete_file(old_file);
    }
}
void removeOldDiskIndex(std::string old_disk_index_prefix){
    std::vector<std::string> suffixes = {"_disk.index", "_pq_compressed.bin", ".index.tags", "_disk.index.tags", "_pq_pivots.bin", "_sample_data.bin", "_sample_ids.bin"};
    for(auto suffix: suffixes){
        std::string old_file = old_disk_index_prefix + suffix;
        removeOldFile(old_file);
    }
}
void removeOldMemIndex(std::string old_mem_index_prefix){
    std::vector<std::string> suffixes = {"", ".data", ".tags"};
    for(auto suffix: suffixes){
        std::string old_file = old_mem_index_prefix + suffix;
        removeOldFile(old_file);
    }
}

void OverwriteOldFile(std::string& old_file, std::string& new_file){
    // Check if the new file exists
    if (!file_exists(new_file)) {
        // diskann::cout << "Error: New disk index file does not exist." << std::endl;
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
    std::vector<std::string> suffixes = {"_disk.index", "_pq_compressed.bin", "_disk.index.tags", "_pq_pivots.bin","_medoids.bin", "_centroids.bin"};
    for(auto suffix: suffixes){
        std::string old_file = old_disk_index_prefix + suffix;
        std::string new_file = new_disk_index_prefix + suffix;
        OverwriteOldFile(old_file,new_file);
    }
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::TriggerMergeMemIndex(){
    std::shared_ptr<lsmidx::MultiPQFlashIndexProxy<T, TagT>> level0_index_ptr = std::dynamic_pointer_cast<lsmidx::MultiPQFlashIndexProxy<T, TagT>>(this->disk_indexes[0]);

    int mem_idx = -1, disk_idx = -1;
    
    // 先获取mem和level0层的读锁
    auto mem_index_read_lock = this->GetMemLevelReadLock();
    if(this->mem_index->GetNumPointsOfCur()<=lsmidx::config::mem_merge_thresh){
        return;
    }
    auto level0_index_read_lock = this->GetDiskLevelReadLock(0);
    /**
     *  进行切换 
     */
    mem_idx = this->mem_index->Switch();
    
    if(mem_idx == -1){
        diskann::cout << "can't switch mem_index"<<std::endl;
        return;
    }
    // 将切换前的Index落盘
    // std::string save_path = this->mem_index->SaveIndex(prev_idx);
    //start timer
    diskann::Timer timer;
    // 再开始merge
    // MergeMemIndex(save_path);
    std::shared_ptr<lsmidx::InMemIndexProxy<T, TagT>> from_mem_index = this->mem_index->GetIndexProxy(mem_idx);
    while((disk_idx = MergeMemIndex(from_mem_index, level0_index_ptr)) == -1){
        level0_index_read_lock.unlock();
        mem_index_read_lock.unlock();
        diskann::cout<<"No Free Slot in level0, wait for 1s"<<std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        mem_index_read_lock.lock();
        level0_index_read_lock.lock();
    }
    diskann::cout << "Merge Mem into Disk time : " << timer.elapsed()/1000 << " ms" << std::endl;
    
    /**
     *  Flush完以后，进行索引状态的切换 
     */
    // 获取mem和level0层的写锁
    // diskann::cout << "Acquiring Mem level write lock in merge_mem" << std::endl;
    // auto mem_index_write_lock = this->GetMemLevelWriteLock();
    // diskann::cout << "Acquiring level0 write lock in merge_mem" << std::endl;
    // auto level0_index_write_lock = this->GetDiskLevelWriteLock(0);

    // disk index重新加载数据
    std::string out_disk_index_prefix = level0_index_ptr->GetIndexProxy(disk_idx)->GetIndexPrefix();
    diskann::cout << "#ReloadIndex in merge mem" << disk_idx << std::endl;
    level0_index_ptr->ReloadIndex(out_disk_index_prefix, disk_idx);
    // mem_index进行清空
    diskann::cout << "Clearing sub index in merge mem "<< mem_idx << std::endl;
    this->mem_index->ClearSubIndex(mem_idx);
    removeOldMemIndex(this->mem_index->GetIndexProxy(mem_idx)->GetIndexPrefix());
    // 将 mem 和 level0层的delete tag set进行更新
    // mem更新在ClearSubIndex中会进行,level0更新会在ReloadIndex中进行
}
template<typename T, typename TagT>
int LSMVectorIndex<T, TagT>::MergeMemIndex(std::shared_ptr<lsmidx::InMemIndexProxy<T, TagT>> from_mem_index, std::shared_ptr<lsmidx::MultiPQFlashIndexProxy<T, TagT>> to_disk_index){
    // 获取level0层对应索引的读锁
    int disk_idx = to_disk_index->GetFreeIndexSlot();
    if(disk_idx == -1){
        return disk_idx;
    }
    { // 将需要flush的mem index 进行consolidate
        auto mem_write_lock = from_mem_index->GetWriteLock();
        from_mem_index->Consolidate();
    }
    // 获取对应内存索引读锁
    auto mem_read_lock = from_mem_index->GetReadLock();

    
    auto disk_read_lock = to_disk_index->GetSubReadLock(disk_idx);

    std::string disk_index_prefix = to_disk_index->GetIndexPrefix();
    std::string out_disk_index_prefix = to_disk_index->GetIndexProxy(disk_idx)->GetIndexPrefix();
    
    
    // 构建一个MemFlusher并进行Merge
    std::unique_ptr<lsmidx::MemFlusher<T, TagT>> flusher = this->ConstructMemFlusher();
    
    flusher->flush(from_mem_index, out_disk_index_prefix);
    diskann::cout << "Flush done to prefix:"<< out_disk_index_prefix<< std::endl; 

    return disk_idx;
}
int ExtractNumber(const std::string& str) {
    // 找到最后一个下划线的位置
    size_t pos = str.rfind('_');
    if (pos != std::string::npos) {
        // 提取下划线后的部分并转换为整数
        return std::stoi(str.substr(pos + 1));  // 从下划线后开始取字符串并转为整数
    }
    return -1; // 如果没有找到下划线，返回错误值
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::TriggerMergeDiskIndex(int level){ // level must be equal to 0
    if(level < 0){
        diskann::cout  << "merge level must >= 0" << std::endl;
        return;
    }
    // 获取Level层的读锁
    std::shared_ptr<lsmidx::MultiPQFlashIndexProxy<T, TagT>> from_index = std::dynamic_pointer_cast<lsmidx::MultiPQFlashIndexProxy<T, TagT>>(this->disk_indexes[level]);
    auto from_level_read_lock = from_index->GetReadLock();
    std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> from_indexes = from_index->GetNonFreeIndexList();
    if(from_indexes.size()<lsmidx::config::level0_merge_index_num_thresh){
        return;
    }
    // std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> from_indexes_raw = from_index->GetNonFreeIndexList();
    // if(from_indexes_raw.size()<=0){
    //     return;
    // }
    // std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> from_indexes;
    // from_indexes.emplace_back(from_indexes_raw[0]);
    // 获取level+1层的读锁
    std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> to_index = std::dynamic_pointer_cast<lsmidx::PQFlashIndexProxy<T, TagT>>(this->disk_indexes[level + 1]);
    auto to_level_read_lock = to_index->GetReadLock();
    //start timer
    diskann::Timer timer;
    MergeDiskIndex(from_indexes, to_index);
    diskann::cout << "Merge level " << level << "to level" << level + 1 << " time : " << timer.elapsed()/1000 << " ms" << std::endl;
    to_level_read_lock.unlock();

    // 重新加载数据
    diskann::cout << "#ReloadIndex in merge disk" << std::endl;
    std::string to_disk_index_prefix = to_index->GetIndexPrefix();
    auto to_level_write_lock = to_index->GetWriteLock();
    to_index->ReloadIndex(to_disk_index_prefix);

    // 删除from索引
    std::vector<int> remove_idx_vec;
    for(auto from_disk_index: from_indexes){
        std::string from_disk_index_prefix = from_disk_index->GetIndexPrefix();
        removeOldDiskIndex(from_disk_index_prefix);
        int idx = ExtractNumber(from_disk_index_prefix);
        remove_idx_vec.emplace_back(idx);
    }
    // 先懒删除(防止抢锁)
    from_index->BatchLazyDeleteIndex(remove_idx_vec);
    to_level_write_lock.unlock();
    // 再实际删除
    for(auto idx : remove_idx_vec){
        diskann::cout << "Clearing sub index in merge disk"<< idx << std::endl;
        from_index->ClearSubIndex(idx);
    }
    from_index->RefreshDeleteTagSet();
}

template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::MergeDiskIndex(std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>>& from_indexes, std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> to_index){
    // 获取合并涉及文件索引前缀名
    std::string to_disk_index_prefix = to_index->GetIndexPrefix();
    std::string out_disk_index_prefix = to_disk_index_prefix +"_merge";
    std::string tmp_folder = this->working_dir + "/tmp/";
    std::vector<std::string> from_disk_index_vec;
    
    // 构建一个levelMerger并进行Merge
    
    // 如果是Level0，就把Level0现有的所有index往下merge(后续可以考虑只merge一部分)
    // int thresh = lsmidx::config::level0_merge_index_num_thresh;
    
    // acquire locks
    std::vector<std::shared_lock<std::shared_mutex>> lock_vec;
    for(size_t i = 0; i < from_indexes.size(); i++){
        lock_vec.emplace_back(from_indexes[i]->GetReadLock());
    }
    std::unique_ptr<Level0Merger<T, TagT>> merger = this->ConstructLevel0Merger(from_indexes, to_index);
    merger->merge(out_disk_index_prefix, tmp_folder);
    diskann::cout << "Merge done" << std::endl;
    
    // 进行合并后的磁盘索引的替换
    
    // 先获取level和level+1层的写锁
    // diskann::cout << "Acquiring level0 write lock in merge disk" << std::endl;
    // auto from_level_write_lock = from_index->GetWriteLock();
    // diskann::cout << "Acquiring level1 write lock in merge disk" << std::endl;
    // auto to_level_write_lock = to_index->GetWriteLock();
    
    // 删除to原本的索引文件，并将新索引重命名
    OverwriteOldIndex(to_disk_index_prefix, out_disk_index_prefix);
}
template<typename T, typename TagT>
std::unique_ptr<LevelNMerger<T, TagT>> LSMVectorIndex<T, TagT>::ConstructLevelNMerger(int to_level){
    std::shared_ptr<diskann::Parameters> param = this->disk_indexes[to_level-1]->GetParameter();
    uint32_t range = param->Get<unsigned>("R");
    uint32_t l_index = param->Get<unsigned>("L");
    uint32_t maxc = param->Get<unsigned>("C");
    float alpha = param->Get<float>("alpha");
    std::unique_ptr<LevelNMerger<T, TagT>> merger = std::make_unique<LevelNMerger<T, TagT>>((uint32_t) this->dimension, this->dist_comp, this->dist_metric, (uint32_t) this->beamwidth, range, l_index, alpha, maxc, this->is_single_file_index);

    return merger;
}
template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::GetMedoid(std::vector<TagT>& medoid_vec){
    std::shared_ptr<lsmidx::MultiPQFlashIndexProxy<T, TagT>> level0_index = std::dynamic_pointer_cast<lsmidx::MultiPQFlashIndexProxy<T, TagT>>(this->disk_indexes[0]);
    std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> level1_index = std::dynamic_pointer_cast<lsmidx::PQFlashIndexProxy<T, TagT>>(this->disk_indexes[1]);
    level0_index->GetMedoids(medoid_vec);
    level1_index->GetMedoids(medoid_vec);
}
template<typename T, typename TagT>
std::shared_lock<std::shared_mutex> LSMVectorIndex<T, TagT>::GetMemLevelReadLock(){
    return mem_index->GetReadLock();
}
template<typename T, typename TagT>
std::unique_lock<std::shared_mutex> LSMVectorIndex<T, TagT>::GetMemLevelWriteLock(){
    return mem_index->GetWriteLock();
}
template<typename T, typename TagT>
std::shared_lock<std::shared_mutex> LSMVectorIndex<T, TagT>::GetDiskLevelReadLock(int level){
    return disk_indexes[level]->GetReadLock();
}
template<typename T, typename TagT>
std::unique_lock<std::shared_mutex> LSMVectorIndex<T, TagT>::GetDiskLevelWriteLock(int level){
    return disk_indexes[level]->GetWriteLock();
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
