#include <lsm_index.h>
#include <status.h>
#include "Neighbor_Tag.h"
#include "index.h"
#include "pq_flash_index.h"
namespace lsmidx
{
template<typename T, typename TagT>
LSMVectorIndex<T, TagT>::~LSMVectorIndex() {
    // put in destructor code
}
template<typename T, typename TagT>
int LSMVectorIndex<T, TagT>::Put(const WriteOptions&, const Slice<T,TagT>& key,
              const TagT& value){
    const T* point = key.data();
    const TagT tag = key.tag();
    while(check_switch_index_.load()){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::shared_lock<std::shared_timed_mutex> lock(index_lock_);
    if((active_index_ == 0) && (active_0_.load() == false)){
        diskann::cout << "Active index indicated as _mem_index_0 but it cannot accept insertions" << std::endl;
        return -1;
    }
    if((active_index_ == 1) && (active_1_.load() == false)){
        diskann::cout << "Active index indicated as _mem_index_1 but it cannot accept insertions" << std::endl;
        return -1;
    }

    if(active_index_ == 0){
        if(mem_index_0_->get_num_points() < mem_index_0_->return_max_points()){
            if(mem_index_0_->insert_point(point, paras_mem_, tag) != 0){
                diskann::cout << "Could not insert point with tag " << tag << std::endl;
                return -3;
            }
            {
                std::unique_lock<std::shared_timed_mutex> lock(_change_lock);
                mem_points_++;
            }
            return 0;
        }else{
            diskann::cout << "Capacity exceeded" << std::endl;
        }
    }else{
        if(mem_index_1_->get_num_points() < mem_index_1_->return_max_points()){
            if(mem_index_1_->insert_point(point, paras_mem_, tag) != 0){
                diskann::cout << "Could not insert point with tag " << tag << std::endl;
                return -3;
            }
            {
                std::unique_lock<std::shared_timed_mutex> lock(_change_lock);
                mem_points_++;
            }
            return 0;
        }else{
            diskann::cout << "Capacity exceeded in mem_index 1" << std::endl;
        }
    }
    return -2;
}

template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::Delete(const WriteOptions&, const TagT& tag){
    std::unique_lock<std::shared_timed_mutex> lock(delete_lock_);
    if((active_delete_set_ == 0) && (active_del_0_.load() == false)){
        diskann::cout << "Active deletion set indicated as _deletion_set_0 but it cannot accept deletions" << std::endl;
    }
    if((active_delete_set_ == 1) && (active_del_1_.load() == false)){
        diskann::cout << "Active deletion set indicated as _deletion_set_1 but it cannot accept deletions" << std::endl;
    }

    if(active_delete_set_ == 0){
        deletion_set_0_.insert(tag);
        mem_index_0_->lazy_delete(tag);
    }
    else{
        deletion_set_1_.insert(tag);
        mem_index_1_->lazy_delete(tag);
    }
}

template<typename T, typename TagT>
void LSMVectorIndex<T, TagT>::Search(const SearchOptions& options, const Slice<T,TagT>& key,
          std::vector<std::pair<TagT,float>>& result){
    std::set<diskann::Neighbor_Tag<TagT>> best;
    const T* query = key.data();
    const uint64_t search_L = options.search_L;
    const uint64_t K = options.K;
    const uint64_t beamwidth = options.beamwidth;
    //search each disk index and get top K tags
    for (int level = 1; level < config::kNumLevels; level++){
        std::shared_lock<std::shared_timed_mutex> lock(disk_locks_[level]);
        assert(switching_disk_prefixes_ == false);
        std::vector<float> disk_result_dists(search_L);
        std::vector<TagT> disk_result_tags(search_L);
        _disk_indexes[level]->cached_beam_search(
            query, search_L, search_L, disk_result_tags.data(), disk_result_dists.data(), beamwidth,
            stats);
        for(unsigned i = 0; i < disk_result_tags.size(); i++){
            diskann::Neighbor_Tag<TagT> n;
            n = diskann::Neighbor_Tag<TagT>(disk_result_tags[i], disk_result_dists[i]);
            best.insert(n);
        }
    }
    //check each memory index - if non empty and not being currently cleared - search and get top K active tags 
    {
        if(clearing_index_0_.load() == false){
            std::shared_lock<std::shared_timed_mutex> lock(clear_lock_0_);
            if(mem_index_0_->get_num_points() > 0){
                std::vector<diskann::Neighbor_Tag<TagT>> best_mem_index_0;
                mem_index_0_->search(query, (uint32_t)search_L, (uint32_t)search_L, best_mem_index_0);
                for(auto iter : best_mem_index_0)
                    best.insert(iter);
            }
        }

        if(clearing_index_1_.load() == false){
            std::shared_lock<std::shared_timed_mutex> lock(clear_lock_1_);
            if(mem_index_1_->get_num_points() > 0){
                std::vector<Neighbor_Tag<TagT>> best_mem_index_1;
                mem_index_1_->search(query, (uint32_t)search_L, (uint32_t)search_L, best_mem_index_1);
                for(auto iter : best_mem_index_1)
                    best.insert(iter);
            }
        }
    }
    std::vector<Neighbor_Tag<TagT>> best_vec;
    for(auto iter : best)
        best_vec.emplace_back(iter);
    { //aggregate results, sort and pick top K candidates
        std::shared_lock<std::shared_timed_mutex> lock(delete_lock_);
        for (auto iter : best_vec) {
            if((deletion_set_0_.find(iter.tag) == deletion_set_0_.end()) && (deletion_set_1_.find(iter.tag) == deletion_set_1_.end())) {
                result.emplace_back(iter.tag, iter.dist);
            }
            if (result.size() == K)
                break;
        }
    }
}

} // namespace lsmidx
