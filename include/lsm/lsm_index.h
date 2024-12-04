#pragma once

#include <set>
#include <string>
#include <memory>
#include "lsm/options.h"
#include "lsm/slice.h"
#include "lsm/level_merger.h"
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "index.h"
#include "lsm/level_index.h"
#include "lsm/tag_deleter.h"
namespace lsmidx{

template<typename T, typename TagT = uint32_t>
class LSMVectorIndex{
  public:
    LSMVectorIndex(const BuildOptions& options, const std::string& working_dir, const std::string& index_name, diskann::Distance<T>* dist);

    LSMVectorIndex(const LSMVectorIndex&) = delete;
    LSMVectorIndex& operator=(const LSMVectorIndex&) = delete;

    ~LSMVectorIndex();

    // Implementations of the DB interface
    int Put(const WriteOptions& options, const VecSlice<T>& key,
              const TagSlice<TagT>& value);
    void Delete(const WriteOptions& options, const TagSlice<TagT>& value);
    void Search(const SearchOptions& options, const VecSlice<T>& key,
              std::vector<diskann::Neighbor_Tag<TagT>>& result);
    
    // Background merge
    void MergeMemIndex(std::string mem_index_path);
    void TriggerMergeMemIndex();
    std::unique_ptr<Level0Merger<T, TagT>> ConstructLevel0Merger();
    // other help functions
    void SetSeachParams(diskann::Parameters& parameters);
    void SetSystemParams(const BuildOptions& options);
    void SetDistanceFunction(diskann::Distance<T>* dist, diskann::Metric dist_metric);
    void SetReader();
  private:
  /**
   * some tools classes
   */
  std::shared_ptr<AlignedFileReader> reader = nullptr; // 各层索引共用一个reader
  // ThreadPool* search_tpool;
  diskann::Metric dist_metric;
  diskann::Distance<T>* dist_comp;
  /**
   * some data structures as helpers
  */
  // std::unordered_map<unsigned, TagT> curr_location_to_tag;
  std::vector<const std::vector<TagT>*> deleted_tags_vector;
  // for global delete
  TagDeleter<TagT> global_in_mem_delete_tag_set;
  // tsl::robin_set<TagT> deletion_set_0;
  // tsl::robin_set<TagT> deletion_set_1;
  // int active_delete_set = 0;                // reflects active _deletion_set
  // std::atomic_bool _active_del_0;           // true except when being saved
  // std::atomic_bool _active_del_1;           // true except when being saved

  /**
   * critical data structures
   */
  std::shared_ptr<InMemIndexProxy<T, TagT>>    mem_index = nullptr;
  std::vector<std::shared_ptr<LevelIndex<T, TagT>>> disk_indexes;
  // lsmidx::StreamingMerger<T, TagT> *  merger_ = nullptr;

  /**
   * locks and signal variables
   */
  std::shared_timed_mutex delete_lock;              // lock to access _deletion_set
  std::shared_timed_mutex index_lock;               // mutex to switch between mem indices
  std::vector<std::shared_timed_mutex> disk_locks;  // mutex to switch between disk indices
  std::atomic_bool switching_disk = false;          // wait if true, search when false
  std::atomic_bool check_switch_index = false;      // true when switch_index acquires _index_lock in writer mode,
                                                     // insert threads wait till it turns back to false

  /**
   * parameters and options
   */
  // diskann::Parameters paras_mem;
  // diskann::Parameters paras_disk;
  bool is_single_file_index;
  // size_t   merge_th = 0;
  // size_t   mem_points = 0;  // reflects number of points in active mem index
  // size_t   index_points = 0;
  size_t   dimension;
  _u32     num_nodes_to_cache;
  _u32     num_search_threads;
  uint64_t beamwidth;

  /**
   * some constants
  */
  // std::string mem_index_prefix;
  // std::string disk_index_prefix_in;
  // std::string disk_index_prefix_out;
  // std::string deleted_tags_file;
  // std::string TMP_FOLDER;
  std::string working_dir;
  std::string index_name;
};
} // namespace lsmidx