#pragma once

#include <set>
#include <string>
#include <memory>
#include "lsm/options.h"
#include "lsm/slice.h"
#include "lsm/merger/level_merger.h"
#include "lsm/merger/leveln_merger.h"
#include "lsm/merger/level0_merger.h"
#include "lsm/merger/mem_to_level1_merger.h"
#include "lsm/merger/mem_flusher.h"
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "index.h"
#include "lsm/level/level_index.h"
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
              const TagSlice<TagT>& value, diskann::InsertStats * stats=nullptr);
    void Delete(const WriteOptions& options, const TagSlice<TagT>& value);
    void Search(const SearchOptions& options, const VecSlice<T>& key,
              TagT* tags, float * distances, diskann::QueryStats * stats);
    
    // Background merge
    
    void TriggerMergeMemIndex();
    void TriggerMergeDiskIndex(int level, diskann::MergeStats* stats = nullptr);
    
    // other help functions
    void SetSeachParams(const diskann::Parameters& parameters);
    void SetSystemParams(const BuildOptions& options);
    void SetDistanceFunction(diskann::Distance<T>* dist, diskann::Metric dist_metric);
    void SetReader();
    void GetActiveTags(tsl::robin_set<TagT>& active_tags);
    void GetMedoid(std::vector<TagT>& medoid_vec);
    void ReportQueryInfo(int query_num);
    void ReportIndexInfo();
    void RedistributeDiskIndex();
    void RecalculateInsertMemIndexEntryPoint();
  private:
  // Background merge
  int MergeMemIndex(std::shared_ptr<lsmidx::InMemIndexProxy<T, TagT>> from_mem_index, std::shared_ptr<lsmidx::MultiPQFlashIndexProxy<T, TagT>> to_disk_index);
  void MergeDiskIndex(std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>>& from_indexes, std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> to_index, diskann::MergeStats* stats = nullptr);
  std::shared_lock<std::shared_mutex> GetMemLevelReadLock();
  std::unique_lock<std::shared_mutex> GetMemLevelWriteLock();
  std::shared_lock<std::shared_mutex> GetDiskLevelReadLock(int level);
  std::unique_lock<std::shared_mutex> GetDiskLevelWriteLock(int level);
  std::unique_ptr<MemFlusher<T, TagT>> ConstructMemFlusher();
  std::unique_ptr<Level0Merger<T, TagT>> ConstructLevel0Merger(std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> from_indexes, std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> to_index);
  std::unique_ptr<Mem2Level1Merger<T, TagT>> ConstructMem2Level1Merger();
  std::unique_ptr<LevelNMerger<T, TagT>> ConstructLevelNMerger(int to_level);
  /**
   * some tools classes
   */
  std::vector<std::shared_ptr<AlignedFileReader>> readers; 
  // ThreadPool* search_tpool;
  diskann::Metric dist_metric;
  diskann::Distance<T>* dist_comp;

  /**
   * critical data structures
   */
  std::shared_ptr<MultiInMemIndexProxy<T, TagT>>    mem_index = nullptr;
  std::vector<std::shared_ptr<LevelIndex<T, TagT>>> disk_indexes;

  /**
   * parameters and options
   */
  std::shared_ptr<diskann::Parameters> paras_mem;
  std::shared_ptr<diskann::Parameters> paras_disk;
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