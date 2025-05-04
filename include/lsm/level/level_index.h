#pragma once

#include <vector>
#include <memory>
#include <atomic>
// #include <boost/thread/shared_mutex.hpp>
#include <shared_mutex>
#include "lsm/options.h"
#include "lsm/slice.h"
#include "Neighbor_Tag.h"
#include "index.h"
#include "pq_flash_index.h"
#include "percentile_stats.h"
#include "aligned_file_reader.h"
#include "lsm/tag_deleter.h"


namespace lsmidx{

typedef enum IndexType {
  IN_MEM_DISKANN,
  ON_DISK_DISKANN
} IndexType;

template<typename T, typename TagT = uint32_t>
class LevelIndex{
public:
  virtual void KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, std::vector<lsmidx::TagDeleter<TagT>*> exclude_set_list, diskann::QueryStats * stats=nullptr) = 0;
  virtual void GetActiveTags(tsl::robin_set<TagT>& active_tags) = 0;
  virtual int GetCurrentNumPoints() = 0;
  virtual void ClearIndex() = 0;
  virtual std::string ReportIndexInfo() = 0;
  std::unique_lock<std::shared_mutex> GetWriteLock(){
    return std::unique_lock<std::shared_mutex>(this->mutex);
  }
  std::shared_lock<std::shared_mutex> GetReadLock(){
    return std::shared_lock<std::shared_mutex>(this->mutex);
  }
  LevelIndex(IndexType type, std::shared_ptr<diskann::Parameters> params, int merge_thresh, int level, size_t dimension, bool is_single_file_index, diskann::Metric dist_metric):type(type), paras(params), merge_thresh(merge_thresh),level(level),dimension(dimension),is_single_file_index(is_single_file_index),dist_metric(dist_metric){}

  std::shared_ptr<diskann::Parameters> GetParameter(){
    return this->paras;
  };
  std::string GetIndexPrefix(){
    return this->index_prefix;
  }
  bool IsEmpty(){
    return this->GetCurrentNumPoints() == 0;
  }
  
  void FilterDeletedTags(std::vector<diskann::Neighbor_Tag<TagT>>& vec){
    if(this->delete_tag_set.IsEmpty() || vec.empty()){
      return;
    }
    vec.erase(
        std::remove_if(vec.begin(), vec.end(),
                        [&](const diskann::Neighbor_Tag<TagT>& neighbor) {
                            return this->delete_tag_set.IsDelete(neighbor.tag);
                        }),
        vec.end());
  }
  void FilterDeletedTags(tsl::robin_set<TagT>& set){
    if(this->delete_tag_set.IsEmpty() || set.empty()){
      return;
    }
    this->delete_tag_set.FilterDeletedTags(set);
    return;
  }
  void Union(TagDeleter<TagT>& union_delete_set){
    this->delete_tag_set.Union(union_delete_set);
  }
  void Union(tsl::robin_set<TagT>& union_delete_set){
    this->delete_tag_set.Union(union_delete_set);
  }
  // std::unique_lock<std::shared_mutex> GetWriteDeleteLock(){
  //   return this->delete_tag_set.GetWriteLock();
  // }
  // std::shared_lock<std::shared_mutex> GetReadDeleteLock(){
  //   return this->delete_tag_set.GetReadLock();
  // }
  lsmidx::TagDeleter<TagT> delete_tag_set;
protected:
  IndexType type;
  std::shared_ptr<diskann::Parameters> paras; // 索引构建参数
  /**
   * 一些基本信息
  */
  int merge_thresh; // 向下层合并的阈值
  int level; // index所在的层数
  size_t dimension; // 向量维度
  std::string index_prefix; // index前缀
  bool is_single_file_index; // 是否是单文件索引
  diskann::Metric dist_metric;
  std::shared_mutex mutex;
};
template<typename T, typename TagT = uint32_t>
class MultiIndex : public LevelIndex<T, TagT>{
public:
  MultiIndex(IndexType type, std::shared_ptr<diskann::Parameters> params, int merge_thresh, int level, size_t dimension, bool is_single_file_index, diskann::Metric dist_metric):LevelIndex<T,TagT>( type, params, merge_thresh, level, dimension, is_single_file_index, dist_metric){};
  virtual std::unique_lock<std::shared_mutex> GetSubWriteLock(size_t idx) = 0;
  virtual std::shared_lock<std::shared_mutex> GetSubReadLock(size_t idx) = 0;
  virtual void ClearSubIndex(size_t idx) = 0;
  virtual void RefreshDeleteTagSet() = 0;
  std::vector<std::unique_ptr<std::atomic_bool>> active_states;
};
template<typename T, typename TagT = uint32_t>
class PQFlashIndexProxy : public LevelIndex<T, TagT>{
public:
  void KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, std::vector<lsmidx::TagDeleter<TagT>*> exclude_set_list, diskann::QueryStats * stats=nullptr) override;
  void KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, std::vector<lsmidx::TagDeleter<TagT>*> exclude_set_list, diskann::QueryStats * stats, diskann::ThreadData<T>* thread_data);

  void GetActiveTags(tsl::robin_set<TagT>& active_tags) override;
  int GetCurrentNumPoints() override;

  PQFlashIndexProxy(diskann::Metric dist_metric, std::string working_dir, int idx, std::shared_ptr<AlignedFileReader> &fileReader, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_disk, int cur_level, bool is_single_file_index, int num_threads);

  // Need to acquire write lock in advance
  void ReloadIndex(const std::string &disk_index_prefix);

  // Need to acquire write lock in advance
  void LoadIndex(const std::string &disk_index_prefix, int num_threads=16);

  // Need to acquire write lock in advance
  void ClearIndex() override;
  void GetMedoids(std::vector<TagT>& medoid_vec);
  std::shared_ptr<diskann::PQFlashIndex<T, TagT>> GetIndex();
  void ReportQueryInfo(int query_num=0);
  std::string ReportIndexInfo() override;
  diskann::ThreadData<T> PopThreadData();
  void PushThreadData(diskann::ThreadData<T> data);
  void PrecomputeChunkDistance(diskann::ThreadData<T>& data, const T *query);
  float GetMinClusterDistance(diskann::ThreadData<T>& data);
private:
  std::shared_ptr<diskann::PQFlashIndex<T, TagT>> index;
  std::shared_ptr<AlignedFileReader> reader;
  diskann::QueryStats stat;
};

template<typename T, typename TagT = uint32_t>
class MultiPQFlashIndexProxy : public MultiIndex<T, TagT>{
public:
  void KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, std::vector<lsmidx::TagDeleter<TagT>*> exclude_set_list, diskann::QueryStats * stats=nullptr) override;

  void GetActiveTags(tsl::robin_set<TagT>& active_tags) override;
  int GetCurrentNumPoints() override;

  MultiPQFlashIndexProxy(diskann::Metric dist_metric, std::string working_dir, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_disk, int cur_level, bool is_single_file_index, int num_threads);

  void ReloadIndex(const std::string &disk_index_prefix, size_t idx);
  void ClearSubIndex(size_t idx) override;
  void ClearIndex() override;
  int  GetFreeIndexSlot();
  std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> GetNonFreeIndexList();
  size_t GetNonFreeIndexesSize();
  std::unique_lock<std::shared_mutex> GetSubWriteLock(size_t idx) override;
  std::shared_lock<std::shared_mutex> GetSubReadLock(size_t idx) override;
  std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> GetIndexProxy(size_t idx);
  std::shared_ptr<diskann::PQFlashIndex<T, TagT>> GetIndex(size_t idx);
  void AddFreeSlot(int idx);
  void RefreshDeleteTagSet() override;
  void SetReader();
  void GetMedoids(std::vector<TagT>& medoid_vec);
  void BatchLazyDeleteIndex(std::vector<int>& remove_idx_vec);
  std::string ReportIndexInfo() override;
private:
  std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> indexes;
  std::shared_mutex free_slots_lock;
  std::queue<int> free_slots;
  int num_threads_for_search;
  std::vector<std::shared_ptr<AlignedFileReader>> readers;
  std::vector<unsigned> index_timestamp;
  unsigned timestamp = 0;
  // std::shared_ptr<AlignedFileReader> reader;
  
};

template<typename T, typename TagT = uint32_t>
class InMemIndexProxy : public LevelIndex<T, TagT>{
public:
  void KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, std::vector<lsmidx::TagDeleter<TagT>*> exclude_set_list, diskann::QueryStats * stats=nullptr) override;
  
  void GetActiveTags(tsl::robin_set<TagT>& active_tags) override;

  InMemIndexProxy(diskann::Metric dist_metric, std::string working_dir, int idx, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_mem, bool is_single_file_index);

  int LazyDelete(TagT tag);
  int Put(const WriteOptions& options, const VecSlice<T>& key, const TagT& value, diskann::InsertStats * stats = nullptr);

  std::string SaveIndex();
  void ClearIndex() override;

  // 返回current指向的mem index的点数量
  int GetCurrentNumPoints() override;
  std::shared_ptr<diskann::Index<T, TagT>> GetIndex();
  void Consolidate();
  std::string ReportIndexInfo() override;
private:
  std::shared_ptr<diskann::Index<T, TagT>> index;
};
template<typename T, typename TagT = uint32_t>
class MultiInMemIndexProxy : public MultiIndex<T, TagT>{
public:
  void KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, std::vector<lsmidx::TagDeleter<TagT>*> exclude_set_list, diskann::QueryStats * stats=nullptr) override;
  
  void GetActiveTags(tsl::robin_set<TagT>& active_tags) override;

  MultiInMemIndexProxy(diskann::Metric dist_metric, std::string working_dir, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_mem, bool is_single_file_index);

  int LazyDelete(TagT tag);
  int Put(const WriteOptions& options, const VecSlice<T>& key, const TagT& value, diskann::InsertStats * stats);
  int Switch();
  std::string SaveIndex(size_t idx);
  void ClearIndex() override;
  void ClearSubIndex(size_t idx) override;
  int GetNextSwitchIdx();

  // 返回current指向的mem index的点数量
  int GetNumPointsOfCur();
  int GetCurrentNumPoints() override;
  std::unique_lock<std::shared_mutex> GetSubWriteLock(size_t idx) override;
  std::shared_lock<std::shared_mutex> GetSubReadLock(size_t idx) override;
  std::shared_lock<std::shared_mutex> GetCurrentMutexReadLock();
  std::unique_lock<std::shared_mutex> GetCurrentMutexWriteLock();
  std::shared_ptr<diskann::Index<T, TagT>> GetIndex(size_t idx);
  std::shared_ptr<lsmidx::InMemIndexProxy<T, TagT>> GetIndexProxy(size_t idx);
  void RefreshDeleteTagSet() override;
  std::string ReportIndexInfo() override;
  void RecalculateInsertMemIndexEntryPoint();
private:
  /**
   * current 表示现在Head Ptr所指向的index
   * 而active_0，active_1则表示对应的index现在是否能使用
  */
  std::shared_mutex current_mutex;
  int current = 0;  // reflects value of writable index
  std::vector<std::shared_ptr<lsmidx::InMemIndexProxy<T, TagT>>> indexes;
  
  std::vector<std::unique_ptr<std::atomic_bool>> index_clearing_states;
};
} // namespace lsmidx