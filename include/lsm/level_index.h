#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include "lsm/options.h"
#include "lsm/slice.h"
#include "Neighbor_Tag.h"
#include "index.h"
#include "aligned_file_reader.h"
#include "lsm/level0_merger.h"

namespace lsmidx{

typedef enum IndexType {
  IN_MEM_DISKANN,
  ON_DISK_DISKANN
} IndexType;

template<typename T, typename TagT = uint32_t>
class LevelIndex{
public:
  virtual void KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, diskann::QueryStats * stats=nullptr) = 0;
  virtual void GetActiveTags(tsl::robin_set<TagT>& active_tags) = 0;

  LevelIndex(IndexType type, std::shared_ptr<diskann::Parameters> params, int merge_thresh, int level, size_t dimension, bool is_single_file_index, diskann::Metric dist_metric):type(type), paras(params), merge_thresh(merge_thresh),level(level),dimension(dimension),is_single_file_index(is_single_file_index),dist_metric(dist_metric){}

  std::shared_ptr<diskann::Parameters> GetParameter(){
    return this->paras;
  };
  std::string GetIndexPrefix(){
    return this->index_prefix;
  }
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
};

template<typename T, typename TagT = uint32_t>
class PQFlashIndexProxy : public LevelIndex<T, TagT>{
public:
  void KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, diskann::QueryStats * stats=nullptr) override;

  void GetActiveTags(tsl::robin_set<TagT>& active_tags) override;

  PQFlashIndexProxy(diskann::Metric dist_metric, std::string working_dir, std::shared_ptr<AlignedFileReader> &fileReader, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_disk, int cur_level, bool is_single_file_index, int num_threads);
  
  void ReloadIndex(const std::string &disk_index_prefix);
private:
  std::shared_ptr<diskann::PQFlashIndex<T, TagT>> index;
};

template<typename T, typename TagT = uint32_t>
class InMemIndexProxy : public LevelIndex<T, TagT>{
public:
  void KNNQuery(const T *query, std::vector<diskann::Neighbor_Tag<TagT>>& res, SearchOptions options, diskann::QueryStats * stats=nullptr) override;
  
  void GetActiveTags(tsl::robin_set<TagT>& active_tags) override;

  InMemIndexProxy(diskann::Metric dist_metric, std::string working_dir, size_t dims, size_t merge_thresh, std::shared_ptr<diskann::Parameters> paras_mem, bool is_single_file_index);

  void LazyDelete(TagT tag);
  int Put(const WriteOptions& options, const VecSlice<T>& key, const TagT& value);
  int Switch();
  std::string SaveIndex(size_t idx);
  void ClearIndex(size_t idx);
  int GetNextSwitchIdx();

  // 返回current指向的mem index的点数量
  int GetCurrentNumPoints();
  
private:
  /**
   * current 表示现在Head Ptr所指向的index
   * 而active_0，active_1则表示对应的index现在是否能使用
  */
  int current = 0;  // reflects value of writable index
  std::vector<std::shared_ptr<diskann::Index<T, TagT>>> indexes;
  std::vector<std::unique_ptr<std::atomic_bool>> active_states;
  std::vector<std::unique_ptr<std::atomic_bool>> index_clearing_states;
  std::vector<std::unique_ptr<std::shared_timed_mutex>> clear_locks;  // lock to prevent an index from being cleared when it
                                        // is being searched  and vice versa
  std::atomic_bool switching_disk_prefixes;  // wait if true, search when false
  
};
} // namespace lsmidx