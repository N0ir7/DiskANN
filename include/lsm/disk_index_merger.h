#pragma once

#include "v2/graph_delta.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "index.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <level_merger.h>
#include "windows_customizations.h"
#include "lsm/index_data_iterator.h"

namespace lsmidx
{

struct DiskIndexParam{
  uint32_t beam_width;
  uint32_t l_index;
  uint32_t range;
  uint32_t maxc;
  float    alpha;
};

template<typename T, typename TagT = uint32_t>
class DiskIndexMerger{
  public:
    DiskIndexFileMeta meta;
    tsl::robin_set<unsigned> delete_local_id_set;
    tsl::robin_set<uint32_t>  free_local_ids;
    diskann::PQFlashIndex<T, TagT> * index;
    diskann::GraphDelta * delta;
    diskann::Metric dist_metric;
    diskann::Distance<T> * dist_cmp;
    DiskIndexParam param;
    std::shared_ptr<AlignedFileReader> reader;
    DiskIndexMerger(DiskIndexFileMeta meta):meta(meta),index(nullptr),delta(nullptr){};

    // 创建一个新的 PQFlashIndex 对象，负责需要merge的磁盘索引的操作
    void InitIndex();
    // 创建一个新的 PQFlashIndex 对象，负责需要merge的磁盘索引的操作,同时全局缓存一些节点
    void InitIndexWithCache();
    // 创建一个新的 GraphDelta 对象，管理图的增量变化
    void InitGraphDelta(uint64_t global_offset);
    // 更新本索引中需要删除的点的local id
    void AddDeleteLocalID(tsl::robin_set<TagT>& deleted_tags);
    // 收集所有被删除的点及其未删除的邻居节点
    tsl::robin_map<uint32_t, std::vector<uint32_t>> PopulateNondeletedHoodsOfDeletedNodes();

    void ProcessDeletes(DiskIndexFileMeta& temp_index_file_meta,
                        tsl::robin_map<uint32_t, std::vector<uint32_t>>& disk_deleted_nhoods,
                        std::vector<uint8_t *>& thread_bufs);
    
    void OffsetIterateToFixedPoint(const T *vec, const uint32_t Lsize,
                                  std::vector<diskann::Neighbor> & expanded_nodes_info,
                                  tsl::robin_map<uint32_t, T *> &coord_map,
                                   diskann::ThreadData<T> *thread_data = nullptr);
    
    void PruneNeighbors(const tsl::robin_map<uint32_t, T *> &coord_map,
                        std::vector<diskann::Neighbor> &pool, 
                        std::vector<uint32_t> &pruned_list);
    void PruneNeighborsWithPQDistance(std::vector<diskann::Neighbor> &pool, 
                            std::vector<uint32_t> &pruned_list,
                            uint8_t *scratch);
    std::vector<uint8_t> DeflateVector(const T *vec);
    /**
     * Getter Functions
    */
    // 获取标签列表
    TagT* tags(){
      return this->index->get_tags();
    }
    // 获取初始ID列表
    std::vector<uint32_t> init_ids(){
      return this->index->get_init_ids();
    }
    // 获取总点数
    uint32_t num_points(){
      return (_u32) this->index->return_nd();
    }
    // 获取线程内存空间
    std::vector<diskann::ThreadData<T>> & thread_data(){
      return this->index->get_thread_data();
    }
    // 获取PQ坐标数据
    uint8_t *pq_data(){
      return this->index->get_pq_config().first;
    }
    // 获取PQ的chunk数
    uint32_t pq_nchunks(){
      return this->index->get_pq_config().second;
    }
    // 获取每个block的点数
    uint32_t nnodes_per_sector(){
      return (_u32) this->index->nnodes_per_sector;
    }
    // 获取每个点的最大长度(by bytes)
    uint32_t max_node_len(){
      return (_u32) this->index->max_node_len;
    }
    uint64_t num_frozen_points(){
      return this->index->get_num_frozen_points();
    }
    uint64_t frozen_loc(){
      return this->index->get_frozen_loc(); 
    }
    uint64_t ndim(){
      return this->index->get_dim(); 
    }
    uint64_t aligned_ndim(){
      return this->index->get_aligned_dim(); 
    }
    uint32_t beam_width(){
      return this->param.beam_width;
    }
    uint32_t range(){
      return this->param.range;
    }
    uint32_t l_index(){
      return this->param.l_index;
    }
    uint32_t maxc(){
      return this->param.maxc;
    }
    float alpha(){
      return this->param.alpha;
    }
    DiskIndexDataIterator<T, TagT> GetIterator();
    bool IsFree(uint32_t local_id);
  private:
    void ConsolidateDeletes(diskann::DiskNode<T> &disk_node, uint8_t * scratch, tsl::robin_map<uint32_t, std::vector<uint32_t>>& disk_deleted_nhoods);
    /**
     * Predicate Functions
    */
    bool IsDeleted(diskann::DiskNode<T> &disk_node);
    /**
     * Helper Functions
    */
    void DumpToDisk(const uint32_t start_id,
                    const char *   buf,
                    const uint32_t n_sectors,
                    std::ofstream & output_writer);
    void OccludeListWithPQDistance(std::vector<diskann::Neighbor> &pool, 
                                  std::vector<diskann::Neighbor> &result,
                                  std::vector<float> &occlude_factor, 
                                  uint8_t *scratch);
    void OccludeList(std::vector<diskann::Neighbor> & pool,
                                            const tsl::robin_map<uint32_t, T *> &coord_map,
                                            std::vector<diskann::Neighbor> &result, 
                                            std::vector<float> &occlude_factor);
};

} // namespace lsmidx