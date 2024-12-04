#include "lsm/disk_index_merger.h"
#include "timer.h"
#include <omp.h>

namespace lsmidx
{
template<typename T, typename TagT>
void DiskIndexMerger<T,TagT>::InitIndexWithCache(){
  std::string index_prefix_path = this->meta.index_prefix_path;
  this->reader = std::make_shared<LinuxAlignedFileReader>();
  // 创建一个新的 PQFlashIndex 对象，负责磁盘索引的操作
  diskann::cout << "Created PQFlashIndex inside disk_index_merger " << std::endl;
  this->index = std::make_shared<diskann::PQFlashIndex<T, TagT>>(this->dist_metric, reader, this->meta.is_single_file, true);
  
  // 加载磁盘索引文件
  diskann::cout << "Loading PQFlashIndex from file: " << index_prefix_path
                << " into object: " << std::hex << (_u64) &
      (this->index) << std::dec << std::endl;
  this->index->load(index_prefix_path.c_str(),NUM_INDEX_LOAD_THREADS);

  // 计算要缓存的节点数量，最多缓存 PQ_FLASH_INDEX_MAX_NODES_TO_CACHE 个节点
  uint32_t node_cache_count = 1 + (uint32_t) round(this->index->return_nd() * 0.01);
  node_cache_count = node_cache_count > PQ_FLASH_INDEX_MAX_NODES_TO_CACHE
                          ? PQ_FLASH_INDEX_MAX_NODES_TO_CACHE
                          : node_cache_count;
  std::vector<uint32_t> cache_node_list;

  // 生成要缓存的 BFS 层级节点列表，并加载这些缓存节点
  // 节点的邻居信息存储到nhood_cache中，并将节点的坐标存储到coord_cache
  this->index->cache_bfs_levels(node_cache_count,
                                      cache_node_list);
  this->index->load_cache_list(cache_node_list);

}
template<typename T, typename TagT>
void DiskIndexMerger<T,TagT>::InitIndex(){
  std::string index_prefix_path = this->meta.index_prefix_path;
  this->reader = std::make_shared<LinuxAlignedFileReader>();
  // 创建一个新的 PQFlashIndex 对象，负责磁盘索引的操作
  diskann::cout << "Created PQFlashIndex inside disk_index_merger " << std::endl;
  this->index = std::make_shared<diskann::PQFlashIndex<T, TagT>>(this->dist_metric, reader, this->meta.is_single_file, true);
  
  // 加载磁盘索引文件
  diskann::cout << "Loading PQFlashIndex from file: " << index_prefix_path
                << " into object: " << std::hex << (_u64) &
      (this->index) << std::dec << std::endl;
  this->index->load(index_prefix_path.c_str(),NUM_INDEX_LOAD_THREADS);
}

template<typename T, typename TagT>
void DiskIndexMerger<T,TagT>::InitGraphDelta(uint64_t global_offset){
  this->delta = new diskann::GraphDelta(global_offset, this->num_points());
}
template<typename T, typename TagT>
void DiskIndexMerger<T,TagT>::AddDeleteLocalID(tsl::robin_set<TagT>& deleted_tags){
  TagT* disk_tags = this->tags();
  for (uint32_t i = 0; i < this->num_points(); i++) {
    TagT i_tag = disk_tags[i];
    if (deleted_tags.find(i_tag) != deleted_tags.end()) {
      this->delete_local_id_set.insert(i);
    }
  }
  diskann::cout << "Found " << this->delete_local_id_set.size()
                << " tags to delete from SSD-DiskANN\n";
}
template<typename T, typename TagT>
tsl::robin_map<uint32_t, std::vector<uint32_t>> DiskIndexMerger<T,TagT>::PopulateNondeletedHoodsOfDeletedNodes(){
  // buf for scratch
  char *index_read_buf = nullptr;
  char *delete_backing_buf = nullptr;
  std::vector<diskann::DiskNode<T>> deleted_nodes;
  tsl::robin_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;

  /**
   * 初始化两片内存缓冲区
   * 1.delete_backing_buf: 用来存储被删除点的未被删除邻居
   * 2.buf: 用来批读取索引内容
  */
  uint64_t delete_backing_buf_size = (uint64_t) this->delete_local_id_set.size() *
                              ROUND_UP(this->max_node_len(), 32);
  delete_backing_buf_size = ROUND_UP(delete_backing_buf_size, 256);
  uint64_t index_read_buf_size = SECTORS_PER_MERGE * SECTOR_LEN;

  diskann::alloc_aligned((void **) &delete_backing_buf, delete_backing_buf_size, 256);
  diskann::alloc_aligned((void **) &index_read_buf, index_read_buf_size, SECTOR_LEN);
  memset(delete_backing_buf, 0, delete_backing_buf_size);
  memset(index_read_buf, 0, index_read_buf_size);

  diskann::cout << "ALLOC: " << (delete_backing_buf_size >> 10)
                << "KiB aligned buffer for deletes.\n";
  
  // scan deleted nodes and get
  this->index->scan_deleted_nodes(this->delete_local_id_set, deleted_nodes,
                                        index_read_buf, delete_backing_buf,
                                        SECTORS_PER_MERGE);

  // insert into deleted_nhoods
  deleted_nhoods.reserve(deleted_nodes.size());
  for (auto &nhood : deleted_nodes) {
    // WARNING :: ASSUMING DISK GRAPH DEGREE NEVER GOES OVER 512
    assert(nhood.nnbrs < 512);
    std::vector<uint32_t> non_deleted_nbrs;
    for (uint32_t i = 0; i < nhood.nnbrs; i++) {
      uint32_t id = nhood.nbrs[i];
      auto     iter = this->delete_local_id_set.find(id);
      if (iter == this->delete_local_id_set.end()) {
        non_deleted_nbrs.push_back(id);
      }
    }
    deleted_nhoods.insert(
        std::make_pair(nhood.id, non_deleted_nbrs));
  }

  // free buf and delete_backing_buf
  diskann::aligned_free((void *) index_read_buf);
  diskann::aligned_free((void *) delete_backing_buf);

  assert(deleted_nodes.size() == this->delete_local_id_set.size());
  assert(deleted_nhoods.size() == this->delete_local_id_set.size());

  return deleted_nhoods;
}
template<typename T, typename TagT>
void DiskIndexMerger<T,TagT>::ProcessDeletes(DiskIndexFileMeta& temp_index_file_meta, 
                                            tsl::robin_map<uint32_t, std::vector<uint32_t>>& disk_deleted_nhoods,
                                            std::vector<uint8_t *>& thread_bufs){
  // buf to hold data being read
  char *index_read_buf = nullptr;
  uint64_t index_read_buf_size = SECTORS_PER_MERGE * SECTOR_LEN;
  diskann::alloc_aligned((void **) &index_read_buf, index_read_buf_size, SECTOR_LEN);

  // open output file for writing
  
  diskann::cout << "Writing delete consolidated graph to "
                << temp_index_file_meta.data_path << std::endl;

  DiskIndexDataIterator<T,TagT> index_data_iter  = this->GetIterator();
  index_data_iter.Init(false /*read write*/, temp_index_file_meta.data_path);

  diskann::Timer delete_timer;
  // batch consolidate deletes

  while(index_data_iter.HasNextBatch()){
    std::vector<diskann::DiskNode<T>>* disk_nodes = nullptr;
    std::tie(disk_nodes, std::ignore, std::ignore) = index_data_iter.NextBatch();

    #pragma omp parallel for schedule(dynamic, 128) num_threads(MAX_N_THREADS)
    for(size_t i = 0; i < disk_nodes->size(); i++){
      diskann::DiskNode<T>& disk_node = (*disk_nodes)[i];
      int      omp_thread_no = omp_get_thread_num();
      uint8_t *pq_coord_scratch = thread_bufs[omp_thread_no];
      this->ConsolidateDeletes(disk_node, pq_coord_scratch, disk_deleted_nhoods);
    }

    for (auto &disk_node : *disk_nodes) {
      if (this->IsDeleted(disk_node)) {
        this->free_local_ids.insert(disk_node.id);
      }
    }
    index_data_iter.NotifyNodeFlushBack();
  }
  
  double e2e_time = ((double) delete_timer.elapsed()) / (1000000.0);
  diskann::cout << "Processed Deletes in " << e2e_time << " s." << std::endl;

  /**
   * write header
  */
  diskann::cout << "Writing header.\n";
  WriteDataFileHeaderAfterDeletePhase(temp_index_file_meta.data_path, disk_deleted_nhoods);

}
template<typename T, typename TagT>
void DiskIndexMerger<T, TagT>::WriteDataFileHeaderAfterDeletePhase(std::string data_path,tsl::robin_map<uint32_t, std::vector<uint32_t>>& disk_deleted_nhoods){
  /**
   * HEADER -->
   * [_u32 #metadata items]
   * [_u32 1]
   * [_u64 nnodes]
   * [_u64 ndims]
   * [_u64 medoid ID]
   * [_u64 max_node_len]
   * [_u64 nnodes_per_sector]
   * [_u64 #frozen points in vamana index]
   * [_u64 frozen point location]
   * [_u64 file size]
   */

  uint64_t file_size =
      SECTOR_LEN + (ROUND_UP(ROUND_UP(this->num_points(), this->nnodes_per_sector()) /
                                  this->nnodes_per_sector(),
                              SECTORS_PER_MERGE)) *
                        (uint64_t) SECTOR_LEN;
  std::vector<uint64_t> output_metadata;
  output_metadata.push_back(this->num_points());
  output_metadata.push_back((uint64_t) this->ndim());
  // determine medoid
  uint64_t medoid = this->init_ids()[0];
  // TODO (correct?, misc) :: better way of selecting new medoid
  while (this->delete_local_id_set.find((_u32) medoid) !=
          this->delete_local_id_set.end()) {
    diskann::cout << "Medoid deleted. Choosing another start node.\n";
    auto iter = disk_deleted_nhoods.find((_u32) medoid);
    assert(iter != disk_deleted_nhoods.end());
    medoid = iter->second[0];
  }
  output_metadata.push_back((uint64_t) medoid);
  uint64_t max_node_len = (this->ndim() * sizeof(T)) + sizeof(uint32_t) +
                          (this->range() * sizeof(uint32_t));
  uint64_t nnodes_per_sector = SECTOR_LEN / max_node_len;
  output_metadata.push_back(max_node_len);
  output_metadata.push_back(nnodes_per_sector);
  output_metadata.push_back(this->num_frozen_points());
  output_metadata.push_back(this->frozen_loc());
  output_metadata.push_back(file_size);

  // write header
  diskann::save_bin<_u64>(data_path, output_metadata.data(),
                          output_metadata.size(), 1, 0);
  
}
template<typename T, typename TagT>
void DiskIndexMerger<T, TagT>::ConsolidateDeletes(diskann::DiskNode<T> &disk_node, uint8_t * scratch, tsl::robin_map<uint32_t, std::vector<uint32_t>>& disk_deleted_nhoods){

  // 检查节点是否已经删除了，如果已经删除了，则将邻居数设为0
  if (this->IsDeleted(disk_node)) {
    disk_node.nnbrs = 0;
    *(disk_node.nbrs - 1) = 0;
    return;
  }

  const uint32_t id = disk_node.id;

  assert(disk_node.nnbrs < 512); // 这个在之前其实也检查过

  // 将该节点的邻居（从 disk_node.nbrs 开始的内存区域）存入 id_nhood 向量
  std::vector<uint32_t> id_nhood(disk_node.nbrs,
                                  disk_node.nbrs + disk_node.nnbrs);

  tsl::robin_set<uint32_t> new_edges;

  bool change = false;
  for (auto &nbr : id_nhood) {
    auto iter = disk_deleted_nhoods.find(nbr);
    if (iter != disk_deleted_nhoods.end()) {
      change = true;
      new_edges.insert(iter->second.begin(), iter->second.end());
    } else {
      new_edges.insert(nbr);
    }
  }
  // no refs to deleted nodes --> move to next node
  if (!change) {
    return;
  }

  // refs to deleted nodes
  id_nhood.clear();
  id_nhood.reserve(new_edges.size());
  for (auto &nbr : new_edges) { // TODO: 是否多余?
    // 2nd order deleted edge
    auto iter = this->delete_local_id_set.find(nbr);
    if (iter != this->delete_local_id_set.end()) {
      continue;
    } else {
      id_nhood.push_back(nbr);
    }
  }

  // TODO (corner case) :: id_nhood might be empty in adversarial cases
  if (id_nhood.empty()) {
    diskann::cout << "Adversarial case -- all neighbors of node's neighbors "
                      "deleted -- ID : "
                  << id << "; exiting\n";
    exit(-1);
  }

  // compute PQ dists and shrink
  std::vector<float> id_nhood_dists(id_nhood.size(), 0.0f);
  assert(scratch != nullptr);
  this->index->compute_pq_dists(id, id_nhood.data(),
                                      id_nhood_dists.data(),
                                      (_u32) id_nhood.size(), scratch);

  // prune neighbor list using PQ distances
  std::vector<diskann::Neighbor> cand_nbrs(id_nhood.size());
  for (uint32_t i = 0; i < id_nhood.size(); i++) {
    cand_nbrs[i].id = id_nhood[i];
    cand_nbrs[i].distance = id_nhood_dists[i];
  }
  // sort and keep only maxc neighbors
  std::sort(cand_nbrs.begin(), cand_nbrs.end());
  if (cand_nbrs.size() > this->maxc()) {
    cand_nbrs.resize(this->maxc());
  }
  std::vector<diskann::Neighbor> pruned_nbrs;
  std::vector<float>    occlude_factor(cand_nbrs.size(), 0.0f);
  pruned_nbrs.reserve(this->range());
  this->OccludeListWithPQDistance(cand_nbrs, pruned_nbrs, occlude_factor, scratch);

  // copy back final nbrs
  disk_node.nnbrs = (_u32) pruned_nbrs.size();
  *(disk_node.nbrs - 1) = disk_node.nnbrs;
  for (uint32_t i = 0; i < (_u32) pruned_nbrs.size(); i++) {
    disk_node.nbrs[i] = pruned_nbrs[i].id;
  }
}
template<typename T, typename TagT>
void DiskIndexMerger<T, TagT>::ProcessInserts(std::vector<diskann::DiskNode<T>>& insert_nodes, TagT* insert_nodes_tag_list){

    DiskIndexDataIterator<T, TagT> index_data_iter = this->GetIterator();
    index_data_iter.Init(false/* read_write*/);

    std::vector<diskann::ThreadData<T>>& disk_thread_data = this->thread_data();
    #pragma omp parallel for schedule(dynamic, 128) num_threads(MAX_N_THREADS)
    for(size_t i = 0; i < insert_nodes.size(); i++){
      diskann::DiskNode<T>& from_node = insert_nodes[i];
      const T* vec = from_node.coords;
      const uint32_t offset_id = from_node.id;
      TagT from_node_tag = insert_nodes_tag_list[i];
      // diskann::Timer timer;
      // float insert_time, delta_time;
      std::vector<diskann::Neighbor> pool;
      tsl::robin_map<uint32_t, T *> coord_map;
      uint32_t       omp_thread_no = omp_get_thread_num();
      diskann::ThreadData<T> &thread_data = disk_thread_data[omp_thread_no];
      this->OffsetIterateToFixedPoint(vec, this->l_index(), pool,
                                        coord_map, &thread_data);
      // insert_time = (float) timer.elapsed();

      // prune neighbors using alpha
      std::vector<uint32_t> new_nhood;
      this->PruneNeighbors(coord_map, pool, new_nhood);

      // add backward edge to disk index delta
      this->delta->inter_insert(offset_id, new_nhood.data(),
                                    (_u32) new_nhood.size());
      diskann::DiskNode<T>* to_node = nullptr;
      uint8_t* to_node_pq_coord = nullptr;
      TagT* to_node_tag = nullptr;

      // find location to insert vector and its new nhoods(must be exclusive)
      #pragma omp critical
      while(index_data_iter.HasNext()){
        std::tie(to_node, to_node_pq_coord, to_node_tag) = index_data_iter.Next();
        if(!this->IsFree(to_node->id)){ // If not free, skip
          continue;
        }
        break;
      }

      if(to_node && to_node_pq_coord && to_node_tag){ // If find the location
        // copy vector
        memcpy(to_node->coords, from_node.coords, this->ndim() * sizeof(T));
        // number of nhoods
        to_node->nnbrs =  new_nhood.size();
        // copy new nhoods
        std::copy(new_nhood.begin(), new_nhood.end(), to_node->nbrs);
        /* 
         * insert pq data
         */
        // compute PQ coords
        std::vector<uint8_t> new_pq_coords = this->DeflateVector(to_node->coords);

        // directly copy into PQFlashIndex PQ data
        memcpy(to_node_pq_coord, new_pq_coords.data(),
                this->pq_nchunks() * sizeof(uint8_t));
        /**
         * insert tag data
        */
        *to_node_tag = from_node_tag;

        // notify the disk index to flush back the insertion
        index_data_iter.NotifyFlushBack();
      }
      // delta_time = (float) timer.elapsed();

      // uint32_t thread_no = omp_get_thread_num();
      // this->insert_times[thread_no] += insert_time;
      // this->delta_times[thread_no] += delta_time;
    }
}
template<typename T, typename TagT>
void DiskIndexMerger<T, TagT>::ProcessPatch(DiskIndexFileMeta& final_index_file_meta, std::vector<uint8_t *>& thread_bufs){

  DiskIndexDataIterator<T, TagT> index_data_iter = this->GetIterator();
  index_data_iter.Init(false/* read_write*/,final_index_file_meta.data_path);
  
  while(index_data_iter.HasNextBatch()){
    std::vector<diskann::DiskNode<T>>* disk_nodes = nullptr;
    std::tie(disk_nodes, std::ignore, std::ignore) = index_data_iter.NextBatch();

    #pragma omp parallel for schedule(dynamic, 128) num_threads(MAX_N_THREADS)
    for(size_t i = 0; i < disk_nodes->size(); i++){

      diskann::DiskNode<T>& disk_node = (*disk_nodes)[i];
      int      omp_thread_no = omp_get_thread_num();
      uint8_t *thread_scratch = thread_bufs[omp_thread_no];

      // get backward edge set
      std::vector<uint32_t> delta_edges = this->delta->get_nhood(disk_node.id);
      if(delta_edges.empty()){
        continue;
      }
      // add backward edge to old nhood
      std::vector<uint32_t> nhood;
      uint32_t nnbrs = disk_node.nnbrs;
      nhood.insert(nhood.end(), disk_node.nbrs, disk_node.nbrs + nnbrs);
      nhood.insert(nhood.end(), delta_edges.begin(), delta_edges.end());
      if (nhood.size() > this->range()) {
        std::vector<float>    dists(nhood.size(), 0.0f);
        std::vector<diskann::Neighbor> pool(nhood.size());
        
        this->index->compute_pq_dists(disk_node.id, nhood.data(), dists.data(),
                                            (_u32) nhood.size(),
                                            thread_scratch);
        for (uint32_t k = 0; k < nhood.size(); k++) {
          pool[k].id = nhood[k];
          pool[k].distance = dists[k];
        }
        nhood.clear();
        this->PruneNeighborsWithPQDistance(pool, nhood, thread_scratch);
      }
      // copy edges from nhood to disk node
      disk_node.nnbrs = (_u32) nhood.size();
      *(disk_node.nbrs - 1) = (_u32) nhood.size();  // write to buf
      memcpy(disk_node.nbrs, nhood.data(),
              disk_node.nnbrs * sizeof(uint32_t));
      // notify the disk index to flush back the insertion
      index_data_iter.NotifyNodeFlushBack();
    }
  }
}
template<typename T, typename TagT>
bool DiskIndexMerger<T, TagT>::IsDeleted(diskann::DiskNode<T> &disk_node){
  // short circuit when disk_node is a `hole` on disk
  if (this->tags()[disk_node.id] == std::numeric_limits<uint32_t>::max()) {
    if (disk_node.nnbrs != 0) {
      throw diskann::ANNException(std::string("Found node with id: ") +
                                      std::to_string(disk_node.id) +
                                      " that has non-zero degree.",
                                  -1, __FUNCSIG__, __FILE__, __LINE__);
      diskann::cerr << "Node with id " << disk_node.id
                    << " is a hole but has non-zero degree "
                    << disk_node.nnbrs << std::endl;
    } else {
      return true;
    }
  }
  return (this->delete_local_id_set.find(disk_node.id) !=
          this->delete_local_id_set.end());
}
template<typename T, typename TagT>
bool DiskIndexMerger<T, TagT>::IsFree(uint32_t local_id){
  return this->free_local_ids.count(local_id)!=0;

}
template<typename T, typename TagT>
void DiskIndexMerger<T, TagT>::DumpToDisk(const uint32_t start_id,
                                            const char *   buf,
                                            const uint32_t n_sectors,
                                            std::ofstream & output_writer) {
  assert(start_id % this->nnodes_per_sector() == 0);
  uint32_t start_sector = (start_id / this->nnodes_per_sector()) + 1;
  uint64_t start_off = start_sector * (uint64_t) SECTOR_LEN;

  // seek fp
  output_writer.seekp(start_off, std::ios::beg);

  // dump
  output_writer.write(buf, (uint64_t) n_sectors * (uint64_t) SECTOR_LEN);

  uint64_t nb_written =
      (uint64_t) output_writer.tellp() - (uint64_t) start_off;
  if (nb_written != (uint64_t) n_sectors * (uint64_t) SECTOR_LEN) {
    std::stringstream sstream;
    sstream << "ERROR!!! Wrote " << nb_written << " bytes to disk instead of "
            << ((uint64_t) n_sectors) * SECTOR_LEN;
    diskann::cerr << sstream.str() << std::endl;
    throw diskann::ANNException(sstream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  }
}
template<typename T, typename TagT>
void DiskIndexMerger<T, TagT>::OccludeListWithPQDistance(std::vector<diskann::Neighbor> &pool, 
                                  std::vector<diskann::Neighbor> &result,
                                  std::vector<float> &occlude_factor, 
                                  uint8_t *scratch){
  if (pool.empty())
    return;
  assert(std::is_sorted(pool.begin(), pool.end()));
  assert(!pool.empty());
  float alpha = this->alpha();
  int range = this->range();
  int maxc = this->maxc();
  float cur_alpha = 1;
  while (cur_alpha <= alpha && result.size() < range) {
    uint32_t start = 0;
    while (result.size() < range && (start) < pool.size() && start < maxc) {
      auto &p = pool[start];
      if (occlude_factor[start] > cur_alpha) {
        start++;
        continue;
      }
      occlude_factor[start] = std::numeric_limits<float>::max();
      result.push_back(p);
      for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
        if (occlude_factor[t] > alpha)
          continue;
        // djk = dist(p.id, pool[t.id])
        float djk;
        this->index->compute_pq_dists(p.id, &(pool[t].id), &djk, 1,
                                            scratch);
        occlude_factor[t] =
            (std::max)(occlude_factor[t], pool[t].distance / djk);
      }
      start++;
    }
    cur_alpha *= 1.2f;
  }
}
template<typename T, typename TagT>
void DiskIndexMerger<T, TagT>::OffsetIterateToFixedPoint(const T *vec, const uint32_t Lsize,
                                  std::vector<diskann::Neighbor> & expanded_nodes_info,
                                  tsl::robin_map<uint32_t, T *> &coord_map,
                                  diskann::ThreadData<T> *thread_data){
  std::vector<diskann::Neighbor> exp_node_info;
  exp_node_info.reserve(2 * Lsize);
  tsl::robin_map<uint32_t, T *> cmap;
  cmap.reserve(2 * Lsize);
  this->index->disk_iterate_to_fixed_point(
      vec, Lsize, this->beam_width(), exp_node_info, &cmap, nullptr,
      thread_data, &this->delete_local_id_set);

  // reduce and pick top maxc expanded nodes only
  std::sort(exp_node_info.begin(), exp_node_info.end());
  expanded_nodes_info.reserve(this->maxc());
  expanded_nodes_info.insert(expanded_nodes_info.end(), exp_node_info.begin(),
                              exp_node_info.end());

  // insert only relevant coords into coord_map
  for (auto &nbr : expanded_nodes_info) {
    uint32_t id = nbr.id;
    auto     iter = cmap.find(id);
    assert(iter != cmap.end());
    coord_map.insert(std::make_pair(iter->first, iter->second));
  }
}
template<typename T, typename TagT>
void DiskIndexMerger<T, TagT>::PruneNeighbors(const tsl::robin_map<uint32_t, T *> &coord_map,
                        std::vector<diskann::Neighbor> &pool, 
                        std::vector<uint32_t> &pruned_list){
  if (pool.size() == 0)
    return;
  
  float alpha = this->alpha();
  int range = this->range();
  // sort the pool based on distance to query
  std::sort(pool.begin(), pool.end());

  std::vector<diskann::Neighbor> result;
  result.reserve(range);
  std::vector<float> occlude_factor(pool.size(), 0);

  OccludeList(pool, coord_map, result, occlude_factor);

  pruned_list.clear();
  assert(result.size() <= range);
  for (auto iter : result) {
    pruned_list.emplace_back(iter.id);
  }

  if (alpha > 1) {
    for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
      if (std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) ==
          pruned_list.end())
        pruned_list.emplace_back(pool[i].id);
    }
  }
}
template<typename T, typename TagT>
void DiskIndexMerger<T, TagT>::OccludeList(
    std::vector<diskann::Neighbor> &              pool,
    const tsl::robin_map<uint32_t, T *> &coord_map,
    std::vector<diskann::Neighbor> &result, std::vector<float> &occlude_factor) {
  if (pool.empty())
    return;
  assert(std::is_sorted(pool.begin(), pool.end()));
  assert(!pool.empty());

  float cur_alpha = 1;
  float alpha = this->alpha();
  int range = this->range();
  int maxc = this->maxc();

  while (cur_alpha <= alpha && result.size() < range) {
    uint32_t start = 0;
    while (result.size() < range && (start) < pool.size() && start < maxc) {
      auto &p = pool[start];
      if (occlude_factor[start] > cur_alpha) {
        start++;
        continue;
      }
      occlude_factor[start] = std::numeric_limits<float>::max();
      result.push_back(p);
      for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
        if (occlude_factor[t] > alpha)
          continue;
        auto iter_right = coord_map.find(p.id);
        auto iter_left = coord_map.find(pool[t].id);
        // HAS to be in coord_map since it was expanded during
        // iterate_to_fixed_point
        assert(iter_right != coord_map.end());
        assert(iter_left != coord_map.end());
        // WARNING :: correct, but not fast -- NO SIMD version if using MSVC,
        // g++ should auto vectorize
        float djk = this->dist_cmp->compare(iter_left->second,
                                            iter_right->second, this->ndim());
        occlude_factor[t] =
            (std::max)(occlude_factor[t], pool[t].distance / djk);
      }
      start++;
    }
    cur_alpha *= 1.2f;
  }
}
template<typename T, typename TagT>
void DiskIndexMerger<T, TagT>::PruneNeighborsWithPQDistance(std::vector<diskann::Neighbor> &pool, 
                                                            std::vector<uint32_t> &pruned_list,
                                                            uint8_t *scratch){
  if (pool.size() == 0)
    return;

  // sort the pool based on distance to query
  std::sort(pool.begin(), pool.end());

  float alpha = this->alpha();
  float range = this->range();
  std::vector<diskann::Neighbor> result;
  result.reserve(range);
  std::vector<float> occlude_factor(pool.size(), 0);

  OccludeListWithPQDistance(pool, result, occlude_factor, scratch);

  pruned_list.clear();
  assert(result.size() <= range);
  for (auto iter : result) {
    pruned_list.emplace_back(iter.id);
  }

  if (alpha > 1) {
    for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
      if (std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) ==
          pruned_list.end())
        pruned_list.emplace_back(pool[i].id);
    }
  }
}
template<typename T, typename TagT>
DiskIndexDataIterator<T, TagT> DiskIndexMerger<T, TagT>::GetIterator(){
  return DiskIndexDataIterator<T,TagT>(this->meta,this->index);
}
template<typename T, typename TagT>
std::vector<uint8_t> DiskIndexMerger<T, TagT>::DeflateVector(const T *vec){
  return this->index->deflate_vector(vec);
}

template class DiskIndexMerger<float, uint32_t>;
template class DiskIndexMerger<uint8_t, uint32_t>;
template class DiskIndexMerger<int8_t, uint32_t>;
template class DiskIndexMerger<float, int64_t>;
template class DiskIndexMerger<uint8_t, int64_t>;
template class DiskIndexMerger<int8_t, int64_t>;
template class DiskIndexMerger<float, uint64_t>;
template class DiskIndexMerger<uint8_t, uint64_t>;
template class DiskIndexMerger<int8_t, uint64_t>;

} // namespace lsmidx
