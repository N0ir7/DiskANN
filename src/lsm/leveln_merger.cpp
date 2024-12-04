#include "lsm/leveln_merger.h"
#include <algorithm>
#include <cassert>
#include <csignal>
#include <iterator>
#include <mutex>
#include <thread>
#include <vector>
#include <limits>
#include <omp.h>
#include <future>
#include <unistd.h>
#include <sys/syscall.h>
#include "neighbor.h"
#include "timer.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "tcmalloc/malloc_extension.h"
#include "logger.h"
#include "ann_exception.h"
#include "lsm/index_data_iterator.h"
#include "lsm/options.h"

namespace lsmidx {
template<typename T, typename TagT>
LevelNMerger<T, TagT>::LevelNMerger(
    const uint32_t ndims, diskann::Distance<T> *dist, diskann::Metric dist_metric, const uint32_t beam_width,
    const uint32_t range, const uint32_t l_index, const float alpha,
    const uint32_t maxc, bool single_file_index) {
  // book keeping
  this->ndims = ndims;
  this->aligned_ndims = (_u32) ROUND_UP(this->ndims, 8);
  this->range = range;
  this->l_index = l_index;
  this->beam_width = beam_width;
  this->maxc = maxc;
  this->alpha = alpha;
  this->dist_metric = dist_metric;
  this->dist_cmp = dist;
  this->single_file_index_ = single_file_index;

  std::cout << "LevelNMerger created with R=" << this->range
            << " L=" << this->l_index << " BW=" << this->beam_width
            << " MaxC=" << this->maxc << " alpha=" << this->alpha
            << " ndims: " << this->ndims << std::endl;
}

template<typename T, typename TagT>
LevelNMerger<T, TagT>::~LevelNMerger() {
  diskann::aligned_free((void *) this->thread_pq_scratch);
}

template<typename T, typename TagT>
void LevelNMerger<T, TagT>::InitIndexPaths(const char * dist_disk_index_path,
                                  const std::vector<std::string> &src_index_paths,
                                  const char * out_disk_index_path,
                                  std::string  &working_folder){
  // load disk index
  const std::string & from_disk_path = src_index_paths[0];
  std::cout << "Working folder : " << working_folder << std::endl;
  
  from_disk_index_merger_ = std::make_unique<DiskIndexMerger<T,TagT>>(DiskIndexFileMeta(from_disk_path,single_file_index_));
  to_disk_index_merger_ = std::make_unique<DiskIndexMerger<T,TagT>>(DiskIndexFileMeta(dist_disk_index_path,single_file_index_));
  from_disk_index_merger_->dist_metric = to_disk_index_merger_->dist_metric = this->dist_metric;
  final_index_file_meta_ = DiskIndexFileMeta(out_disk_index_path,single_file_index_);
  intermediate_index_file_meta_ = DiskIndexFileMeta(getTempFilePath(working_folder, "temp_disk_index"),
                                                    getTempFilePath(working_folder, "temp_tags"),
                                                    getTempFilePath(working_folder, "temp_pq_compressed"),
                                                    "");
}
template<typename T, typename TagT>
void LevelNMerger<T, TagT>::merge(const char * dist_disk_index_path,
                                  const std::vector<std::string> &src_index_paths,
                                  const char * out_disk_index_path,
                                  std::vector<const std::vector<TagT>*> &deleted_tags_vectors,
                                  std::string  &working_folder) {
  InitIndexPaths(dist_disk_index_path,src_index_paths,out_disk_index_path,working_folder);

  // load to disk index
  this->to_disk_index_merger_->InitIndexWithCache();
  this->to_disk_index_merger_->InitGraphDelta(0);
  
  // 计算每个节点的最大度数，并设置搜索范围
  _u32 max_degree =
      (this->to_disk_index_merger_->max_node_len() - (sizeof(T) * this->ndims)) / sizeof(uint32_t) - 1;
  this->range = max_degree; 
  diskann::cout << "Setting range to: " << this->range << std::endl;
  
  // 设置一些搜索与构建参数
  this->to_disk_index_merger_->param.beam_width = this->beam_width;
  this->to_disk_index_merger_->param.l_index = this->l_index;
  this->to_disk_index_merger_->param.range = this->range;
  this->to_disk_index_merger_->param.maxc = this->maxc;
  this->to_disk_index_merger_->param.alpha = this->alpha;
  this->to_disk_index_merger_->dist_cmp = this->dist_cmp;
  this->to_disk_index_merger_->dist_metric = this->dist_metric;

  // load from disk index
  this->from_disk_index_merger_->InitIndex();

  // 处理删除标签向量，生成后续删除标签集合
  for (size_t j = 0; j < deleted_tags_vectors.size(); j++) {
    this->latter_deleted_tags.push_back(tsl::robin_set<TagT>());
    for (size_t i = j+1; i < deleted_tags_vectors.size(); i++) {
      for (size_t k = 0; k < deleted_tags_vectors[i]->size(); k++) {
        this->latter_deleted_tags[j].insert((*deleted_tags_vectors[i])[k]);
      }
    }
  }
  // 将所有删除标签插入到一个全局的删除标签集合中
  //TODO: See if this can be included in the previous loop
  for (auto &deleted_tags_vector : deleted_tags_vectors) {
    for (size_t i = 0; i < deleted_tags_vector->size(); i++) {
      this->deleted_tags.insert((*deleted_tags_vector)[i]);
    }
  }

  // 分配每个线程的 scratch 空间，用于并行处理
  diskann::cout << "Allocating thread scratch space -- "
                << PER_THREAD_BUF_SIZE / (1 << 20) << " MB / thread.\n";
  diskann::alloc_aligned((void **) &this->thread_pq_scratch,
                MAX_N_THREADS * PER_THREAD_BUF_SIZE, SECTOR_LEN);
  this->thread_bufs.resize(MAX_N_THREADS);
  for (uint32_t i = 0; i < thread_bufs.size(); i++) {
    this->thread_bufs[i] = this->thread_pq_scratch + i * PER_THREAD_BUF_SIZE;
  }

  // 执行实际的合并操作
  MergeImpl();
}

template<typename T, typename TagT>
void LevelNMerger<T, TagT>::MergeImpl() {
/**
 * 1. Delete Phase
 */
DeletePhase();

/**
 * 2. Insert Prepare Phase
 */
uint32_t new_max_pts = this->ComputeNewMaxPts();
// 在DeletePhase的最后阶段，进行了DataFile写入中间文件的工作，这里只需要写入PQ坐标和Tag数据即可
this->WriteIntermediateIndexFile(this->to_disk_index_merger_->meta,this->intermediate_index_file_meta_,new_max_pts);

// 第四步：重新加载更新后的索引
this->to_disk_index_merger_->index->reload_index(this->intermediate_index_file_meta_.data_path,
                                this->intermediate_index_file_meta_.pq_coords_path,
                                this->intermediate_index_file_meta_.tag_path);
this->to_disk_index_merger_->meta = this->intermediate_index_file_meta_;

assert(this->to_disk_index_merger_->num_points() == new_max_pts);

std::cout << "AFTER RELOAD: PQ_NChunks: " << this->to_disk_index_merger_->pq_nchunks()
          << " Disk points: " << this->to_disk_index_merger_->num_points()
          << " Frozen point id: " << this->to_disk_index_merger_->init_ids()[0] << std::endl;
/**
 * 3. insert phase
 */
InsertPhase();

// END -- PQ data on disk consistent and in correct order
/**
 * 
 * 4. patch phase
 */
PatchPhase();

/**
 * 5. patch end phase
*/
// copy pq coord
CopyFile(intermediate_index_file_meta_.pq_coords_path,final_index_file_meta_.pq_coords_path);

// copy pq table
CopyFile(intermediate_index_file_meta_.pq_table_path,final_index_file_meta_.pq_table_path);

// copy medoids file
CopyFile(intermediate_index_file_meta_.medoids_file_path,final_index_file_meta_.medoids_file_path);

// copy centroids file
CopyFile(intermediate_index_file_meta_.centroids_file_path,final_index_file_meta_.centroids_file_path);

// copy tag
CopyFile(intermediate_index_file_meta_.tag_path, final_index_file_meta_.tag_path);

}
template<typename T, typename TagT>
uint32_t LevelNMerger<T, TagT>::ComputeNewMaxPts(){
  uint32_t needed = 0;
  // 先计算预计要插入多少点
  needed += this->from_disk_index_merger_->num_points();
  needed -= this->from_disk_index_merger_->delete_local_id_set.size();

  diskann::cout << "New Disk Index: Need " << needed
                << ", free: " << this->to_disk_index_merger_->free_local_ids.size() << "\n";
  // 再看现有的disk index的空闲位置数量是否充足，不充足则扩容
  uint32_t last_id = this->to_disk_index_merger_->num_points();
  if (needed > this->to_disk_index_merger_->free_local_ids.size()) {
    this->to_disk_index_merger_->free_local_ids.reserve(needed);
  }
  while (this->to_disk_index_merger_->free_local_ids.size() < needed) {
    this->to_disk_index_merger_->free_local_ids.insert(last_id);
    last_id++;
  }
  return last_id;
}
/**
 * 删除阶段主要有三件事：
 * 1.确定to disk index中有哪些点是要被删除的
 * 2.先扫一遍索引，收集所有被删除点的未被删除的邻居
 * 3.再扫一遍索引，进行实际删除并写回
*/
template<typename T, typename TagT>
void LevelNMerger<T, TagT>::DeletePhase(){

  tsl::robin_map<uint32_t, std::vector<uint32_t>> disk_deleted_nhoods;

  // process disk deleted tags
  this->to_disk_index_merger_->AddDeleteLocalID(this->deleted_tags);
  
  // populate deleted nodes
  disk_deleted_nhoods = this->to_disk_index_merger_->PopulateNondeletedHoodsOfDeletedNodes();

  // process all deletes
  this->to_disk_index_merger_->ProcessDeletes(this->intermediate_index_file_meta_,
                                            disk_deleted_nhoods,
                                            this->thread_bufs); 
  // END -- graph on disk has NO deleted references, maybe some holes
}
/**
 * 该阶段主要就是把点及其正向边插入到索引中去，同时更新相应的PQ坐标与tag
*/
template<typename T, typename TagT>
void LevelNMerger<T, TagT>::InsertPhase(){
  DiskIndexDataIterator<T, TagT> from_disk_index_data_iter = std::move(this->from_disk_index_merger_->GetIterator());
  from_disk_index_data_iter.Init(true/* read_only*/);
  
  while (from_disk_index_data_iter.HasNextBatch()){
    // prepare inserted point
    std::vector<diskann::DiskNode<T>>* from_node_batch = nullptr;
    TagT* tag_list = nullptr;
    std::tie(from_node_batch, std::ignore, tag_list) = from_disk_index_data_iter.NextBatch();

    this->to_disk_index_merger_->ProcessInserts(*from_node_batch, tag_list);
    
  }
  
}
/*
 * 该阶段主要将disk index扫一遍，将backward edge插一遍
 */
template<typename T, typename TagT>
void LevelNMerger<T, TagT>::PatchPhase(){
  this->to_disk_index_merger_->ProcessPatch(this->final_index_file_meta_, this->thread_bufs);
}

template<typename T, typename TagT>
bool LevelNMerger<T, TagT>::CopyAndExpandFile(const std::string& srcPath, const std::string& destPath, std::streamsize targetSize){
  // 打开源文件进行读取
    std::ifstream srcFile(srcPath, std::ios::binary);
    if (!srcFile) {
        std::cerr << "unable to open src file: " << srcPath << std::endl;
        return false;
    }
    srcFile.seekg(0, std::ios::end);
    std::streamsize srcSize = srcFile.tellg();
    srcFile.seekg(0, std::ios::beg);
    // 打开目标文件进行写入
    std::ofstream destFile(destPath, std::ios::binary);
    if (!destFile) {
        std::cerr << "unable to open dest file: " << destPath << std::endl;
        return false;
    }

    // 将源文件内容复制到目标文件
    destFile << srcFile.rdbuf();
    // 如果目标大小小于源文件大小，则无需扩容
    if (targetSize <= srcSize) {
        return true;
    }
    
    // 扩容：在目标文件末尾添加指定大小的空白字节（用 '\0' 填充）
    // 计算需要扩容的字节数
    std::streamsize expansionSize = targetSize - srcSize;
    if (expansionSize > 0) {
        destFile.seekp(0, std::ios::end);  // 移动到文件末尾
        for (std::streamsize i = 0; i < expansionSize; ++i) {
            destFile.put('\0');  // 写入空字节
        }
    }

    srcFile.close();
    destFile.close();
    return true;
}
template<typename T, typename TagT>
bool LevelNMerger<T, TagT>::CopyFile(const std::string& srcPath, const std::string& destPath){
  // 打开源文件进行读取
    std::ifstream srcFile(srcPath, std::ios::binary);
    if (!srcFile) {
        std::cerr << "unable to open src file: " << srcPath << std::endl;
        return false;
    }
    // 打开目标文件进行写入
    std::ofstream destFile(destPath, std::ios::binary);
    if (!destFile) {
        std::cerr << "unable to open dest file: " << destPath << std::endl;
        return false;
    }

    // 将源文件内容复制到目标文件
    destFile << srcFile.rdbuf();
    srcFile.close();
    destFile.close();
    return true;
}
template<typename T, typename TagT>
void LevelNMerger<T, TagT>::WriteIntermediateIndexFile(DiskIndexFileMeta& src_index_file_meta,
                                                      DiskIndexFileMeta& temp_index_file_meta,
                                                      uint32_t new_max_pts){
  // 先写PQ坐标的中间文件
  std::streamsize pq_file_size =
    ((uint64_t) new_max_pts * (uint64_t) this->to_disk_index_merger_->pq_nchunks()) +
    (2 * sizeof(uint32_t));
  CopyAndExpandFile(src_index_file_meta.pq_coords_path,temp_index_file_meta.pq_coords_path,pq_file_size);
  
  // 修改PQ坐标中间文件的元信息
  std::ofstream pq_writer(temp_index_file_meta.pq_coords_path, std::ios::binary);
  pq_writer.seekp(0, std::ios::beg);
  uint32_t npts_u32 = new_max_pts, ndims_u32 = this->to_disk_index_merger_->pq_nchunks();
  pq_writer.write((char *) &npts_u32, sizeof(uint32_t));
  pq_writer.write((char *) &ndims_u32, sizeof(uint32_t));
  pq_writer.close();
  
  // 再写tag的中间文件
  std::streamsize tag_file_size = new_max_pts * sizeof(TagT) + 2 * sizeof(uint32_t);
  CopyAndExpandFile(src_index_file_meta.tag_path, temp_index_file_meta.tag_path, tag_file_size);

  // 修改tag中间文件的元信息
  std::ofstream tag_writer(temp_index_file_meta.pq_coords_path, std::ios::binary);
  tag_writer.seekp(0, std::ios::beg);
  int npts_i32 = new_max_pts, ndims_i32 = 1;
  tag_writer.write((char *) &npts_i32, sizeof(int));
  tag_writer.write((char *) &ndims_i32, sizeof(int));
  tag_writer.close();

}
// template class instantiations
template class LevelNMerger<float, uint32_t>;
template class LevelNMerger<uint8_t, uint32_t>;
template class LevelNMerger<int8_t, uint32_t>;
template class LevelNMerger<float, int64_t>;
template class LevelNMerger<uint8_t, int64_t>;
template class LevelNMerger<int8_t, int64_t>;
template class LevelNMerger<float, uint64_t>;
template class LevelNMerger<uint8_t, uint64_t>;
template class LevelNMerger<int8_t, uint64_t>;

}  // namespace diskann
