#include "lsm/merger/level0_merger.h"
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
// #include <iomanip>
#include <unistd.h>
#include <sys/syscall.h>
#include "neighbor.h"
#include "timer.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "utils.h"
// #include "aux_utils.h"
#include "tcmalloc/malloc_extension.h"
#include "logger.h"
#include "ann_exception.h"
#include "lsm/merger/util/index_data_iterator.h"
#include "lsm/options.h"

namespace lsmidx {
template<typename T, typename TagT>
Level0Merger<T, TagT>::Level0Merger(
    const uint32_t ndims, diskann::Distance<T> *dist, diskann::Metric dist_metric, const uint32_t beam_width,
    const uint32_t range, const uint32_t l_index, const float alpha,
    const uint32_t maxc, bool single_file_index,
    std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> from_indexes,
    std::shared_ptr<lsmidx::PQFlashIndexProxy<T,TagT>> to_index) {
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
  this->from_indexes = std::move(from_indexes);
  this->reader = std::make_shared<LinuxAlignedFileReader>();
  std::string index_prefix = to_index->GetIndexPrefix();
  std::string working_dir = index_prefix.substr(0, index_prefix.find_last_of('/'));
  this->to_index = std::make_shared<lsmidx::PQFlashIndexProxy<T, TagT>>(dist_metric, working_dir, 0, this->reader, ndims, lsmidx::config::leveln_merge_thresh[0],to_index->GetParameter(), 1, single_file_index, 16);
  std::cout << "Level0Merger created with R=" << this->range
            << " L=" << this->l_index << " BW=" << this->beam_width
            << " MaxC=" << this->maxc << " alpha=" << this->alpha
            << " ndims: " << this->ndims << std::endl;
}

template<typename T, typename TagT>
Level0Merger<T, TagT>::~Level0Merger() {
  diskann::aligned_free((void *) this->thread_pq_scratch);
}

template<typename T, typename TagT>
void Level0Merger<T, TagT>::InitIndexPaths(std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>> dist_index,
                                  std::vector<std::shared_ptr<lsmidx::PQFlashIndexProxy<T, TagT>>> &src_indexes,
                                  std::string out_disk_index_path,
                                  std::string  &working_folder){
  // construct from disk index merger
  
  from_disk_index_merger_ = std::make_unique<SourceDiskIndexMerger<T,TagT>>(src_indexes);
  
  // construct to disk index merger
  const std::string dist_disk_index_path = dist_index->GetIndexPrefix();
  to_disk_index_merger_ = std::make_unique<DiskIndexMerger<T,TagT>>(DiskIndexFileMeta(dist_disk_index_path,single_file_index_), dist_index->GetIndex());

  to_disk_index_merger_->dist_metric = this->dist_metric;

  // construct final index meta
  final_index_file_meta_ = DiskIndexFileMeta(out_disk_index_path,single_file_index_);

  // construct temp index meta
    std::cout << "Working folder : " << working_folder << std::endl;
}
template<typename T, typename TagT>
void Level0Merger<T, TagT>::merge(const char * dist_disk_index_path,
                const std::vector<std::string> &src_index_paths,
                const char * out_disk_index_path,
                std::vector<const std::vector<TagT>*> &deleted_tags,
                std::string &working_folder){
  
}
template<typename T, typename TagT>
void Level0Merger<T, TagT>::merge(std::string out_disk_index_path,
                std::string &working_folder,
                diskann::MergeStats* stats){
  InitIndexPaths(this->to_index, this->from_indexes, out_disk_index_path,working_folder);
  /**
   * initialize some to disk index merger param
  */
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
  MergeImpl(stats);
}
template<typename T, typename TagT>
void Level0Merger<T, TagT>::MergeImpl(diskann::MergeStats* stats) {

  diskann::Timer timer;
  auto report_time = [](diskann::Timer &timer, const std::string &msg) {
    diskann::cout << "【" << msg<< " 】";
    double time = ((double) timer.elapsed_and_reset()) / (1000000.0);
    diskann::cout << " cost time: " << time << " s." << std::endl;
    return time;
  };
  set_low_priority();
  /**
   * 1. Delete Phase
   */
  // DeletePhase(stats);
  DeletePhaseWithSingleScan(stats);
  double delete_phase_time = report_time(timer, "Delete Phase");
  if(stats){
    stats->delete_phase_time = delete_phase_time;
  }
  
  /**
   * 2. Insert Prepare Phase
   */
  uint32_t new_max_pts = this->ComputeNewMaxPts();
  // 在DeletePhase的最后阶段，进行了data, pq, tag写入中间文件的工作, 这里只需要针对InsertPhase阶段的点, 对上述三个文件数据进行扩充,预留好对应空间即可
  auto s = std::chrono::high_resolution_clock::now();
  this->ExpandIntermediateIndexFile(this->final_index_file_meta_, new_max_pts, stats);
  auto e = std::chrono::high_resolution_clock::now();
  if(stats){
    std::chrono::duration<double> diff = e - s;
    stats->delete_phase_io_time += diff.count();
  }
  // 第四步：重新加载更新后的索引

  this->to_disk_index_merger_->index->reload_index(this->final_index_file_meta_.data_path,
                                  this->final_index_file_meta_.pq_coords_path,
                                  this->final_index_file_meta_.tag_path);
  this->to_disk_index_merger_->meta = this->final_index_file_meta_;
  assert(this->to_disk_index_merger_->num_points() == new_max_pts);

  std::cout << "AFTER RELOAD: PQ_NChunks: " << this->to_disk_index_merger_->pq_nchunks()
            << " Disk points: " << this->to_disk_index_merger_->num_points()
            << " Frozen point id: " << this->to_disk_index_merger_->init_ids()[0] << std::endl;
  
  double insert_prepare_phase_time = report_time(timer, "Insert Prepare Phase");
  /**
   * 3. insert phase
   */
  InsertPhase(stats);
  // 后续不再使用from_disk_index_merger_了，提前释放资源
  this->from_disk_index_merger_.reset();
  double insert_phase_time = report_time(timer, "Insert Phase");
  if(stats){
    stats->insert_phase_time = insert_prepare_phase_time + insert_phase_time;
  }
  // END -- PQ data on disk consistent and in correct order
  /**
   * 
   * 4. patch phase
   */

  PatchPhase(stats);
  double patch_phase_time = report_time(timer, "Patch Phase");
  if(stats){
    stats->patch_phase_time = patch_phase_time;
  }
}
template<typename T, typename TagT>
uint32_t Level0Merger<T, TagT>::ComputeNewMaxPts(){
  uint32_t needed = 0;
  // 先计算预计要插入多少点
  needed += this->from_disk_index_merger_->GetNumPoints();

  diskann::cout << "New Disk Index: Need " << needed
                << ", free: " << this->to_disk_index_merger_->free_local_ids.size() << "\n";
  // 再看现有的disk index的空闲位置数量是否充足，不充足则扩容
  uint32_t last_id = this->to_disk_index_merger_->num_points();
  uint32_t initial_last_id = last_id;
  while (this->to_disk_index_merger_->free_local_ids.size() < needed) {
    this->to_disk_index_merger_->free_local_ids.push(last_id);
    last_id++;
  }
  diskann::cout << "expand capacity,id range: ["<<initial_last_id<<","<<last_id-1<<"]"<<std::endl;
  return last_id;
}
/**
 * 删除阶段主要有三件事：
 * 1.确定to disk index中有哪些点是要被删除的
 * 2.先扫一遍索引，收集所有被删除点的未被删除的邻居
 * 3.再扫一遍索引，进行实际删除并写回
*/
template<typename T, typename TagT>
void Level0Merger<T, TagT>::DeletePhase(diskann::MergeStats* stats){
  // process disk deleted tags
  this->to_disk_index_merger_->AddDeleteLocalID(this->from_disk_index_merger_->GetDeleteTagSet());
  
  // populate deleted nodes
  tsl::robin_map<uint32_t, std::vector<uint32_t>> disk_deleted_nhoods = this->to_disk_index_merger_->PopulateNondeletedHoodsOfDeletedNodes(stats);

  // process all deletes
  this->to_disk_index_merger_->ProcessDeletes(this->final_index_file_meta_,
                                            disk_deleted_nhoods,
                                            this->thread_bufs,
                                            stats); 
  // END -- graph on disk has NO deleted references, maybe some holes
}
template<typename T, typename TagT>
void Level0Merger<T, TagT>::DeletePhaseWithSingleScan(diskann::MergeStats* stats){
  auto set = this->from_disk_index_merger_->GetDeleteTagSet();
  // process disk deleted tags
  this->to_disk_index_merger_->AddDeleteLocalID(set);
  
  if(!set.empty() && !this->to_disk_index_merger_->buf_pool){
    this->to_disk_index_merger_->buf_pool = std::make_shared<ReadOnlySectorBufferPool>();
    this->to_disk_index_merger_->buf_pool->InitIndexReader(this->to_disk_index_merger_->meta.data_path);
  }
  // process all deletes
  this->to_disk_index_merger_->ProcessDeletes(this->final_index_file_meta_, this->thread_bufs, stats); 
  // END -- graph on disk has NO deleted references, maybe some holes
}
/**
 * 该阶段主要就是把点及其正向边插入到索引中去，同时更新相应的PQ坐标与tag
*/
template<typename T, typename TagT>
void Level0Merger<T, TagT>::InsertPhase(diskann::MergeStats* stats){
  MultiDiskIndexDataIterator<T, TagT> from_disk_index_data_iter = std::move(this->from_disk_index_merger_->GetIterator());
  from_disk_index_data_iter.Init();
  DiskIndexDataIterator<T, TagT> to_disk_index_data_iter = std::move(this->to_disk_index_merger_->GetIterator());
  to_disk_index_data_iter.Init(false/* read_write*/, 1);
  int batch_cnt = 0, num_cnt = 0;
  diskann::Timer timer;
  while (from_disk_index_data_iter.HasNextBatch()){
    // prepare inserted point
    std::vector<diskann::DiskNode<T>>* from_node_batch = nullptr;
    TagT* tag_list = nullptr;
    std::tie(from_node_batch, std::ignore, tag_list) = from_disk_index_data_iter.NextBatch();
    tsl::robin_set<TagT>* delete_tag_set = from_disk_index_data_iter.GetCurDeleteTagSet();
    int frozen_location = from_disk_index_data_iter.GetCurIndexFrozenPoint();
    this->to_disk_index_merger_->ProcessInserts(*from_node_batch, tag_list, to_disk_index_data_iter, delete_tag_set, frozen_location);

    num_cnt += from_node_batch->size();
    diskann::cout << "batch count: " << ++batch_cnt << "; total points: " << num_cnt << "; cost time: " << ((double) timer.elapsed() / (double) 1000000) << "s" << std::endl;
    timer.reset();
  }
  to_disk_index_data_iter.TryFlushBack();
  double io_time = to_disk_index_data_iter.GetIOTime();
  if(stats){
    stats->insert_phase_random_read_4k += to_disk_index_data_iter.GetRandomRead();
    stats->insert_phase_random_write_4k += to_disk_index_data_iter.GetRandomWrite();
    stats->insert_phase_seq_read_4k += to_disk_index_data_iter.GetSeqRead();
    stats->insert_phase_seq_write_4k += to_disk_index_data_iter.GetSeqWrite();
    stats->insert_phase_io_time += io_time;

    stats->insert_phase_random_read_4k += from_disk_index_data_iter.GetRandomRead();
    stats->insert_phase_random_write_4k += from_disk_index_data_iter.GetRandomWrite();
    stats->insert_phase_seq_read_4k += from_disk_index_data_iter.GetSeqRead();
    stats->insert_phase_seq_write_4k += from_disk_index_data_iter.GetSeqWrite();
    stats->insert_phase_io_time += from_disk_index_data_iter.GetIOTime();
  }
  diskann::cout << "read io cost time in InsertPhase: " <<  io_time << " s" << std::endl;
  // this->to_disk_index_merger_->ReportGraphDelta();
  auto s = std::chrono::high_resolution_clock::now();
  this->to_disk_index_merger_->WriteDataFileHeaderAfterInsertPhase();
  auto e = std::chrono::high_resolution_clock::now();
  if(stats){
    std::chrono::duration<double> diff = e - s;
    stats->insert_phase_random_write_4k += 1;
    stats->insert_phase_io_time += diff.count();
  }
}
/*
 * 该阶段主要将disk index扫一遍，将backward edge插一遍
 */
template<typename T, typename TagT>
void Level0Merger<T, TagT>::PatchPhase(diskann::MergeStats* stats){
  this->to_disk_index_merger_->ProcessPatch(this->final_index_file_meta_, this->thread_bufs, stats);
}

template<typename T, typename TagT>
bool Level0Merger<T, TagT>::CopyAndExpandFile(const std::string& srcPath, const std::string& destPath, std::streamsize targetSize){
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
    std::cout<< "copy file from (" << srcPath << ") to (" << destPath << ") and expand size: " << expansionSize <<" bytes; "  << std::endl;
    srcFile.close();
    destFile.close();
    return true;
}
template<typename T, typename TagT>
bool Level0Merger<T, TagT>::CopyFile(const std::string& srcPath, const std::string& destPath){
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
    std::cout<< "copy file from (" << srcPath << ") to (" << destPath << ")" << std::endl;
    // 将源文件内容复制到目标文件
    destFile << srcFile.rdbuf();
    srcFile.close();
    destFile.close();
    return true;
}
template<typename T, typename TagT>
uint64_t Level0Merger<T, TagT>::ExpandFile(const std::string& destPath, std::streamsize targetSize){
    uint64_t bytes_written = 0;
  // 打开目标文件进行写入
    std::ofstream destFile(destPath, std::ios::binary | std::ios::app);
    if (!destFile) {
        std::cerr << "unable to open dest file: " << destPath << std::endl;
        return bytes_written;
    }
    std::streamsize srcSize = destFile.tellp();

    // 如果目标大小小于源文件大小，则无需扩容
    if (targetSize <= srcSize) {
        return bytes_written;
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
    bytes_written = expansionSize;
    std::cout<< "expand file (" << destPath << ") and expand size: " << expansionSize <<" bytes; "  << std::endl;
    destFile.close();
    return bytes_written;
}
template<typename T, typename TagT>
void Level0Merger<T, TagT>::ExpandIntermediateIndexFile(
                                                      DiskIndexFileMeta& temp_index_file_meta,
                                                      uint32_t new_max_pts,
                                                      diskann::MergeStats* stats){
  // 扩充 data file的大小
  std::streamsize data_file_size =
    SECTOR_LEN + (ROUND_UP(
                      (uint64_t) new_max_pts,
                      this->to_disk_index_merger_->nnodes_per_sector()
                    ) /this->to_disk_index_merger_->nnodes_per_sector())
                   * (uint64_t) SECTOR_LEN;
  uint64_t data_file_bytes_written = ExpandFile(temp_index_file_meta.data_path, data_file_size);
  if(stats){
    if(data_file_bytes_written != 0){
      int sectors = (data_file_bytes_written + SECTOR_LEN - 1) / SECTOR_LEN;
      stats->insert_phase_random_write_4k += 1;
      stats->insert_phase_seq_write_4k += sectors - 1;
    }
  }
  // 修改data file的元信息
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
  size_t nnodes_offset = sizeof(uint32_t) * 2;
  size_t file_size_offset = sizeof(uint32_t) * 2 + sizeof(uint64_t) * 8;
  uint64_t new_nnodes = new_max_pts;
  uint64_t new_file_size = static_cast<uint64_t>(data_file_size);
  std::ofstream data_writer(temp_index_file_meta.data_path, std::ios::binary | std::ios::in | std::ios::out);
  // 定位到 nnodes 的位置并写入新值
  data_writer.seekp(nnodes_offset, std::ios::beg);
  data_writer.write((char *)(&new_nnodes), sizeof(uint64_t));
  // 定位到 file_size 的位置并写入新值
  data_writer.seekp(file_size_offset, std::ios::beg);
  data_writer.write((char *)(&new_file_size), sizeof(uint64_t));

  data_writer.close();
  // 扩充PQ坐标的中间文件
  std::streamsize pq_file_size =
    ((uint64_t) new_max_pts * (uint64_t) this->to_disk_index_merger_->pq_nchunks()) +
    (2 * sizeof(uint32_t));
  uint64_t pq_file_bytes_written = ExpandFile(temp_index_file_meta.pq_coords_path, pq_file_size);
  if(stats){
    if(pq_file_bytes_written != 0){
      int sectors = (pq_file_bytes_written + SECTOR_LEN - 1) / SECTOR_LEN;
      stats->insert_phase_random_write_4k += 1;
      stats->insert_phase_seq_write_4k += sectors - 1;
    }
  }
  // 修改PQ坐标中间文件的元信息
  std::ofstream pq_writer(temp_index_file_meta.pq_coords_path, std::ios::binary | std::ios::in | std::ios::out);
  pq_writer.seekp(0, std::ios::beg);
  uint32_t npts_u32 = new_max_pts, ndims_u32 = this->to_disk_index_merger_->pq_nchunks();
  pq_writer.write((char *) &npts_u32, sizeof(uint32_t));
  pq_writer.write((char *) &ndims_u32, sizeof(uint32_t));
  pq_writer.close();
  
  // 扩充tag的中间文件
  std::streamsize tag_file_size = new_max_pts * sizeof(TagT) + 2 * sizeof(uint32_t);
  uint64_t tag_file_bytes_written = ExpandFile(temp_index_file_meta.tag_path, tag_file_size);
  if(stats){
    if(tag_file_bytes_written != 0){
      int sectors = (tag_file_bytes_written + SECTOR_LEN - 1) / SECTOR_LEN;
      stats->insert_phase_random_write_4k += 1;
      stats->insert_phase_seq_write_4k += sectors - 1;
    }
  }
  // 修改tag中间文件的元信息
  std::ofstream tag_writer(temp_index_file_meta.tag_path, std::ios::binary | std::ios::in | std::ios::out);
  tag_writer.seekp(0, std::ios::beg);
  int npts_i32 = new_max_pts, ndims_i32 = 1;
  tag_writer.write((char *) &npts_i32, sizeof(int));
  tag_writer.write((char *) &ndims_i32, sizeof(int));
  tag_writer.close();

}
// template class instantiations
template class Level0Merger<float, uint32_t>;
template class Level0Merger<uint8_t, uint32_t>;
template class Level0Merger<int8_t, uint32_t>;
template class Level0Merger<float, int64_t>;
template class Level0Merger<uint8_t, int64_t>;
template class Level0Merger<int8_t, int64_t>;
template class Level0Merger<float, uint64_t>;
template class Level0Merger<uint8_t, uint64_t>;
template class Level0Merger<int8_t, uint64_t>;

}  // namespace diskann
