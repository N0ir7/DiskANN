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
// bool load = false;
// unsigned   *gt_ids = nullptr;
// uint32_t   *gt_tags = nullptr;
// float      *gt_dists = nullptr;
// size_t      gt_num, gt_dim;
// template<typename T, typename TagT=uint32_t>
// void test_index(std::shared_ptr<diskann::PQFlashIndex<T, TagT>> index){
//   tsl::robin_set<TagT> active_tags;
//   std::cout << "【 Load Active Tags 】" << std::endl;
//   index->get_active_tags(active_tags);
//   print_tags(active_tags);
//   std::cout << "Loaded " << active_tags.size() << " tags" << std::endl;
//   uint64_t recall_at = 5;
//   std::string query_file = "/home/hlqiu/data/sift/sift_query.bin";
//   std::string truthset_file = "/home/hlqiu/data/sift/sift_groundtruth.bin";
//   // hold data
//   T        *query = nullptr;
//   // unsigned *gt_ids = nullptr;
//   // uint32_t *gt_tags = nullptr;
//   // float    *gt_dists = nullptr;
//   size_t    query_num, query_dim, query_aligned_dim;

//   std::cout << "Loading query : " << query_file << std::endl;
//   // load query + truthset
//   diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
//                                query_aligned_dim);
//   std::cout << "Loaded query : " << truthset_file << std::endl;
//   // std::cout << "Loading gt : " << query_file << std::endl;
//   // diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim,
//   //                        &gt_tags);
//   // std::cout << "Loaded gt" << std::endl;
//   if (!load) {
//     std::cout << "Loading truthset : " << truthset_file << std::endl;
//     diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num,
//                            gt_dim, &gt_tags);
//     load = true;
//   }
//   // if (gt_num != query_num) {
//   //   std::cout << "Error. Mismatch in number of queries and ground truth data"
//   //             << std::endl;
//   // }
//   std::vector<uint32_t> query_result_ids;
//   std::vector<TagT> query_result_tags;
//   std::vector<float>    query_result_dists;
//   query_result_ids.resize(recall_at * query_num);
//   query_result_dists.resize(recall_at * query_num);
//   query_result_tags.resize(recall_at * query_num);
//   std::vector<uint32_t> query_result_ids_32(recall_at * query_num);

//   diskann::QueryStats *stats = new diskann::QueryStats[query_num];
//   uint32_t             L = 75;
//   std::vector<double>  latency_stats(query_num, 0);
//   auto                 s = std::chrono::high_resolution_clock::now();
//   omp_set_max_active_levels(4);
// #pragma omp parallel for num_threads(6)
//   for (_s64 i = 0; i < (int64_t) query_num; i++) {
//     auto qs = std::chrono::high_resolution_clock::now();
//     index->cached_beam_search(query + (i * query_aligned_dim), recall_at, L,
//                                (query_result_tags.data() + (i * recall_at)),
//                                query_result_dists.data() + (i * recall_at),
//                                4,
//                                stats + i);
//     auto qe = std::chrono::high_resolution_clock::now();

//     std::chrono::duration<double> diff = qe - qs;
//     latency_stats[i] = diff.count() * 1000;
//     //      std::this_thread::sleep_for(std::chrono::milliseconds(2));
//   }
//   auto                          e = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> diff = e - s;
//   float qps = (float) (((double) query_num) / diff.count());
//   // compute mean recall, IOs
//   float mean_recall = 0.0f;
//   std::vector<uint32_t> query_result_tags2;
//   for(auto tag : query_result_tags){
//     query_result_tags2.emplace_back(tag);
//   }
//   tsl::robin_set<uint32_t> active_tags2;
//   for(auto tag : active_tags){
//     active_tags2.insert(tag);
//   }
//   mean_recall = diskann::calculate_recall(
//       (unsigned) query_num, gt_ids, gt_dists, (unsigned) gt_dim,
//       query_result_tags2.data(), (unsigned) recall_at, (unsigned) recall_at,
//       active_tags2);
//   //    mean_recall /= (float) query_num;
//   float mean_ios = (float) diskann::get_mean_stats(
//       stats, query_num,
//       [](const diskann::QueryStats &stats) { return stats.n_ios; });
//   std::sort(latency_stats.begin(), latency_stats.end());
//   std::string recall_string = "Recall@" + std::to_string(recall_at);
//     std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
//               << std::setw(18) << "Mean Latency (ms)" << std::setw(12)
//               << "90 Latency" << std::setw(12) << "95 Latency" << std::setw(12)
//               << "99 Latency" << std::setw(12) << "99.9 Latency"
//               << std::setw(12) << recall_string << std::setw(12)
//               << "Mean disk IOs" << std::endl;
//     std::cout
//         << "==============================================================="
//            "==============="
//         << std::endl;
//     std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
//             << ((float) std::accumulate(latency_stats.begin(),
//                                         latency_stats.end(), 0)) /
//                     (float) query_num
//             << std::setw(12)
//             << (float) latency_stats[(_u64) (0.90 * ((double) query_num))]
//             << std::setw(12)
//             << (float) latency_stats[(_u64) (0.95 * ((double) query_num))]
//             << std::setw(12)
//             << (float) latency_stats[(_u64) (0.99 * ((double) query_num))]
//             << std::setw(12)
//             << (float) latency_stats[(_u64) (0.999 * ((double) query_num))]
//             << std::setw(12) << mean_recall << std::setw(12) << mean_ios
//             << std::endl;
//   delete[] stats;
//   diskann::aligned_free(query);
//   // delete[] gt_ids;
//   // delete[] gt_dists;
//   // delete[] gt_tags;
// }
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
  // this->to_index = std::move(to_index);
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
  intermediate_index_file_meta_ = DiskIndexFileMeta(getTempFilePath(working_folder, "temp_disk_index"),
                                                    getTempFilePath(working_folder, "temp_tags"),
                                                    getTempFilePath(working_folder, "temp_pq_compressed"),
                                                    "");
}
// template<typename T, typename TagT>
// void Level0Merger<T, TagT>::merge(const char * dist_disk_index_path,
//                                   const std::vector<std::string> &src_index_paths,
//                                   const char * out_disk_index_path,
//                                   std::vector<const std::vector<TagT>*> &deleted_tags_vectors,
//                                   std::string  &working_folder) {
//   InitIndexPaths(dist_disk_index_path,src_index_paths,out_disk_index_path,working_folder);

//   // load to disk index
//   this->to_disk_index_merger_->InitIndexWithCache();
//   this->to_disk_index_merger_->InitGraphDelta(0);
  
//   // 计算每个节点的最大度数，并设置搜索范围
//   _u32 max_degree =
//       (this->to_disk_index_merger_->max_node_len() - (sizeof(T) * this->ndims)) / sizeof(uint32_t) - 1;
//   this->range = max_degree; 
//   diskann::cout << "Setting range to: " << this->range << std::endl;
  
//   // 设置一些搜索与构建参数
//   this->to_disk_index_merger_->param.beam_width = this->beam_width;
//   this->to_disk_index_merger_->param.l_index = this->l_index;
//   this->to_disk_index_merger_->param.range = this->range;
//   this->to_disk_index_merger_->param.maxc = this->maxc;
//   this->to_disk_index_merger_->param.alpha = this->alpha;
//   this->to_disk_index_merger_->dist_cmp = this->dist_cmp;
//   this->to_disk_index_merger_->dist_metric = this->dist_metric;

//   // load from disk index
//   this->from_disk_index_merger_->InitIndex();

//   // 处理删除标签向量，生成后续删除标签集合
//   // for (size_t j = 0; j < deleted_tags_vectors.size(); j++) {
//   //   this->latter_deleted_tags.push_back(tsl::robin_set<TagT>());
//   //   for (size_t i = j+1; i < deleted_tags_vectors.size(); i++) {
//   //     for (size_t k = 0; k < deleted_tags_vectors[i]->size(); k++) {
//   //       this->latter_deleted_tags[j].insert((*deleted_tags_vectors[i])[k]);
//   //     }
//   //   }
//   // }
//   // 将所有删除标签插入到一个全局的删除标签集合中
//   //TODO: See if this can be included in the previous loop
//   // for (auto &deleted_tags_vector : deleted_tags_vectors) {
//   //   for (size_t i = 0; i < deleted_tags_vector->size(); i++) {
//   //     this->deleted_tags.insert((*deleted_tags_vector)[i]);
//   //   }
//   // }

//   // 分配每个线程的 scratch 空间，用于并行处理
//   diskann::cout << "Allocating thread scratch space -- "
//                 << PER_THREAD_BUF_SIZE / (1 << 20) << " MB / thread.\n";
//   diskann::alloc_aligned((void **) &this->thread_pq_scratch,
//                 MAX_N_THREADS * PER_THREAD_BUF_SIZE, SECTOR_LEN);
//   this->thread_bufs.resize(MAX_N_THREADS);
//   for (uint32_t i = 0; i < thread_bufs.size(); i++) {
//     this->thread_bufs[i] = this->thread_pq_scratch + i * PER_THREAD_BUF_SIZE;
//   }

//   // 执行实际的合并操作
//   MergeImpl();
// }
template<typename T, typename TagT>
void Level0Merger<T, TagT>::merge(const char * dist_disk_index_path,
                const std::vector<std::string> &src_index_paths,
                const char * out_disk_index_path,
                std::vector<const std::vector<TagT>*> &deleted_tags,
                std::string &working_folder){
  
}
template<typename T, typename TagT>
void Level0Merger<T, TagT>::merge(std::string out_disk_index_path,
                std::string &working_folder){
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
  MergeImpl();
}
template<typename T, typename TagT>
void Level0Merger<T, TagT>::MergeImpl() {

  diskann::Timer timer;
  auto report_time = [](diskann::Timer &timer, const std::string &msg) {
    diskann::cout << "【" << msg<< " 】";
    double time = ((double) timer.elapsed_and_reset()) / (1000000.0);
    diskann::cout << " cost time: " << time << " s." << std::endl;
  };
  set_low_priority();
  /**
   * 1. Delete Phase
   */
  DeletePhase();
  report_time(timer, "Delete Phase");
  
  /**
   * 2. Insert Prepare Phase
   */
  uint32_t new_max_pts = this->ComputeNewMaxPts();
  // 在DeletePhase的最后阶段，进行了data, pq, tag写入中间文件的工作, 这里只需要针对InsertPhase阶段的点, 对上述三个文件数据进行扩充,预留好对应空间即可
  this->ExpandIntermediateIndexFile(this->intermediate_index_file_meta_, new_max_pts);

  // 第四步：重新加载更新后的索引
  this->to_disk_index_merger_->index->reload_index(this->intermediate_index_file_meta_.data_path,
                                  this->intermediate_index_file_meta_.pq_coords_path,
                                  this->intermediate_index_file_meta_.tag_path);
  this->to_disk_index_merger_->meta = this->intermediate_index_file_meta_;

  assert(this->to_disk_index_merger_->num_points() == new_max_pts);

  std::cout << "AFTER RELOAD: PQ_NChunks: " << this->to_disk_index_merger_->pq_nchunks()
            << " Disk points: " << this->to_disk_index_merger_->num_points()
            << " Frozen point id: " << this->to_disk_index_merger_->init_ids()[0] << std::endl;
  
  report_time(timer, "Insert Prepare Phase");
  // std::cout << "【Search after Insert Prepare Phase】"<<std::endl;
  // test_index(this->to_disk_index_merger_->index);
  /**
   * 3. insert phase
   */
  InsertPhase();
  // 后续不再使用from_disk_index_merger_了，提前释放资源
  this->from_disk_index_merger_.reset();
  report_time(timer, "Insert Phase");
  // std::cout << "【Search after Insert Phase】"<<std::endl;
  // test_index(this->to_disk_index_merger_->index);
  // END -- PQ data on disk consistent and in correct order
  /**
   * 
   * 4. patch phase
   */

  PatchPhase();
  report_time(timer, "Patch Phase");
  // std::cout << "【Search after Patch Phase】"<<std::endl;
  // test_index(this->to_disk_index_merger_->index);
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
  if (needed > this->to_disk_index_merger_->free_local_ids.size()) {
    this->to_disk_index_merger_->free_local_ids.reserve(needed);
  }
  uint32_t initial_last_id = last_id;
  while (this->to_disk_index_merger_->free_local_ids.size() < needed) {
    this->to_disk_index_merger_->free_local_ids.insert(last_id);
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
void Level0Merger<T, TagT>::DeletePhase(){

  // process disk deleted tags
  this->to_disk_index_merger_->AddDeleteLocalID(this->from_disk_index_merger_->GetDeleteTagSet());
  
  // populate deleted nodes
  tsl::robin_map<uint32_t, std::vector<uint32_t>> disk_deleted_nhoods = this->to_disk_index_merger_->PopulateNondeletedHoodsOfDeletedNodes();

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
void Level0Merger<T, TagT>::InsertPhase(){
  MultiDiskIndexDataIterator<T, TagT> from_disk_index_data_iter = std::move(this->from_disk_index_merger_->GetIterator());
  from_disk_index_data_iter.Init();
  DiskIndexDataIterator<T, TagT> to_disk_index_data_iter = std::move(this->to_disk_index_merger_->GetIterator());
  to_disk_index_data_iter.Init(false/* read_write*/);
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
  diskann::cout << "read io cost time in InsertPhase: " <<  io_time << " s" << std::endl;
  // this->to_disk_index_merger_->ReportGraphDelta();
  this->to_disk_index_merger_->WriteDataFileHeaderAfterInsertPhase();
}
/*
 * 该阶段主要将disk index扫一遍，将backward edge插一遍
 */
template<typename T, typename TagT>
void Level0Merger<T, TagT>::PatchPhase(){
  this->to_disk_index_merger_->ProcessPatch(this->final_index_file_meta_, this->thread_bufs);
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
bool Level0Merger<T, TagT>::ExpandFile(const std::string& destPath, std::streamsize targetSize){
  // 打开目标文件进行写入
    std::ofstream destFile(destPath, std::ios::binary | std::ios::app);
    if (!destFile) {
        std::cerr << "unable to open dest file: " << destPath << std::endl;
        return false;
    }
    std::streamsize srcSize = destFile.tellp();

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
    std::cout<< "expand file (" << destPath << ") and expand size: " << expansionSize <<" bytes; "  << std::endl;
    destFile.close();
    return true;
}
template<typename T, typename TagT>
void Level0Merger<T, TagT>::ExpandIntermediateIndexFile(
                                                      DiskIndexFileMeta& temp_index_file_meta,
                                                      uint32_t new_max_pts){
  // 扩充 data file的大小
  std::streamsize data_file_size =
    SECTOR_LEN + (ROUND_UP(
                      (uint64_t) new_max_pts,
                      this->to_disk_index_merger_->nnodes_per_sector()
                    ) /this->to_disk_index_merger_->nnodes_per_sector())
                   * (uint64_t) SECTOR_LEN;
  ExpandFile(temp_index_file_meta.data_path, data_file_size);
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
  ExpandFile(temp_index_file_meta.pq_coords_path, pq_file_size);
  
  // 修改PQ坐标中间文件的元信息
  std::ofstream pq_writer(temp_index_file_meta.pq_coords_path, std::ios::binary | std::ios::in | std::ios::out);
  pq_writer.seekp(0, std::ios::beg);
  uint32_t npts_u32 = new_max_pts, ndims_u32 = this->to_disk_index_merger_->pq_nchunks();
  pq_writer.write((char *) &npts_u32, sizeof(uint32_t));
  pq_writer.write((char *) &ndims_u32, sizeof(uint32_t));
  pq_writer.close();
  
  // 扩充tag的中间文件
  std::streamsize tag_file_size = new_max_pts * sizeof(TagT) + 2 * sizeof(uint32_t);
  ExpandFile(temp_index_file_meta.tag_path, tag_file_size);
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
