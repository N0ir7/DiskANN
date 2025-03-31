#include "lsm/merger/mem_flusher.h"
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
#include "aux_utils.h"
#include "partition_and_pq.h"
#include "tcmalloc/malloc_extension.h"
#include "logger.h"
#include "ann_exception.h"
#include "lsm/merger/util/index_data_iterator.h"
#include "lsm/options.h"

namespace lsmidx {
// template<typename T, typename TagT>
// MemFlusher<T, TagT>::MemFlusher(
//     const uint32_t ndims, diskann::Distance<T> *dist, diskann::Metric dist_metric, const uint32_t beam_width,
//     const uint32_t range, const uint32_t l_index, const float alpha,
//     const uint32_t maxc, bool single_file_index) {
//   // book keeping
//   this->ndims = ndims;
//   this->aligned_ndims = (_u32) ROUND_UP(this->ndims, 8);
//   this->range = range;
//   this->l_index = l_index;
//   this->beam_width = beam_width;
//   this->maxc = maxc;
//   this->alpha = alpha;
//   this->dist_metric = dist_metric;
//   this->dist_cmp = dist;
//   this->single_file_index_ = single_file_index;

//   std::cout << "MemFlusher created with R=" << this->range
//             << " L=" << this->l_index << " BW=" << this->beam_width
//             << " MaxC=" << this->maxc << " alpha=" << this->alpha
//             << " ndims: " << this->ndims << std::endl;
// }
template<typename T, typename TagT>
MemFlusher<T, TagT>::MemFlusher(const uint32_t ndims, diskann::Distance<T> *dist, diskann::Metric dist_metric, bool single_file_index) {
  // book keeping
  this->ndims = ndims;
  this->aligned_ndims = (_u32) ROUND_UP(this->ndims, 8);
  this->dist_metric = dist_metric;
  this->dist_cmp = dist;
  this->single_file_index_ = single_file_index;

  std::cout << "MemFlusher created with"
            << " ndims: " << this->ndims << std::endl;
}
template<typename T, typename TagT>
MemFlusher<T, TagT>::~MemFlusher() {
  // diskann::aligned_free((void *) this->thread_pq_scratch);
}
template<typename T, typename TagT>
void MemFlusher<T, TagT>::flush(std::shared_ptr<lsmidx::InMemIndexProxy<T, TagT>> mem_index, std::string out_disk_index_prefix){
    out_index_file_meta_ = DiskIndexFileMeta(out_disk_index_prefix, single_file_index_);
    // 将mem_index 和 delete_tag_list flush到磁盘
    /**
     * graph flush
     */
    // 进行转换
    diskann::convert_index_to_disk<T, TagT>(mem_index->GetIndex(), single_file_index_, out_disk_index_prefix);
    /**
     * delete tag list flush
     */
    mem_index->delete_tag_set.Save(out_disk_index_prefix);
    return;
}
// template<typename T, typename TagT>
// void MemFlusher<T, TagT>::merge(const char * dist_disk_index_path,
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
//   for (size_t j = 0; j < deleted_tags_vectors.size(); j++) {
//     this->latter_deleted_tags.push_back(tsl::robin_set<TagT>());
//     for (size_t i = j+1; i < deleted_tags_vectors.size(); i++) {
//       for (size_t k = 0; k < deleted_tags_vectors[i]->size(); k++) {
//         this->latter_deleted_tags[j].insert((*deleted_tags_vectors[i])[k]);
//       }
//     }
//   }
//   // 将所有删除标签插入到一个全局的删除标签集合中
//   //TODO: See if this can be included in the previous loop
//   for (auto &deleted_tags_vector : deleted_tags_vectors) {
//     for (size_t i = 0; i < deleted_tags_vector->size(); i++) {
//       this->deleted_tags.insert((*deleted_tags_vector)[i]);
//     }
//   }

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
bool MemFlusher<T, TagT>::CopyAndExpandFile(const std::string& srcPath, const std::string& destPath, std::streamsize targetSize){
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
bool MemFlusher<T, TagT>::CopyFile(const std::string& srcPath, const std::string& destPath){
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
bool MemFlusher<T, TagT>::ExpandFile(const std::string& destPath, std::streamsize targetSize){
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

// template class instantiations
template class MemFlusher<float, uint32_t>;
template class MemFlusher<uint8_t, uint32_t>;
template class MemFlusher<int8_t, uint32_t>;
template class MemFlusher<float, int64_t>;
template class MemFlusher<uint8_t, int64_t>;
template class MemFlusher<int8_t, int64_t>;
template class MemFlusher<float, uint64_t>;
template class MemFlusher<uint8_t, uint64_t>;
template class MemFlusher<int8_t, uint64_t>;

}  // namespace diskann
