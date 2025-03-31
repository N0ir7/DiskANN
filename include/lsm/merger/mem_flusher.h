#pragma once

#include "lsm/merger/level_merger.h"
#include "lsm/merger/util/disk_index_merger.h"
#include "lsm/level/level_index.h"

namespace lsmidx {


template<typename T, typename TagT = uint32_t>
class MemFlusher{
  public:
    /**
     * constructor to read a constructed index, allocated IDs
     *  disk_in : SSD-DiskANN index to merge into
     *  mem_in : list of mem-DiskANN indices to merge into disk_in
     *  disk_out : SSD-DiskANN index to write out
     *  delete_list : list of IDs to delete from disk_in
     *  ndims : dimensionality of full-prec vectors
     *  dist : distance comparator -- WARNING :: assumed to be L2
     *  beam_width : BW for search on disk_in
     *  range : max out-degree
     *  l_index : L param for indexing
     *  maxc : max num of candidates to consider while pruning
    */
    // MemFlusher(const uint32_t ndims, diskann::Distance<T> *dist,
    //                                     diskann::Metric dist_metric,
    //                                   const uint32_t beam_width,
    //                                   const uint32_t range,
    //                                   const uint32_t l_index, const float alpha,
    //                                   const uint32_t maxc,
    //                                   bool           single_file_index);
    MemFlusher(const uint32_t ndims, diskann::Distance<T> *dist, diskann::Metric dist_metric, bool single_file_index);
    ~MemFlusher();
    void flush(std::shared_ptr<lsmidx::InMemIndexProxy<T, TagT>> mem_index,std::string out_disk_index_path);
    // void merge(const char * dist_disk_index_path,
    //             const std::vector<std::string> &src_index_paths,
    //             const char * out_disk_index_path,
    //             std::vector<const std::vector<TagT>*> &deleted_tags,
    //             std::string &working_folder) override;
  private:
    bool CopyAndExpandFile(const std::string& srcPath, const std::string& destPath, std::streamsize expansionSize);
    bool CopyFile(const std::string& srcPath, const std::string& destPath);
    bool ExpandFile(const std::string& filePath, std::streamsize targetSize);

    DiskIndexFileMeta out_index_file_meta_;

    diskann::Distance<T> * dist_cmp;
    diskann::Metric dist_metric;
    // vector info
    uint32_t ndims, aligned_ndims;
    // search + index params
    // uint32_t beam_width;
    // uint32_t l_index, range, maxc;
    // float    alpha;
    bool single_file_index_ = false;
};
};