#pragma once

#include "lsm/level_merger.h"
#include "lsm/disk_index_merger.h"

namespace lsmidx {


template<typename T, typename TagT = uint32_t>
class LevelNMerger : public LevelMerger<T, TagT> {
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
    LevelNMerger(const uint32_t ndims, diskann::Distance<T> *dist,
                                        diskann::Metric dist_metric,
                                      const uint32_t beam_width,
                                      const uint32_t range,
                                      const uint32_t l_index, const float alpha,
                                      const uint32_t maxc,
                                      bool           single_file_index);
    ~LevelNMerger();

    void merge(const char * dist_disk_index_path,
                const std::vector<std::string> &src_index_paths,
                const char * out_disk_index_path,
                std::vector<const std::vector<TagT>*> &deleted_tags,
                std::string &working_folder) override;
  private:
    void MergeImpl();
    void DeletePhase();
    void InsertPhase();
    void PatchPhase();
    void InitIndexPaths(const char * dist_disk_index_path,
                const std::vector<std::string> &src_index_paths,
                const char * out_disk_index_path,
                std::string  &working_folder);
    void WriteIntermediateIndexFile(DiskIndexFileMeta& src_index_file_meta,
                                    DiskIndexFileMeta& temp_index_file_meta,
                                    uint32_t new_max_pts);
    bool CopyAndExpandFile(const std::string& srcPath, const std::string& destPath, std::streamsize expansionSize);
    bool CopyFile(const std::string& srcPath, const std::string& destPath);

    uint32_t ComputeNewMaxPts();

    // deletes
    tsl::robin_set<TagT>                            deleted_tags;

    std::unique_ptr<DiskIndexMerger<T,TagT>> from_disk_index_merger_;
    std::unique_ptr<DiskIndexMerger<T,TagT>>  to_disk_index_merger_;
    DiskIndexFileMeta final_index_file_meta_;
    DiskIndexFileMeta intermediate_index_file_meta_;

    diskann::Distance<T> *                                  dist_cmp;
    diskann::Metric                                         dist_metric;
    std::vector<tsl::robin_set<TagT>>                       latter_deleted_tags;

    // book keeping
    uint8_t *              thread_pq_scratch = nullptr;
    std::vector<uint8_t *> thread_bufs;

    // vector info
    uint32_t ndims, aligned_ndims;
    // search + index params
    uint32_t beam_width;
    uint32_t l_index, range, maxc;
    float    alpha;
    bool single_file_index_ = false;

    // timing stuff
    std::vector<float> insert_times, delta_times, search_times, prune_times;
};
};