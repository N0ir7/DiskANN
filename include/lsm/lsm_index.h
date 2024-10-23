#pragma once
#include <set>
#include <string>
#include <options.h>
#include <slice.h>
#include <status.h>
#include <level_merger.h>
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "index.h"
namespace lsmidx{
// Grouping of constants.  We may want to make some of these
// parameters set via options.
namespace config {
static const int kNumLevels = 2;

// Level-0 compaction is started when we hit this many files.
static const int kL0_CompactionTrigger = 4;

// Soft limit on number of level-0 files.  We slow down writes at this point.
static const int kL0_SlowdownWritesTrigger = 8;

// Maximum number of level-0 files.  We stop writes at this point.
static const int kL0_StopWritesTrigger = 12;

// Maximum level to which a new compacted memtable is pushed if it
// does not create overlap.  We try to push to level 2 to avoid the
// relatively expensive level 0=>1 compactions and to avoid some
// expensive manifest file operations.  We do not push all the way to
// the largest level since that can generate a lot of wasted disk
// space if the same key space is being repeatedly overwritten.
static const int kMaxMemCompactLevel = 2;

// Approximate gap in bytes between samples of data read during iteration.
static const int kReadBytesPeriod = 1048576;

}  // namespace config

template<typename T, typename TagT = uint32_t>
class LSMVectorIndex{
  public:
    LSMVectorIndex();
    LSMVectorIndex(const Options& options, const std::string& index_name);

    LSMVectorIndex(const LSMVectorIndex&) = delete;
    LSMVectorIndex& operator=(const LSMVectorIndex&) = delete;

    ~LSMVectorIndex();

    // Implementations of the DB interface
    int Put(const WriteOptions&, const Slice<T,TagT>& key,
              const TagT& value);
    void Delete(const WriteOptions&, const TagT& tag);
    void Search(const SearchOptions& options, const Slice<T,TagT>& key,
              std::vector<std::pair<TagT,float>>& result);
  private:
  /**
   * some tools classes
   */
  std::shared_ptr<AlignedFileReader> reader_ = nullptr;
  ThreadPool* search_tpool_;
  diskann::Metric dist_metric_;
  diskann::Distance<T>* dist_comp_;
  /**
   * some data structures as helpers
  */
  std::unordered_map<unsigned, TagT> curr_location_to_tag_;
  std::vector<const std::vector<TagT>*> deleted_tags_vector_;
  tsl::robin_set<TagT> deletion_set_0_;
  tsl::robin_set<TagT> deletion_set_1_;
  /**
   * some constants
  */
  std::string mem_index_prefix_;
  std::string disk_index_prefix_in_;
  std::string disk_index_prefix_out_;
  std::string deleted_tags_file_;
  std::string TMP_FOLDER_;
  /**
   * critical data structures
   */
  std::shared_ptr<diskann::Index<T, TagT>>    mem_index_0 = nullptr;
  std::shared_ptr<diskann::Index<T, TagT>>    mem_index_1 = nullptr;
  std::vector<std::shared_ptr<diskann::PQFlashIndex<T, TagT>>> disk_indexes_;
  lsmidx::StreamingMerger<T, TagT> *  merger_ = nullptr;
  /**
   * locks and signal variables
   */
  std::shared_timed_mutex delete_lock_;  // lock to access _deletion_set
  std::shared_timed_mutex index_lock_;  // mutex to switch between mem indices
  std::shared_timed_mutex change_lock_;  // mutex to switch increment _mem_pts
  std::vector<std::shared_timed_mutex> disk_locks_;  // mutex to switch between disk indices
  /**
   * parameters and options
   */
  size_t   merge_th_ = 0;
  size_t   mem_points_ = 0;  // reflects number of points in active mem index
  size_t   index_points_ = 0;
  size_t   dim_;
  _u32     num_nodes_to_cache_;
  _u32     num_search_threads_;
  uint64_t beamwidth_;

  diskann::Parameters paras_mem_;
  diskann::Parameters paras_disk_;
};
} // namespace diskann