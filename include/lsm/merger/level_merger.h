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
#include "windows_customizations.h"

namespace lsmidx {

template<typename T, typename TagT = uint32_t>
class LevelMerger{
public:
  virtual void merge(const char * dist_disk_index_path,
                const std::vector<std::string> &src_index_paths,
                const char *                    out_disk_index_path,
                std::vector<const std::vector<TagT>*> &deleted_tags,
                std::string &working_folder) = 0;
protected:
  int kLevel;
};

};  // namespace lsmidx
