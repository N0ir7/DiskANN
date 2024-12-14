#include <vector>
#include <string>
#include "aux_utils.h"
#include "math_utils.h"
#include "utils.h"
#include "index.h"
#include "lsm/index_data_iterator.h"
#include "linux_aligned_file_reader.h"

int main(int argc, char** argv) {
  std::string data_prefix(argv[1]);
  bool is_single_file_index = false;
  diskann::Metric               metric = diskann::Metric::L2;
  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());
  std::shared_ptr<diskann::PQFlashIndex<float, uint32_t>> index = std::make_shared<diskann::PQFlashIndex<float, uint32_t>>(metric, reader, is_single_file_index, true/*enable tags*/);
  index->load(data_prefix.c_str(), 16);
  lsmidx::DiskIndexFileMeta meta(data_prefix, is_single_file_index);
  lsmidx::DiskIndexDataIterator<float, uint32_t> from_disk_index_data_iter(meta, index);
  from_disk_index_data_iter.Init(true/* read_only*/);
  while (from_disk_index_data_iter.HasNextBatch()){
    // prepare inserted point
    std::vector<diskann::DiskNode<float>>* from_node_batch = nullptr;
    uint32_t* tag_list = nullptr;
    std::tie(from_node_batch, std::ignore, tag_list) = from_disk_index_data_iter.NextBatch();

    for(auto node : *from_node_batch){
        int num = node.nnbrs;
        auto id = node.id;
        bool print = false;
        if(id == 164954){
          print = true;
        }
        for(int i = 0; i < num ; i++){
          auto nid = node.nbrs[i];
          auto tag = tag_list[i];
          if(print){
            std::cout<< "ids[" << i <<"] = "<< nid << std::endl;
          }
          
          if(nid > 320000){
            std::cout<< "Error: " << nid << std::endl;
          }
        }
    }
  }
}