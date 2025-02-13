#include <vector>
#include <string>
#include <limits>
#include "aux_utils.h"
#include "math_utils.h"
#include "utils.h"
#include "index.h"
#include "lsm/index_data_iterator.h"
#include "linux_aligned_file_reader.h"

// int main(int argc, char** argv) {
//   std::string data_prefix(argv[1]);
//   bool is_single_file_index = false;
//   diskann::Metric               metric = diskann::Metric::L2;
//   std::shared_ptr<AlignedFileReader> reader = nullptr;
//   reader.reset(new LinuxAlignedFileReader());
//   std::shared_ptr<diskann::PQFlashIndex<float, uint32_t>> index = std::make_shared<diskann::PQFlashIndex<float, uint32_t>>(metric, reader, is_single_file_index, true/*enable tags*/);
//   index->load(data_prefix.c_str(), 16);
//   lsmidx::DiskIndexFileMeta meta(data_prefix, is_single_file_index);
//   lsmidx::DiskIndexDataIterator<float, uint32_t> from_disk_index_data_iter(meta, index);
//   from_disk_index_data_iter.Init(true/* read_only*/);
//   while (from_disk_index_data_iter.HasNextBatch()){
//     // prepare inserted point
//     std::vector<diskann::DiskNode<float>>* from_node_batch = nullptr;
//     uint32_t* tag_list = nullptr;
//     std::tie(from_node_batch, std::ignore, tag_list) = from_disk_index_data_iter.NextBatch();

//     for(auto node : *from_node_batch){
//         int num = node.nnbrs;
//         auto id = node.id;
//         bool print = false;
//         if(id == 164954){
//           print = true;
//         }
//         for(int i = 0; i < num ; i++){
//           auto nid = node.nbrs[i];
//           auto tag = tag_list[i];
//           if(print){
//             std::cout<< "ids[" << i <<"] = "<< nid << std::endl;
//           }
          
//           if(nid > 320000){
//             std::cout<< "Error: " << nid << std::endl;
//           }
//         }
//     }
//   }
// }
void DisplayTagInfo(const char* tag_filename){
  size_t file_dim, file_num_points;
  unsigned  *tag_data;
  diskann::load_bin<unsigned>(std::string(tag_filename), tag_data, file_num_points,
                  file_dim, 0);
  unsigned lower_bound = std::numeric_limits<unsigned>::max(), upper_bound = 0;
  for(int i = 0; i < file_num_points; i++){
    if(tag_data[i]<lower_bound){
      lower_bound = tag_data[i];
    }
    if(tag_data[i]>upper_bound){
      upper_bound = tag_data[i];
    }
  }
  std::cout << tag_filename << "\n"
            << "Tag file num points: " <<  file_num_points << "\n"
            << "dims: " << file_dim <<"\n"
            << "tag range: " << lower_bound << " ~ " << upper_bound << "\n";
}
int main(int argc, char** argv) {
  DisplayTagInfo(argv[1]);
  
  // const char* level1_tag_filename = "/home/hlqiu/index/lsm/level1_disk.index.tags";
  // const char* level2_tag_filename = "/home/hlqiu/index/lsm/level2_disk.index.tags";
  // DisplayTagInfo(level1_tag_filename);
  // DisplayTagInfo(level2_tag_filename);
}