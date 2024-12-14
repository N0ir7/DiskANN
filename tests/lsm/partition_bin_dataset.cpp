#include "aux_utils.h"
#include "math_utils.h"
#include "utils.h"

void partition_dataset(std::vector<int>& partition_size, std::string data_path){
    std::ifstream readr(data_path, std::ios::binary);
    int npts_s32;
    int ndims_s32;
    readr.read((char*) &npts_s32, sizeof(_s32));
    readr.read((char*) &ndims_s32, sizeof(_s32));
    float* read_buf = new float[npts_s32 * ndims_s32];
    std::vector<int> tags(npts_s32);
    std::iota(tags.begin(), tags.end(), 0);
    readr.read((char*) read_buf, npts_s32 * ndims_s32 * sizeof(float));
    size_t last_dot = data_path.find_last_of('.');
    std::string filename = data_path.substr(0, last_dot);
    std::string extension = data_path.substr(last_dot + 1);
    int offset = 0;
    int sum = 0;
    for(auto size : partition_size){
      sum+=size;
    }
    if(sum<npts_s32){
      partition_size.emplace_back(npts_s32-sum);
    }
    for(auto size : partition_size){
      std::string out_path_prefix = filename + "_" + std::to_string(size/1000) +"k";
      std::string out_data_path = out_path_prefix + "." + extension;
      std::string out_tag_path = out_path_prefix + ".tags";
      std::ofstream writr(out_data_path, std::ios::binary);
      std::ofstream writr2(out_tag_path, std::ios::binary);
      writr.write((char*) &size, sizeof(_s32));
      writr.write((char*) &ndims_s32, sizeof(_s32));
      writr.write((char*) (read_buf + offset * ndims_s32), size * ndims_s32 * sizeof(float));
      int dim = 1;
      writr2.write((char*) &size, sizeof(_s32));
      writr2.write((char*) &dim, sizeof(_s32));
      writr2.write((char*) (tags.data() + offset), size * sizeof(int));

      offset += size;
      writr.close();
      writr2.close();
    }
    readr.close();
    delete[] read_buf;
}
int parse_size(std::string str){
  if (str.empty()) {
        throw std::invalid_argument("Input string is empty");
    }

    // Find the numeric part
    size_t i = 0;
    while (i < str.size() && std::isdigit(str[i])) {
        ++i;
    }

    // Extract the number part
    std::string number_part = str.substr(0, i);
    if (number_part.empty()) {
        throw std::invalid_argument("No numeric value found in input");
    }
    int number = std::stoi(number_part);

    // Determine the unit
    std::string unit = str.substr(i);
    if (unit.empty()) {
        return number; // No unit, return the number
    }

    // Normalize the unit to lowercase for comparison
    for (auto& ch : unit) {
        ch = std::tolower(ch);
    }

    if (unit == "k") {
        return number * 1000;
    } else if (unit == "m") {
        return number * 1000000;
    } else {
        throw std::invalid_argument("Invalid unit in input: " + unit);
    }
}
int main(int argc, char** argv) {
  if (argc <= 2) {
    diskann::cout << "Usage: " << argv[0]
                  << " <input dataset(.bin)> <split size>"
                  << std::endl;
  } else {
    std::vector<int> partition_size;
    for(int i = 2;i<argc;i++){
      partition_size.emplace_back(parse_size(std::string(argv[i])));
    }
    partition_dataset(partition_size, std::string(argv[1]));
  }
}