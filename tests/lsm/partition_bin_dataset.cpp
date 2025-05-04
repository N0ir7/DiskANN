#include "aux_utils.h"
#include "math_utils.h"
#include "utils.h"
template<typename T>
void partition_dataset(std::vector<int>& partition_size, const std::string& data_path){
    std::ifstream readr(data_path, std::ios::binary);
    if (!readr) {
        std::cerr << "Cannot open file: " << data_path << std::endl;
        return;
    }

    int npts_s32;
    int ndims_s32;
    readr.read((char*) &npts_s32, sizeof(int));
    readr.read((char*) &ndims_s32, sizeof(int));

    // 准备标签
    std::vector<int> tags(npts_s32);
    std::iota(tags.begin(), tags.end(), 0);

    size_t last_dot = data_path.find_last_of('.');
    std::string filename = data_path.substr(0, last_dot);
    std::string extension = data_path.substr(last_dot + 1);

    int offset_pts = 0;
    int sum = 0;
    for (auto size : partition_size) sum += size;
    if (sum < npts_s32) {
        partition_size.emplace_back(npts_s32 - sum);
    }

    constexpr size_t MAX_BUFFER_SIZE = 64 * 1024 * 1024; // 64MB 缓冲区
    size_t max_pts_per_buffer = MAX_BUFFER_SIZE / (sizeof(T) * ndims_s32);
    if (max_pts_per_buffer == 0) max_pts_per_buffer = 1;

    for (auto size : partition_size) {
        std::string out_path_prefix = filename;
        if (size >= 1000000) {
            out_path_prefix = out_path_prefix + "_" + std::to_string(size / 1000000) + "m";
        } else {
            out_path_prefix = out_path_prefix + "_" + std::to_string(size / 1000) + "k";
        }
        std::string out_data_path = out_path_prefix + "." + extension;
        int i = 1;
        while (file_exists(out_data_path)) {
            out_data_path = out_path_prefix + "_" + std::to_string(++i) + "." + extension;
        }
        std::string out_tag_path = out_path_prefix + ".tags";
        if (i != 1) {
            out_tag_path = out_path_prefix + "_" + std::to_string(i) + ".tags";
        }

        std::ofstream writr(out_data_path, std::ios::binary);
        std::ofstream writr2(out_tag_path, std::ios::binary);
        assert(writr && writr2);

        writr.write((char*) &size, sizeof(int));
        writr.write((char*) &ndims_s32, sizeof(int));

        int dim = 1;
        writr2.write((char*) &size, sizeof(int));
        writr2.write((char*) &dim, sizeof(int));

        int pts_written = 0;
        while (pts_written < size) {
            size_t pts_to_read = std::min<size_t>(max_pts_per_buffer, size - pts_written);

            std::vector<T> buffer(pts_to_read * ndims_s32);
            std::vector<int> tag_buf(pts_to_read);

            readr.read((char*) buffer.data(), pts_to_read * ndims_s32 * sizeof(T));
            std::copy(tags.begin() + offset_pts + pts_written,
                      tags.begin() + offset_pts + pts_written + pts_to_read,
                      tag_buf.begin());

            writr.write((char*) buffer.data(), pts_to_read * ndims_s32 * sizeof(T));
            writr2.write((char*) tag_buf.data(), pts_to_read * sizeof(int));

            pts_written += pts_to_read;
        }

        offset_pts += size;

        writr.close();
        writr2.close();
    }

    readr.close();
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
  if (argc <= 3) {
    diskann::cout << "Usage: " << argv[0]
                  << " <vector data type> <input dataset(.bin)> <split size>"
                  << std::endl;
  } else {
    std::vector<int> partition_size;
    std::string type = std::string(argv[1]);
    for(int i = 3;i<argc;i++){
      partition_size.emplace_back(parse_size(std::string(argv[i])));
    }
    if(type == "uint8"){
      partition_dataset<_u8>(partition_size, std::string(argv[2]));
    }else if( type == "float"){
      partition_dataset<float>(partition_size, std::string(argv[2]));
    }
  }
}