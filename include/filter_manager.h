#pragma once
#include "common_includes.h"
#include "utils.h"
#include <any>

namespace diskann
{
// This class is responsible for filter actions in index, and should not be used outside.
template <typename label_type> class FilterManager
{
  public:
    DISKANN_DLLEXPORT FilterManager(const size_t num_points);
    ~FilterManager() = default;

    // needs some internal lock
    DISKANN_DLLEXPORT bool detect_common_filters(uint32_t point_id, bool search_invocation,
                                                 const std::vector<label_type> &incoming_labels);

    DISKANN_DLLEXPORT const std::vector<label_type> &get_labels_by_point_id(const location_t point_id);
    DISKANN_DLLEXPORT const tsl::robin_set<label_type> &get_all_label_set();
    // Throws: out of range exception
    DISKANN_DLLEXPORT void add_label_to_point(const location_t point_id, label_type label);
    // returns internal mapping for given raw_label
    DISKANN_DLLEXPORT label_type get_converted_label(const std::string &raw_label);

    DISKANN_DLLEXPORT void update_medoid_by_label(const label_type &label, const uint32_t new_medoid);
    DISKANN_DLLEXPORT const uint32_t &get_medoid_by_label(const label_type &label);
    DISKANN_DLLEXPORT bool label_has_medoid(const label_type &label);
    DISKANN_DLLEXPORT void calculate_best_medoids(const size_t num_points_to_load, const uint32_t num_candidates);

    // TODO: in future we may accept a set or vector of universal labels
    DISKANN_DLLEXPORT void set_universal_label(label_type universal_label);
    DISKANN_DLLEXPORT const label_type get_universal_label() const;

    // ideally takes raw label file and then genrate internal mapping and keep the info of mapping
    DISKANN_DLLEXPORT size_t load_labels(const std::string &labels_file);
    DISKANN_DLLEXPORT size_t load_medoids(const std::string &labels_to_medoid_file);
    DISKANN_DLLEXPORT void load_label_map(const std::string &labels_map_file);

    DISKANN_DLLEXPORT void save_labels(const std::string &save_path_prefix, const size_t total_points);
    DISKANN_DLLEXPORT void save_medoids(const std::string &save_path);

  private:
    size_t _num_points;
    std::vector<std::vector<label_type>> _pts_to_labels;
    tsl::robin_set<label_type> _labels;
    std::unordered_map<std::string, label_type> _label_map;

    // medoids
    std::unordered_map<label_type, uint32_t> _label_to_medoid_id;
    std::unordered_map<uint32_t, uint32_t> _medoid_counts; // medoids only happen for filtered index
    // universal label
    bool _use_universal_label = false;
    label_type _universal_label = 0; // this is the internal mapping, may not always be true in future
    tsl::robin_set<label_type> _universal_labels_set;

    // populates pts_to labels and _labels from given label file
    size_t parse_label_file(const std::string &label_file);
};

} // namespace diskann
