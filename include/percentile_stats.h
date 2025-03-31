// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann {
  struct QueryStats {
    double total_us = 0;      // total time to process query in micros
    double n_4k = 0;          // # of 4kB reads
    double n_8k = 0;          // # of 8kB reads
    double n_12k = 0;         // # of 12kB reads
    double n_ios = 0;         // total # of IOs issued
    double read_size = 0;     // total # of bytes read
    double io_us = 0;         // total time spent in IO
    double cpu_us = 0;        // total time spent in CPU
    double n_cmps_saved = 0;  // # cmps saved
    double n_cmps = 0;        // # cmps
    double n_cache_hits = 0;  // # cache_hits
    double n_hops = 0;        // # search hops
    void   Aggregate(QueryStats &other) {
        total_us += other.total_us;
        n_4k += other.n_4k;
        n_8k += other.n_8k;
        n_12k += other.n_12k;
        n_ios += other.n_ios;
        read_size += other.read_size;
        io_us += other.io_us;
        cpu_us += other.cpu_us;
        n_cmps_saved += other.n_cmps_saved;
        n_cmps += other.n_cmps;
        n_cache_hits += other.n_cache_hits;
        n_hops += other.n_hops;
    };
    std::string ToString() {
      std::ostringstream oss;
      oss << "total_us:" << total_us << "; "
          << "n_4k:" << n_4k << " ; "
          << "n_8k:" << n_8k << " ; "
          << "n_12k:" << n_12k << " ; "
          << "n_ios:" << n_ios << " ; "
          << "read_size:" << read_size << " ; "
          << "io_us:" << io_us << " us; "
          << "cpu_us:" << cpu_us << " us; "
          << "n_cmps_saved:" << n_cmps_saved << "; "
          << "n_cmps:" << n_cmps << " ; "
          << "n_cache_hits:" << n_cache_hits << " ; "
          << "n_hops:" << n_hops << " ;";
      return oss.str();
    };
    std::string ToString(int query_count) const {
      if (query_count <= 0)
        return "Invalid query count";
      std::ostringstream oss;
      oss << "avg_total_us:" << (total_us / query_count) << "us; "
          << "avg_n_4k:" << (n_4k / query_count) << "; "
          << "avg_n_8k:" << (n_8k / query_count) << "; "
          << "avg_n_12k:" << (n_12k / query_count) << "; "
          << "avg_n_ios:" << (n_ios / query_count) << "; "
          << "avg_read_size:" << (read_size / query_count) << "B; "
          << "avg_io_us:" << (io_us / query_count) << "us; "
          << "avg_cpu_us:" << (cpu_us / query_count) << "us; "
          << "avg_n_cmps_saved:" << (n_cmps_saved / query_count) << "; "
          << "avg_n_cmps:" << (n_cmps / query_count) << "; "
          << "avg_n_cache_hits:" << (n_cache_hits / query_count) << "; "
          << "avg_n_hops:" << (n_hops / query_count) << ";";
      return oss.str();
    };
  };

  struct InsertStats {
    double get_mem_level_read_lock_time =
        0;  // Time to acquire memory level read lock
    double get_cur_mutex_read_lock_time =
        0;  // Time to acquire current mutex read lock
    double get_cur_index_read_lock_time =
        0;                   // Time to acquire index write lock
    double insert_time = 0;  // Total insert time

    // 统计多个 InsertStats 的数据
    void Aggregate(const InsertStats &other) {
      get_mem_level_read_lock_time += other.get_mem_level_read_lock_time;
      get_cur_mutex_read_lock_time += other.get_cur_mutex_read_lock_time;
      get_cur_index_read_lock_time += other.get_cur_index_read_lock_time;
      insert_time += other.insert_time;
    }

    // 转换为字符串（输出所有数据）
    std::string ToString() const {
      std::ostringstream oss;
      oss << "get_mem_level_read_lock_time: " << get_mem_level_read_lock_time
          << " s; "
          << "get_cur_mutex_read_lock_time: " << get_cur_mutex_read_lock_time
          << " s; "
          << "get_cur_index_read_lock_time: " << get_cur_index_read_lock_time
          << " s; "
          << "insert_time: " << insert_time << " s;";
      return oss.str();
    }

    // 按查询数计算平均值
    std::string ToString(int query_count) const {
      if (query_count <= 0)
        return "Invalid query count";
      std::ostringstream oss;
      oss << "avg_get_mem_level_read_lock_time: "
          << (get_mem_level_read_lock_time / query_count) << " s; "
          << "avg_get_cur_mutex_read_lock_time: "
          << (get_cur_mutex_read_lock_time / query_count) << " s; "
          << "avg_get_cur_index_read_lock_time: "
          << (get_cur_index_read_lock_time / query_count) << " s; "
          << "avg_insert_time: " << (insert_time / query_count) << " s;";
      return oss.str();
    }
  };

  inline double get_percentile_stats(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<double(const QueryStats &)> &member_fn) {
    std::vector<double> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(
        vals.begin(), vals.end(),
        [](const double &left, const double &right) { return left < right; });

    auto retval = vals[(uint64_t) (percentile * ((float) len))];
    vals.clear();
    return retval;
  }

  inline double get_mean_stats(
      QueryStats *stats, uint64_t len,
      const std::function<double(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += member_fn(stats[i]);
    }
    return avg / ((double) len);
  }
}  // namespace diskann
