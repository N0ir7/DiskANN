// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <chrono>

namespace diskann {
  class Timer {
    typedef std::chrono::high_resolution_clock _clock;
    std::chrono::time_point<_clock>            check_point;

   public:
    Timer() : check_point(_clock::now()) {
    }

    void reset() {
      check_point = _clock::now();
    }

    long long elapsed() const {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 _clock::now() - check_point)
          .count();
    }
    long long elapsed_and_reset() {
      auto res = std::chrono::duration_cast<std::chrono::microseconds>(
                     _clock::now() - check_point)
                     .count();
      this->reset();
      return res;
    }
  };
}  // namespace diskann
