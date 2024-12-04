#pragma once

#include <string>

namespace lsmidx
{
template<typename T>
class VecSlice {
 public:
  // Create an empty slice.
  VecSlice() : data_(nullptr), dims_(0) {}

  // Create a slice that refers to d[0,n-1].
  VecSlice(const T* d, size_t n) : data_(d), dims_(n) {}

  // Create a slice that refers to the contents of "s"
  VecSlice(const std::vector<T>& s) : data_(s.data()), dims_(s.size()) {}

  // Intentionally copyable.
  VecSlice(const VecSlice&) = default;
  VecSlice& operator=(const VecSlice&) = default;

  // Return a pointer to the beginning of the referenced data
  const T* data() const { return data_; }

  // Return the length (in bytes) of the referenced data
  size_t size() const { return dims_* sizeof(T); }

  // Return true iff the length of the referenced data is zero
  bool empty() const { return dims_ == 0; }

  // Return the ith byte in the referenced data.
  // REQUIRES: n < size()
  T operator[](size_t n) const {
    assert(n < dims_);
    return data_[n];
  }

  // Change this slice to refer to an empty array
  void clear() {
    data_ = nullptr;
    dims_ = 0;
  }

  // Return a string that contains the copy of the referenced data.
  // std::string ToString() const { return std::string(data_, size_); }

  // Three-way comparison.  Returns value:
  //   <  0 iff "*this" <  "b",
  //   == 0 iff "*this" == "b",
  //   >  0 iff "*this" >  "b"
  int compare(const VecSlice<T>& b) const;

 private:
  const T* data_;
  size_t dims_;
};

template<typename T>
class TagSlice {
 public:

  // Create a slice that refers to d[0,n-1].
  TagSlice(const T* tag) : tag_(tag){}

  // Intentionally copyable.
  TagSlice(const TagSlice&) = default;
  TagSlice& operator=(const TagSlice&) = default;

  // Return a pointer to the beginning of the referenced data
  const T tag() const { return tag_; }

  // Return a string that contains the copy of the referenced data.
  // std::string ToString() const { return std::string(data_, size_); }

  // Three-way comparison.  Returns value:
  //   <  0 iff "*this" <  "b",
  //   == 0 iff "*this" == "b",
  //   >  0 iff "*this" >  "b"
  int compare(const TagSlice<T>& b) const;

 private:
  const T tag_;
};
} // namespace lsmidx
