#pragma once
#include <string>

namespace lsmidx
{
template<typename T, typename TagT = uint32_t>
class Slice {
 public:
  // Create an empty slice.
  Slice() : data_(nullptr), dims_(0) {}

  // Create a slice that refers to d[0,n-1].
  Slice(const T* d, size_t n) : data_(d), dims_(n) {}

  // Create a slice that refers to the contents of "s"
  Slice(const std::vector<T>& s) : data_(s.data()), size_(s.size()) {}

  // Intentionally copyable.
  Slice(const Slice&) = default;
  Slice& operator=(const Slice&) = default;

  // Return a pointer to the beginning of the referenced data
  const T* data() const { return data_; }
  const TagT* tag() const { return tag_; }
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

  // Drop the first "n" bytes from this slice.
  // void remove_prefix(size_t n) {
  //   assert(n <= size());
  //   data_ += n;
  //   size_ -= n;
  // }

  // Return a string that contains the copy of the referenced data.
  // std::string ToString() const { return std::string(data_, size_); }

  // Three-way comparison.  Returns value:
  //   <  0 iff "*this" <  "b",
  //   == 0 iff "*this" == "b",
  //   >  0 iff "*this" >  "b"
  int compare(const Slice& b) const;

 private:
  const T* data_;
  const TagT tag_;
  size_t dims_;
};
} // namespace lsmidx
