#pragma once

#include <array>
#include <algorithm>
#include <functional>
#include <vector>

template <typename T, int Capacity, typename Compare = std::less<T>>
class StackHeap {
public:
  StackHeap() : size_(0) {}

  void push(const T& value) {
    if (size_ >= Capacity) return;
    data_[size_] = value;
    std::push_heap(data_.begin(), data_.begin() + size_ + 1, comp_);
    size_++;
  }

  void pop() {
    if (size_ <= 0) return;
    std::pop_heap(data_.begin(), data_.begin() + size_, comp_);
    size_--;
  }

  const T& top() const {
    return data_[0];
  }

  bool empty() const {
    return size_ == 0;
  }

  size_t size() const {
    return size_;
  }

private:
  std::array<T, Capacity> data_;
  size_t size_;
  Compare comp_;
};
