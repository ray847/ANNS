#pragma once

#include <math.h> // std::sqrt

#include <random> // std::normal_distribution
#include <vector> // std::vector
#include <algorithm> // std::copy_n

#include "Global.h" // global::rng

namespace solution {
template<typename T, bool ROW_MAJOR = true>
class Mat {
public:
  Mat(size_t n, size_t m = 1) : n_(n), m_(m), data_(n * m) {}
  Mat(size_t n, size_t m, const T* iter) : Mat(n, m) {
    std::copy_n(iter, n * m, data_.begin());
  }
  T& at(size_t i, size_t j = 0) {
    if constexpr (ROW_MAJOR) return data_[i * m_ + j];
    else return data_[j * n_ + m_];
  }
  T* data() {return data_.data();}
  const T* data() const {return data_.data();}
  size_t n() const {return n_;}
  size_t m() const {return m_;}
private:
  size_t n_, m_;
  std::vector<T> data_;
};

template<typename T>
using Vec = Mat<T, true>;

template<typename T>
T norm(const T* x, size_t n) {
  T tmp = *x * *x;
  for (size_t i = 1; i < n; ++i) {
    tmp += *(x + i) * *(x + i);
  }
  return std::sqrt(tmp);
}
template<typename T>
void normalize(T* x, size_t n) {
  T tmp = norm(x, n);
  for (size_t i = 0; i < n; ++i) {
    *(x + i) /= tmp;
  }
}
template<typename T>
T dot(const T* x, const T* y, size_t n) {
  T tmp = *x * *y;
  for (size_t i = 1; i < n; ++i) {
    tmp += *(x + i) * *(y + i);
  }
  return tmp;
}
template<typename T>
void random_uniform(T* x, size_t n, T mean = 0, T stddev = 1) {
  std::normal_distribution<T> distribution(mean, stddev);
  for (size_t i = 0; i < n; ++i) {
    *(x + i) = distribution(global::rng);
  }
  normalize(x, n);
}
template<typename T>
T L2(const T* x, const T* y, size_t n) {
  T res;
  for (size_t i = 0; i < n; ++i) {
    res += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return std::sqrt(res);
}
} // namespace solution
