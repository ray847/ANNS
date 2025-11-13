#pragma once

#include <math.h> // sqrt

#include <memory> // std::unique_ptr
#include <algorithm> // std::copy_n

namespace solution {
/* Mat & Vec definition */
class Mat {
public:
  /* Constructor */
  Mat(size_t n = 1, size_t m = 1) : n_(n), m_(m), data_(new float[n * m]) {
    std::fill_n(data_.get(), n_ * m_, 0);
  }
  Mat(const Mat& other) : Mat(other.n_, other.m_) {
    std::copy_n(other.data_.get(), n_ * m_, data_.get());
  }
  Mat& operator=(const Mat& other) {
    n_ = other.n_;
    m_ = other.m_;
    data_ = std::unique_ptr<float[]>(new float[n_ * m_]);
    std::copy_n(other.data_.get(), n_ * m_, data_.get());
    return *this;
  }
  /* Accessors */
  size_t n() const {return n_;}
  size_t m() const {return m_;}
  float* operator[](size_t r) {return data_.get() + r * m_;}
  const float* operator[](size_t r) const {return data_.get() + r * m_;}
private:
  /* Variables */
  size_t n_, m_;
  std::unique_ptr<float[]> data_;
};
using Vec = Mat;

/* Functions & Operators */
inline Mat operator-(const Mat& mat) {
  Mat res = mat;
  for (size_t r = 0; r < mat.n(); ++r) {
    for (size_t c = 0; c < mat.m(); ++c) {
      res[r][c] = -res[r][c];
    }
  }
  return res;
}
inline Mat operator*(const Mat& A, const Mat& B) {
  size_t l = A.n(), m = A.m(), n = B.m();
  Mat C(l, n);
  for (size_t r = 0; r < l; ++r) {
    for (size_t c = 0; c < n; ++c) {
      for (size_t e = 0; e < m; ++e) {
        C[r][c] += A[r][e] * B[e][c];
      }
    }
  }
  return C;
}
inline Mat operator/(const Mat& mat, float val) {
  auto res = mat;
  for (size_t r = 0; r < mat.n(); ++r) {
    for (size_t c = 0; c < mat.m(); ++c) {
      res[r][c] /= val;
    }
  }
  return res;
}
inline Mat& operator/=(Mat& mat, float val) {
  for (size_t r = 0; r < mat.n(); ++r) {
    for (size_t c = 0; c < mat.m(); ++c) {
      mat[r][c] /= val;
    }
  }
  return mat;
}
/**
 * Return the Euclidean norm of the vector.
 */
inline float norm(const Mat& mat) {
  float res = 0.0f;
  for (size_t i = 0; i < mat.n(); ++i) {
    res += mat[i][0] * mat[i][0];
  }
  return sqrtf(res);
}
/**
 * Return a normalized version of the input vector.
 */
inline Mat normalized(const Mat& mat) {
  return mat / norm(mat);
}
/**
 * Normalize the input vector.
 */
inline void normalize(Mat& mat) {
  mat /= norm(mat);
}
/**
 * Calculate the eignvector of the input matrix with the greatest eignvalue.
 */
inline std::pair<Mat, float> eign(const Mat& mat) {
  Mat eign_vec(mat.n());
  eign_vec[0][0] = 1.0f;
  while (true) {
    auto tmp = normalized(mat * eign_vec);
    for (size_t i = 0; i < eign_vec.n(); ++i) {
      if (eign_vec[i][0] - tmp[i][0] > 1e-6) {
        eign_vec = tmp;
        break;
      }
      return {eign_vec, norm(mat * eign_vec)};
    }
  }
};
} // namespace solution
