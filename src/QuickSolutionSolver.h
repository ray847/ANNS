#pragma once

#include <iostream> // std::clog
#include <fstream> // std::ofstream
#include <set> // std::set
#include <vector> // std::vector
#include <algorithm> // std::n_th_element

#include "Public.h" // kDEBUG
#include "QuickSolutionMatrix.h" // solution::Mat, solution::Vec
#include "QuickSolutionPlane.h" // solution::Plane

namespace solution {

class Solver {
public:
  /* Constructor */
  inline Solver(const Mat& base):
    base_(base) {
    build();
  }
  /* Destructor */
  inline ~Solver() {
    if constexpr (kDEBUG) {
      /* Save the split indicies. */
      {
      std::ofstream os("split_indicies.txt");
      for (auto& [split_idx, _] : planes_) {
        os << split_idx << ' ';
      }
      os.close();
      }
      /* Save the sorted bae vectors. */
      {
      std::ofstream os("base_sorted.txt");
      for (size_t i = 0; i < n_base(); ++i) {
        for (size_t j = 0; j < dim(); ++j) {
          os << base_[j][i] << ' ';
        }
        os << '\n';
      }
      os.close();
      }
    }
  }
  /* Functions */
  inline void search(const Vec& query) {

  }
  /* Accessors */
  const auto& base() const {return base_;}
private:
  /* Variables */
  Mat base_;
  struct PlaneNode {
    size_t split_idx;
    Plane plane;
    bool operator<(const PlaneNode& other) const {
      return split_idx < other.split_idx;
    }
  };
  std::set<PlaneNode> planes_;
  /* Functions */
  inline void build() {
    quick_partion(0, n_base());
  }
  inline void quick_partion(size_t st, size_t ed) {
    if (ed - st <= 1) {
      //std::clog << ed - st << '\n';
      return;
    }
    auto plane = select_pivot(st, ed);
    size_t mid = partion(st, ed, plane);
    if (st == mid || ed == mid) {
      //std::clog << ed - st << '\n';
      return;
    }
    planes_.insert({mid, plane});
    /* Recursion */
    quick_partion(st, mid);
    quick_partion(mid, ed);
  }
  /**
   * Principle Component Analysis based pivot selection.
   */
  inline Plane select_pivot(size_t st, size_t ed) {
    /* Calculate the average of the base vectors. */
    Vec ave_vec(dim());
    for (size_t i = 0; i < dim(); ++i) {
      for (size_t j = st; j < ed; ++j) {
        ave_vec[i][0] += base_[i][j];
      }
      ave_vec[i][0] /= (ed - st);
    }
    /* Calculate XX' */
    Mat mat(dim(), dim());
    for (size_t i = 0; i < dim(); ++i) {
      for (size_t j = 0; j < dim(); ++j) {
        mat[i][j] = 0.0f;
        for (size_t k = st; k < ed; ++k) {
          mat[i][j] += 
            (base_[i][k] - ave_vec[i][0]) * (base_[j][k] - ave_vec[j][0]);
        }
        mat[i][j] /= ed - st;
      }
    }
    /* Get the eignvector with the greatest eignval. */
    auto [eignvec, eignval] = eign(mat);
    if (eignvec[0][0] < 0) eignvec = -eignvec;
    return {eignvec, -(eignvec * ave_vec)[0][0]};
  }
  /**
   * Partion the base vectors so that all vectors in [st, mid) satisfy:
   * pivot(vec) >= 0
   * and all vectors in [mid, ed) satisfy:
   * pivot(vec) < 0.
   *
   * @param st Starting index of base vector.
   * @param ed Ending index of base vector.
   * @param pivot The selection criterion.
   * @return Index to the first vector that satisfies: pivot(vector) >= median.
   */
  inline size_t partion(size_t st, size_t ed, const Plane& pivot) {
    std::vector<float> res(ed - st);
    for (size_t i = st; i < ed; ++i) {
      float product = pivot.intercept_;
      for (size_t j = 0; j < pivot.dim(); ++j) {
        product += pivot.coeff_[j][0] * base_[j][i];
      }
      res[i - st] = product;
    }
    /* Find the median of the products. */
    float median;
    {
      auto tmp = res;
      std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
      median = tmp[tmp.size() / 2];
    }
    /* Partion based on the median. */
    size_t l = 0, r = ed - st;
    while (l < r) {
      if (res[l] > median) {
        /* Swap l with r - 1. */
        r--;
        std::swap(res[l], res[r]);
        for (size_t i = 0; i < pivot.dim(); ++i) {
          std::swap(base_[i][l + st], base_[i][r + st]);
        }
      } else l++;
    }
    return st + l;
  }
  /* Accessors */
  inline size_t dim() {return base_.n();}
    inline size_t n_base() {return base_.m();}
};
} // namespace solution
