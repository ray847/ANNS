#pragma once

#include <memory> // std::unique_ptr

namespace solution {
template<typename T>
class BTree {
public:
  /* Type Definitions */
  class Node {
  public:
    /* Constructor */
    template<typename ...Args>
    Node(Args... args) : obj_(args...) {}
    /* Functions */
    T& val() {return obj_;}
    auto& left() {return left_;}
    auto& right() {return right_;}
    template<typename ...Args>
    auto& new_left(Args... args){
      return left_ = std::make_unique<Node>(args...);
    }
    template<typename ...Args>
    Node& new_right(Args... args){
      return right_ = std::make_unique<Node>(args...);
    }
  private:
    /* Variables */
    T obj_;
    std::unique_ptr<Node> left_ = nullptr, right_ = nullptr;
  };
  /* Variables */
  std::unique_ptr<Node> root_;
  /* Functions */
  auto& root() {return root_;}
};
} // namespace solution
