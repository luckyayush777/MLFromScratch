#pragma once
#include <cmath>
#include <iostream>
#include <limits>

struct Debug {

  static void checkShape(const Tensor &t, const std::vector<size_t> &expected,
                         const std::string &name) {
    if (t.getShape() != expected) {
      std::cerr << "SHAPE ERROR in " << name << "\nExpected: ";
      for (auto d : expected)
        std::cerr << d << " ";
      std::cerr << "\nGot: ";
      for (auto d : t.getShape())
      
        std::cerr << d << " ";
      std::cerr << std::endl;
      std::exit(1);
    }
  }

  static void checkFinite(const Tensor &t, const std::string &name) {
    for (size_t i = 0; i < t.noOfElements(); ++i) {
      double v = t.flat(i);
      if (!std::isfinite(v)) {
        std::cerr << "NaN/Inf detected in " << name << " at index " << i
                  << " value=" << v << std::endl;
        std::exit(1);
      }
    }
  }

  static void stats(const Tensor &t, const std::string &name) {
    double minV = std::numeric_limits<double>::infinity();
    double maxV = -minV;
    double sum = 0.0;

    for (size_t i = 0; i < t.noOfElements(); ++i) {
      double v = t.flat(i);
      minV = std::min(minV, v);
      maxV = std::max(maxV, v);
      sum += std::abs(v);
    }

    double meanAbs = sum / t.noOfElements();

    std::cout << "[STATS] " << name << " | min=" << minV << " max=" << maxV
              << " mean|x|=" << meanAbs << std::endl;
  }

  static void checkNotAllZero(const Tensor &t, const std::string &name) {
    double sum = 0.0;
    for (size_t i = 0; i < t.noOfElements(); ++i)
      sum += std::abs(t.flat(i));

    if (sum == 0.0) {
      std::cerr << "ZERO GRADIENT detected in " << name << std::endl;
      std::exit(1);
    }
  }
};
