#pragma once

#include <cstddef>
#include <string>

struct TrainingConfig {
  double learningRate = 0.01;
  double beta = 0.9;
  size_t batchSize = 32;
  size_t epochs = 3;
  unsigned int seed = 42;
  bool enableRunLogging = true;
  std::string optimizationTitle =
      "Momentum(beta=0.9) + ReLU + MaxPool baseline";
};
