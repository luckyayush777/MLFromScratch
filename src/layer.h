#pragma once

#include "tensor.h"

struct Layer {
  Tensor W, b;
  Tensor dW, db;
  Tensor X_cache;

  Layer(size_t in, size_t out, std::mt19937 &rng);

  Tensor forward(const Tensor &X);
  Tensor backward(const Tensor &dY);
  void step(double learningRate);
  Tensor forwardNoCache(const Tensor &X);
};
