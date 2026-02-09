#pragma once
#include <random>

#include "matrix.h"
#include "tensor.h"
struct Conv2d {
  int inChannels;
  int outChannels;
  int kernel;
  int stride;
  int padding;

  // W : [outChannels, inChannels, kernel, kernel]
  Tensor W;
  Tensor b;

  Tensor X_cache; // [batch, Channels, Height, Width]

  Conv2d(size_t inCh, size_t outCh, size_t k, size_t stride, size_t padding)
      : inChannels(inCh), outChannels(outCh), kernel(k), stride(stride),
        padding(padding), W({outCh, inCh, k, k}), b({outCh}) {
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.01);
    for (size_t i = 0; i < W.noOfElements(); ++i)
      W.flat(i) = dist(rng);
    for (size_t i = 0; i < b.noOfElements(); ++i)
      b.flat(i) = 0.0;
  }

  double getPaddedInput(const Tensor &X, size_t batch, size_t channel, int h,
                        int w) const;

  static Tensor conv2dForward(const Conv2d &conv, const Tensor &input);
  static Tensor flattenForward(const Tensor &input);
  static Tensor maxPool2dForward(const Tensor &input, size_t poolSize,
                                 size_t stride);
};