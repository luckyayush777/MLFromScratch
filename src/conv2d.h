#pragma once
#include <random>

#include "matrix.h"
#include "tensor.h"
struct Conv2d {
  size_t inChannels;
  size_t outChannels;
  size_t kernel;
  size_t stride;
  size_t padding;

  // W : [outChannels, inChannels, kernel, kernel]
  Tensor W;
  Tensor b;

  Tensor X_cache; // [batch, Channels, Height, Width]

  Conv2d(size_t inCh, size_t outCh, size_t k, size_t stride, size_t padding)
      : inChannels(inCh), outChannels(outCh), kernel(k), stride(stride),
        padding(padding), W({outCh, inCh, k, k}), b({outCh}) {
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.005);
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
  static Tensor softmaxCrossEntropyBackward(const Tensor &logits,
                                            const Tensor &targets);
  static double softmaxCrossEntropyLoss(const Tensor &logits,
                                        const Tensor &targets);
  static Tensor reluBackward(const Tensor &Z, const Tensor &dA);
  static Tensor flattenBackward(const Tensor &dOut, const Tensor &inputShape);
  static Tensor maxPool2dBackward(const Tensor &dOut, const Tensor &input,
                                  size_t poolSize, size_t stride);
  void conv2dBackward(const Tensor &input, const Tensor &dOut, Tensor &dInput,
                      Tensor &dW, Tensor &db);
  // this doesnt belong here ideally but we can move it later
  static void testSoftmaxCrossEntropyBackwardPerfectPrediction();
  void overfitSingleBatch(Layer &fc1, Layer &fc2, const Tensor &X_img,
                          const Tensor &y, double learningRate, int steps);
};