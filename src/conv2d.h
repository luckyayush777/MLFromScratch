#pragma once
#include <random>

#include "layer.h"
#include "matrix.h"
#include "tensor.h"
#include"mnist.h"


struct Layer;
struct MNISTDataset;

struct Conv2d {
  size_t inChannels;
  size_t outChannels;
  size_t kernel;
  size_t stride;
  size_t padding;

  // W : [outChannels, inChannels, kernel, kernel]
  Tensor W;
  Tensor b;

  //velocities for momentum
  Tensor vW;
  Tensor vb;

  Tensor X_cache; // [batch, Channels, Height, Width]

  Conv2d(size_t inCh, size_t outCh, size_t k, size_t stride, size_t padding)
      : inChannels(inCh), outChannels(outCh), kernel(k), stride(stride),
        padding(padding), W({outCh, inCh, k, k}), b({outCh}), vW({outCh, inCh, k, k}), vb({outCh}) {
    std::mt19937 rng(42);
    double std = std::sqrt(2.0 / (inCh * k * k));
    std::normal_distribution<double> dist(0.0, std);
    for (size_t i = 0; i < W.noOfElements(); ++i)
      W.flat(i) = dist(rng);
    for (size_t i = 0; i < b.noOfElements(); ++i)
      b.flat(i) = 0.0;
    for (size_t i = 0; i < vW.noOfElements(); ++i)
      vW.flat(i) = 0.0;
    for (size_t i = 0; i < vb.noOfElements(); ++i)
      vb.flat(i) = 0.0;
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
  static Tensor reshapeToImage(const Tensor &X, size_t batchSize);
  double trainBatch(Layer &fc1, Layer &fc2, const Tensor &X_img,
                    const Tensor &y, size_t batchSize, double learningRate,
                    double beta, size_t &correct,
                    double &fwdMs, double &bwdMs, double &updateMs);
  double evaluateTestSet(Layer &fc1, Layer &fc2,
                         const MNISTDataset &testDataset);
  void trainMNIST(Layer &fc1, Layer &fc2, const MNISTDataset &dataset,
                  const MNISTDataset &testDataset, double learningRate,
                  size_t batchSize, size_t epochs, double beta);

  // this doesnt belong here ideally but we can move it later
  static void testSoftmaxCrossEntropyBackwardPerfectPrediction();
  void overfitSingleBatch(Layer &fc1, Layer &fc2, const Tensor &X_img,
                          const Tensor &y, double learningRate, int steps);
  double computeAccuracy(const Tensor &logits, const Tensor &labels);
  size_t argmaxRow(const Tensor &logits, size_t row);
};