#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "matrix.h"
#include "mnist.h"
#include "tensor.h"
#include "util.h"


template <typename Dataset>
static void loadBatch(const Dataset &dataset, Tensor &X, Tensor &y,
                      size_t offset) {
  const size_t B = X.dim(0);
  const size_t D = X.dim(1);
  for (size_t i = 0; i < B; ++i) {
    const size_t idx = offset + i;
    for (size_t j = 0; j < D; ++j)
      X.at(i, j) = dataset.images.at(idx, j);
    y.flat(i) = dataset.labels.flat(idx);
  }
}

static void initLinearParams(Tensor &W, Tensor &b, std::mt19937 &rng) {
  std::normal_distribution<double> dist(0.0, 0.01);
  for (size_t i = 0; i < W.noOfElements(); ++i)
    W.flat(i) = dist(rng);
  for (size_t i = 0; i < b.noOfElements(); ++i)
    b.flat(i) = 0.0;
}

static double trainStep(const Tensor &X, const Tensor &y, Tensor &W, Tensor &b,
                        Tensor &dW, Tensor &db, double learningRate) {
  Tensor logits = Tensor::linearForward(X, W, b);
  Tensor probs = logits;
  Tensor::softmax(probs);
  const double loss = Tensor::crossEntropyLoss(probs, y);

  Tensor dZ = probs;
  Tensor::softmaxCrossEntropyBackward(dZ, y);
  Tensor::linearBackward(X, dZ, dW, db);

  for (size_t i = 0; i < W.noOfElements(); ++i)
    W.flat(i) -= learningRate * dW.flat(i);
  for (size_t i = 0; i < b.noOfElements(); ++i)
    b.flat(i) -= learningRate * db.flat(i);

  return loss;
}

static double evalAccuracy(const Tensor &X, const Tensor &y, const Tensor &W,
                           const Tensor &b) {
  Tensor logits = Tensor::linearForward(X, W, b);
  Tensor probs = logits;
  Tensor::softmax(probs);
  return computeAccuracy(probs, y);
}

template <typename Dataset>
static double evalAccuracyFull(const Dataset &testDataset, const Tensor &W,
                               const Tensor &b) {
  const size_t total = testDataset.labels.noOfElements();
  const size_t D = W.dim(0);
  size_t correct = 0;
  for (size_t i = 0; i < total; ++i) {
    Tensor x1({1, D});
    for (size_t j = 0; j < D; ++j)
      x1.at(0, j) = testDataset.images.at(i, j);
    Tensor logits = Tensor::linearForward(x1, W, b);
    const size_t pred = argmaxRow(logits, 0);
    const size_t truth = static_cast<size_t>(testDataset.labels.flat(i));
    if (pred == truth)
      ++correct;
  }
  return static_cast<double>(correct) / static_cast<double>(total);
}


int main() {
  auto dataset = loadMnist("datasets/train-images-idx3-ubyte",
                           "datasets/train-labels-idx1-ubyte");

  auto testDataset = loadMnist("datasets/t10k-images-idx3-ubyte",
                               "datasets/t10k-labels-idx1-ubyte");

  size_t trainSize = dataset.labels.noOfElements();

  constexpr size_t B = 32;      // batch size
  constexpr size_t D = 28 * 28; // input dimension
  constexpr size_t C = 10;      // number of classes
  double learningRate = 0.1;
  const int evalEvery = 50;

  Tensor X({B, D});
  Tensor y({B});

  Tensor X_test({B, D});
  Tensor y_test({B});

  loadBatch(testDataset, X_test, y_test, 0);

  const int epochs = 3;
  size_t batchesPerEpoch = trainSize / B;

  Tensor W({D, C});
  Tensor b({C});

  std::mt19937 rng(42);
  initLinearParams(W, b, rng);

  Tensor dW({D, C});
  Tensor db({C});

  for (int epoch = 0; epoch < epochs; ++epoch) {

    std::cout << "\nEpoch " << epoch << "\n";

    for (size_t batch = 0; batch < batchesPerEpoch; ++batch) {
      size_t offset = batch * B;

      loadBatch(dataset, X, y, offset);
      const double loss = trainStep(X, y, W, b, dW, db, learningRate);

      if (batch % evalEvery == 0) {
        const double testAcc = evalAccuracy(X_test, y_test, W, b);
        std::cout << "  batch " << batch << " | loss " << loss << " | test acc "
                  << testAcc * 100.0 << "%\n";
      }
    }
  }
  const double finalTestAcc = evalAccuracyFull(testDataset, W, b);
  std::cout << "\nFinal Test Accuracy = " << finalTestAcc * 100.0 << "%\n";

  return 0;
}