#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "layer.h"
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

static double evalAccuracyFullMLP(const auto &testDataset, layer &fc1,
                                  layer &fc2) {
  const size_t total = testDataset.labels.noOfElements();
  const size_t D = fc1.W.dim(0);
  size_t correct = 0;

  for (size_t i = 0; i < total; ++i) {
    Tensor x1({1, D});
    for (size_t j = 0; j < D; ++j)
      x1.at(0, j) = testDataset.images.at(i, j);

    Tensor z1 = Layer::forward(fc1, x1);
    Tensor a1 = z1;
    Tensor::relu(a1);

    Tensor logits = Layer::forward(fc2, a1);
    size_t pred = argmaxRow(logits, 0);
    size_t truth = static_cast<size_t>(testDataset.labels.flat(i));

    if (pred == truth)
      ++correct;
  }

  return static_cast<double>(correct) / static_cast<double>(total);
}

static double evalAccuracyMLP(const Tensor &X, const Tensor &y,
                              const layer &fc1, const layer &fc2) {
  Tensor Z1 = Layer::forward(fc1, X);
  Tensor A1 = Z1;
  Tensor::relu(A1);
  Tensor logits = Layer::forward(fc2, A1);
  Tensor probs = logits;
  Tensor::softmax(probs);
  return computeAccuracy(probs, y);
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
  constexpr size_t H = 128;     // hidden layer size
  double learningRate = 0.01;
  const int evalEvery = 250;

  Tensor X({B, D});
  Tensor y({B});

  Tensor X_test({B, D});
  Tensor y_test({B});

  loadBatch(testDataset, X_test, y_test, 0);

  const int epochs = 3;
  size_t batchesPerEpoch = trainSize / B;

  layer fc1(D, H);
  layer fc2(H, C);
  std::mt19937 rng(42);
  initLinearParams(fc1.W, fc1.b, rng);
  initLinearParams(fc2.W, fc2.b, rng);
  for (int epoch = 0; epoch < epochs; ++epoch) {

    std::cout << "\nEpoch " << epoch << "\n";

    for (size_t batch = 0; batch < batchesPerEpoch; ++batch) {
      size_t offset = batch * B;

      loadBatch(dataset, X, y, offset);

      // forward pass
      Tensor Z1 = Layer::forward(fc1, X);
      Tensor A1 = Z1;
      Tensor::relu(A1);

      Tensor logits = Layer::forward(fc2, A1);
      Tensor probs = logits;
      Tensor::softmax(probs);
      double loss = Tensor::crossEntropyLoss(probs, y);

      // backward pass
      Tensor dZ = probs;
      Tensor::softmaxCrossEntropyBackward(dZ, y);
      Tensor dA1 = Layer::backward(fc2, dZ);

      // relu backward
      for (size_t i = 0; i < A1.noOfElements(); ++i) {
        if (Z1.flat(i) <= 0)
          dA1.flat(i) = 0.0;
      }
      Tensor dX = Layer::backward(fc1, dA1);
      // update weights
      Layer::step(fc2, learningRate);
      Layer::step(fc1, learningRate);
      if (batch % evalEvery == 0) {
        const double testAcc = evalAccuracyMLP(X_test, y_test, fc1, fc2);
        std::cout << "  batch " << batch << " | loss " << loss << " | test acc "
                  << testAcc * 100.0 << "%\n";
      }
    }
  }
  const double finalTestAcc = evalAccuracyFullMLP(testDataset, fc1, fc2);
  std::cout << "\nFinal Test Accuracy = " << finalTestAcc * 100.0 << "%\n";

  return 0;
}