#include <algorithm>
#include <chrono>
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
static void loadBatch(const Dataset &dataset,
                      const std::vector<size_t> &indices, Tensor &X, Tensor &y,
                      size_t offset) {
  const size_t B = X.dim(0);
  const size_t D = X.dim(1);
  for (size_t i = 0; i < B; ++i) {
    const size_t idx = indices[offset + i];
    for (size_t j = 0; j < D; ++j)
      X.at(i, j) = dataset.images.at(idx, j);
    y.flat(i) = dataset.labels.flat(idx);
  }
}

static double evalAccuracyFullMLP(const auto &testDataset, Layer &fc1,
                                  Layer &fc2) {
  const size_t total = testDataset.labels.noOfElements();
  const size_t D = fc1.W.dim(0);
  size_t correct = 0;

  for (size_t i = 0; i < total; ++i) {
    Tensor x1({1, D});
    for (size_t j = 0; j < D; ++j)
      x1.at(0, j) = testDataset.images.at(i, j);

    Tensor z1 = fc1.forwardNoCache(x1);
    Tensor a1 = z1;
    Tensor::relu(a1);

    Tensor logits = fc2.forwardNoCache(a1);
    size_t pred = argmaxRow(logits, 0);
    size_t truth = static_cast<size_t>(testDataset.labels.flat(i));

    if (pred == truth)
      ++correct;
  }

  return static_cast<double>(correct) / static_cast<double>(total);
}

static double evalAccuracyMLP(const Tensor &X, const Tensor &y, Layer &fc1,
                              Layer &fc2) {
  Tensor Z1 = fc1.forwardNoCache(X);
  Tensor A1 = Z1;
  Tensor::relu(A1);
  Tensor logits = fc2.forwardNoCache(A1);
  Tensor probs = logits;
  Tensor::softmax(probs);
  return computeAccuracy(probs, y);
}

int main() {

  // hyperparameters
  const int epochs = 10;
  double learningRate = 0.005;

  auto start = std::chrono::high_resolution_clock::now();

  auto dataset = loadMnist("datasets/train-images-idx3-ubyte",
                           "datasets/train-labels-idx1-ubyte");
  auto testDataset = loadMnist("datasets/t10k-images-idx3-ubyte",
                               "datasets/t10k-labels-idx1-ubyte");
  size_t trainSize = dataset.labels.noOfElements();
  std::vector<size_t> indices(trainSize);
  for (size_t i = 0; i < trainSize; ++i)
    indices[i] = i;
  constexpr size_t B = 32;      // batch size
  constexpr size_t D = 28 * 28; // input dimension
  constexpr size_t C = 10;      // number of classes
  constexpr size_t H = 128;     // hidden layer size
  const int evalEvery = 250;
  Tensor X({B, D});
  Tensor y({B});
  Tensor X_test({B, D});
  Tensor y_test({B});
  loadBatch(testDataset, indices, X_test, y_test, 0);

  size_t batchesPerEpoch = trainSize / B;
  std::mt19937 rng(42);
  Layer fc1(D, H, rng);
  Layer fc2(H, C, rng);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    double epoch_loss_sum = 0.0;
    double epoch_relu_alive_sum = 0.0;
    size_t batch_count = 0;
    std::shuffle(indices.begin(), indices.end(), rng);
    for (size_t batch = 0; batch < batchesPerEpoch; ++batch) {
      // Progress bar
      if (batch % 100 == 0) {
        int progress = (batch * 100) / batchesPerEpoch;
        std::cout << "\rEpoch " << epoch << " [";
        for (int i = 0; i < 50; ++i) {
          if (i < progress / 2)
            std::cout << "=";
          else if (i == progress / 2)
            std::cout << ">";
          else
            std::cout << " ";
        }
        std::cout << "] " << progress << "%" << std::flush;
      }

      size_t offset = batch * B;
      loadBatch(dataset, indices, X, y, offset);
      // forward
      Tensor Z1 = fc1.forward(X);
      Tensor A1 = Z1;
      Tensor::relu(A1);
      // ReLU stats
      size_t alive = 0;
      for (size_t i = 0; i < A1.noOfElements(); ++i)
        if (A1.flat(i) > 0)
          alive++;
      double alive_frac = double(alive) / A1.noOfElements();
      Tensor logits = fc2.forward(A1);
      Tensor probs = logits;
      Tensor::softmax(probs);
      double loss = Tensor::crossEntropyLoss(probs, y);
      // backward
      Tensor dZ = probs;
      Tensor::softmaxCrossEntropyBackward(dZ, y);
      Tensor dA1 = fc2.backward(dZ);
      for (size_t i = 0; i < A1.noOfElements(); ++i)
        if (Z1.flat(i) <= 0)
          dA1.flat(i) = 0.0;
      fc1.backward(dA1);
      fc2.step(learningRate);
      fc1.step(learningRate);
      // accumulate
      epoch_loss_sum += loss;
      epoch_relu_alive_sum += alive_frac;
      batch_count++;
    }
    // Clear progress bar line
    std::cout << "\r" << std::string(70, ' ') << "\r";

    // epoch summary
    double avg_loss = epoch_loss_sum / batch_count;
    double avg_relu_alive = epoch_relu_alive_sum / batch_count;
    double train_acc = evalAccuracyMLP(X_test, y_test, fc1, fc2);
    std::cout << "Epoch " << epoch << " | avg loss " << avg_loss
              << " | ReLU alive " << avg_relu_alive * 100.0 << "%"
              << " | test acc " << train_acc * 100.0 << "%\n";
    if (epoch == epochs - 1) {
      double full_test_acc = evalAccuracyFullMLP(testDataset, fc1, fc2);
      std::cout << "Final test accuracy over full test set: "
                << full_test_acc * 100.0 << "%\n";
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Total time: " << duration.count() / 1000.0 << "s\n";

  return 0;
}