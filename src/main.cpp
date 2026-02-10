#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include "conv2d.h"
#include "layer.h"
#include "matrix.h"
#include "mnist.h"
#include "stb_image_write.h"
#include "tensor.h"
#include "util.h"
#include "debug.h"

void printShape(const std::string &name, const Tensor &t) {
  const auto &s = t.getShape();
  std::cout << name << ": [";
  for (size_t i = 0; i < s.size(); ++i) {
    std::cout << s[i] << (i < s.size() - 1 ? ", " : "");
  }
  std::cout << "]\n";
}

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

int main() {
  // hyperparameters
  double learningRate = 0.005;
  (void)learningRate;

  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "Loading MNIST Data...\n";
  auto dataset = loadMnist("datasets/train-images-idx3-ubyte",
                           "datasets/train-labels-idx1-ubyte");
  auto testDataset = loadMnist("datasets/t10k-images-idx3-ubyte",
                               "datasets/t10k-labels-idx1-ubyte");

  size_t trainSize = dataset.labels.noOfElements();
  std::vector<size_t> indices(trainSize);
  for (size_t i = 0; i < trainSize; ++i)
    indices[i] = i;

  constexpr size_t B = 32;      // batch size
  constexpr size_t D = 28 * 28; // input dimension (flat)
  constexpr size_t C = 10;      // number of classes

  Tensor X({B, D});
  Tensor y({B});

  loadBatch(dataset, indices, X, y, 0);

  std::mt19937 rng(42);

  Conv2d conv1(1, 8, 3, 1, 1);

  constexpr size_t flattened_dim = 8 * 14 * 14;
  constexpr size_t hidden_dim = 128;

  Layer fc1(flattened_dim, hidden_dim, rng);
  Layer fc2(hidden_dim, C, rng);

  std::cout << "Starting Forward Pass Debug \n";

  Tensor X_img({B, 1, 28, 28});
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < 28; ++h) {
      for (size_t w = 0; w < 28; ++w) {

        double pixel = X.at(b, h * 28 + w);

        X_img.at(b, 0, h, w) = pixel;
      }
    }
  }
  printShape("Input Image", X_img);

  Tensor Z_conv = Conv2d::conv2dForward(conv1, X_img);
  printShape("Conv Output", Z_conv);

  Tensor A_conv = Z_conv;
  Tensor::relu(A_conv);
  std::cout << "ReLU Applied.\n";

  Tensor Z_pool = Conv2d::maxPool2dForward(A_conv, 2, 2);
  printShape("Pool Output", Z_pool);

  Tensor Z_flat = Conv2d::flattenForward(Z_pool);
  printShape("Flatten Output", Z_flat);

  Tensor Z_fc1 = fc1.forward(Z_flat);
  Tensor A_fc1 = Z_fc1;
  Tensor::relu(A_fc1);
  printShape("FC1 Output", A_fc1);

  Tensor logits = fc2.forward(A_fc1);
  printShape("Logits Output", logits);

  double loss = Conv2d::softmaxCrossEntropyLoss(logits, y);
  std::cout << "Loss: " << loss << "\n";
  Tensor dLogits = Conv2d::softmaxCrossEntropyBackward(logits, y);

  // BACKWARD
  std::cout << "Backward: fc2.backward" << std::endl;
  Tensor dA_fc1 = fc2.backward(dLogits);
  std::cout << "Backward: reluBackward(Z_fc1)" << std::endl;
  Tensor dZ_fc1 = Conv2d::reluBackward(Z_fc1, dA_fc1);

  std::cout << "Backward: fc1.backward" << std::endl;
  Tensor dZ_flat = fc1.backward(dZ_fc1);
  std::cout << "Backward: flattenBackward" << std::endl;
  Tensor dZ_pool = Conv2d::flattenBackward(dZ_flat, Z_pool);

  std::cout << "Backward: maxPool2dBackward" << std::endl;
  Tensor dA_conv = Conv2d::maxPool2dBackward(dZ_pool, A_conv, 2, 2);
  std::cout << "Backward: reluBackward(Z_conv)" << std::endl;
  Tensor dZ_conv = Conv2d::reluBackward(Z_conv, dA_conv);

  Tensor dX_img, dW_conv, db_conv;
  dW_conv = Tensor({8, 1, 3, 3});
  db_conv = Tensor({8});
  dX_img = Tensor({B, 1, 28, 28});

  std::cout << "Backward: conv2dBackward" << std::endl;
  conv1.conv2dBackward(X_img, dZ_conv, dX_img, dW_conv, db_conv);
  Debug::stats(dLogits, "dLogits");
  Debug::stats(fc2.dW, "fc2 dW");
  Debug::stats(fc1.dW, "fc1 dW");
  Debug::stats(dW_conv, "conv dW");
  Debug::stats(db_conv, "conv db");

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Total time: " << duration.count() / 1000.0 << "s\n";
  Conv2d::testSoftmaxCrossEntropyBackwardPerfectPrediction();
  return 0;
}