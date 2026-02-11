#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "conv2d.h"
#include "debug.h"
#include "layer.h"
#include "matrix.h"
#include "mnist.h"
#include "stb_image_write.h"
#include "tensor.h"
#include "util.h"
#include "timer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>



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
  double learningRate = 0.02;
  (void)learningRate;


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
  Timer timer("Single batch overfitting");
  conv1.overfitSingleBatch(fc1, fc2, X_img, y, learningRate, 200);
  return 0;
}