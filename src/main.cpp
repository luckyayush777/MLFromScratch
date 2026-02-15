#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "conv2d.h"
#include "debug.h"
#include "layer.h"
#include "matrix.h"
#include "mnist.h"
#include "stb_image_write.h"
#include "tensor.h"
#include "training_config.h"
#include "training_run_logging.h"
#include "util.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

int main() {
  omp_set_num_threads(4);

  //config has defaults for hyperparameters 
  TrainingConfig config;


  std::cout << "Loading MNIST Data...\n";
  auto dataset = loadMnist("datasets/train-images-idx3-ubyte",
                           "datasets/train-labels-idx1-ubyte");
  auto testDataset = loadMnist("datasets/t10k-images-idx3-ubyte",
                               "datasets/t10k-labels-idx1-ubyte");

  //size_t trainSize = dataset.labels.noOfElements();
  //this is just to test the code on a smaller subset of the data, to speed up debugging
  size_t trainSize = 10000; // testDataset.labels.noOfElements();
  std::vector<size_t> indices(trainSize);
  for (size_t i = 0; i < trainSize; ++i)
    indices[i] = i;

  const size_t B = config.batchSize;
  constexpr size_t D = 28 * 28; // input dimension (flat)
  constexpr size_t C = 10;      // number of classes

  Tensor X({B, D});
  Tensor y({B});

  loadBatch(dataset, indices, X, y, 0);

  std::mt19937 rng(config.seed);

  constexpr size_t conv1InCh = 1;
  constexpr size_t conv1OutCh = 16;
  constexpr size_t conv1Kernel = 3;
  constexpr size_t conv1Stride = 1;
  constexpr size_t conv1Padding = 1;
  Conv2d conv1(conv1InCh, conv1OutCh, conv1Kernel, conv1Stride, conv1Padding);

  constexpr size_t flattenedDim = 16 * 14 * 14;
  constexpr size_t hiddenDim = 256;

  Layer fc1(flattenedDim, hiddenDim, rng);
  Layer fc2(hiddenDim, C, rng);

  Tensor xImg({B, 1, 28, 28});
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < 28; ++h) {
      for (size_t w = 0; w < 28; ++w) {

        double pixel = X.at(b, h * 28 + w);

        xImg.at(b, 0, h, w) = pixel;
      }
    }
  }
  printShape("Input Image", xImg);
  // conv1.overfitSingleBatch(fc1, fc2, xImg, y, learningRate, 200);

  std::optional<TrainingRunLoggingSession> logger;
  if (config.enableRunLogging) {
    logger.emplace(config.optimizationTitle, conv1, fc1, fc2, C,
                   dataset.labels.noOfElements(), testDataset.labels.noOfElements(),
                   config.learningRate, config.beta, config.batchSize, config.epochs,
                   config.seed);
  }

  conv1.trainMNIST(fc1, fc2, dataset, testDataset, config.learningRate,
                   config.batchSize, config.epochs, config.beta,
                   logger ? &logger->summary() : nullptr);

  return 0;
}