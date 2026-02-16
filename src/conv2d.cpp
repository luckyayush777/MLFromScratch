#include "conv2d.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>

Tensor Conv2d::conv2dForward(const Conv2d &convLayer,
                             const Tensor &inputTensor) {
  // Input dimensions
  const size_t batchSize = inputTensor.dim(0);
  const size_t inputChannels = inputTensor.dim(1);
  const size_t inputHeight = inputTensor.dim(2);
  const size_t inputWidth = inputTensor.dim(3);

  // Convolution parameters
  const size_t outputChannels = convLayer.outChannels;
  const size_t kernelSize = convLayer.kernel;
  const size_t stride = convLayer.stride;
  const size_t padding = convLayer.padding;

  // Output spatial dimensions
  const size_t outputHeight =
      (inputHeight + 2 * padding - kernelSize) / stride + 1;

  const size_t outputWidth =
      (inputWidth + 2 * padding - kernelSize) / stride + 1;

  Tensor outputTensor({batchSize, outputChannels, outputHeight, outputWidth});

  // Raw data pointers
  const float *inputData = inputTensor.raw();
  const float *weightData = convLayer.W.raw();
  const float *biasData = convLayer.b.raw();
  float *outputData = outputTensor.raw();

  // Precomputed strides (for index calculation)
  const size_t inputImageSize = inputChannels * inputHeight * inputWidth;
  const size_t outputImageSize = outputChannels * outputHeight * outputWidth;
  const size_t weightKernelSizePerOutput =
      inputChannels * kernelSize * kernelSize;

  // Main convolution loops
  for (size_t batchIndex = 0; batchIndex < batchSize; ++batchIndex) {

    const size_t inputBatchOffset = batchIndex * inputImageSize;
    const size_t outputBatchOffset = batchIndex * outputImageSize;

    for (size_t outputChannel = 0; outputChannel < outputChannels;
         ++outputChannel) {

      const size_t weightOutputChannelOffset =
          outputChannel * weightKernelSizePerOutput;

      for (size_t outputY = 0; outputY < outputHeight; ++outputY) {
        for (size_t outputX = 0; outputX < outputWidth; ++outputX) {

          float accumulatedSum = biasData[outputChannel];

          // Loop over input channels
          for (size_t inputChannel = 0; inputChannel < inputChannels;
               ++inputChannel) {

            const size_t inputChannelOffset =
                inputBatchOffset + inputChannel * inputHeight * inputWidth;

            const size_t weightInputChannelOffset =
                weightOutputChannelOffset +
                inputChannel * kernelSize * kernelSize;

            // Loop over kernel window
            for (size_t kernelY = 0; kernelY < kernelSize; ++kernelY) {

              const int inputY = static_cast<int>(outputY * stride + kernelY) -
                                 static_cast<int>(padding);

              if (inputY < 0 || inputY >= static_cast<int>(inputHeight))
                continue;

              for (size_t kernelX = 0; kernelX < kernelSize; ++kernelX) {

                const int inputX =
                    static_cast<int>(outputX * stride + kernelX) -
                    static_cast<int>(padding);

                if (inputX < 0 || inputX >= static_cast<int>(inputWidth))
                  continue;

                const size_t inputIndex =
                    inputChannelOffset +
                    static_cast<size_t>(inputY) * inputWidth +
                    static_cast<size_t>(inputX);

                const size_t weightIndex =
                    weightInputChannelOffset + kernelY * kernelSize + kernelX;

                accumulatedSum +=
                    inputData[inputIndex] * weightData[weightIndex];
              }
            }
          }

          const size_t outputIndex =
              outputBatchOffset + outputChannel * outputHeight * outputWidth +
              outputY * outputWidth + outputX;

          outputData[outputIndex] = accumulatedSum;
        }
      }
    }
  }

  return outputTensor;
}

float Conv2d::getPaddedInput(const Tensor &X, size_t batch, size_t channel,
                             int h, int w) const {
  const size_t height = X.dim(2);
  const size_t width = X.dim(3);

  if (h < 0 || w < 0 || h >= (int)height || w >= (int)width) {
    return 0.0f; // zero padding
  }
  return X.at(batch, channel, static_cast<size_t>(h), static_cast<size_t>(w));
}

Tensor Conv2d::flattenForward(const Tensor &input) {
  size_t batchSize = input.dim(0);
  size_t channels = input.dim(1);
  size_t height = input.dim(2);
  size_t width = input.dim(3);

  Tensor out({batchSize, channels * height * width});
  for (size_t batch = 0; batch < batchSize; ++batch) {
    for (size_t channel = 0; channel < channels; ++channel) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          size_t outIndex = channel * (height * width) + h * width + w;
          out.at(batch, outIndex) = input.at(batch, channel, h, w);
        }
      }
    }
  }
  return out;
}

Tensor Conv2d::maxPool2dForward(const Tensor &input, size_t poolSize,
                                size_t stride) {
  // Input shape
  size_t batchSize = input.dim(0);
  size_t channels = input.dim(1);
  size_t height = input.dim(2);
  size_t width = input.dim(3);

  size_t outHeight = (height - poolSize) / stride + 1;
  size_t outWidth = (width - poolSize) / stride + 1;

  Tensor output({batchSize, channels, outHeight, outWidth});

  for (size_t b = 0; b < batchSize; ++b) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t oy = 0; oy < outHeight; ++oy) {
        for (size_t ox = 0; ox < outWidth; ++ox) {

          float maxVal = -std::numeric_limits<float>::infinity();

          for (size_t py = 0; py < poolSize; ++py) {
            for (size_t px = 0; px < poolSize; ++px) {

              size_t inY = oy * stride + py;
              size_t inX = ox * stride + px;

              float val = input.at(b, c, inY, inX);
              if (val > maxVal)
                maxVal = val;
            }
          }

          output.at(b, c, oy, ox) = maxVal;
        }
      }
    }
  }

  return output;
}

Tensor Conv2d::softmaxCrossEntropyBackward(const Tensor &logits,
                                           const Tensor &targets) {
  size_t B = logits.dim(0);
  size_t C = logits.dim(1);

  Tensor probs({B, C});

  // Softmax
  for (size_t i = 0; i < B; ++i) {
    float maxLogit = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < C; ++j)
      maxLogit = std::fmax(maxLogit, logits.at(i, j));

    float sumExp = 0.0f;
    for (size_t j = 0; j < C; ++j) {
      probs.at(i, j) = std::exp(logits.at(i, j) - maxLogit);
      sumExp += probs.at(i, j);
    }

    for (size_t j = 0; j < C; ++j)
      probs.at(i, j) /= sumExp;
  }

  // Gradient
  for (size_t i = 0; i < B; ++i) {
    size_t y = static_cast<size_t>(targets.flat(i));
    if (y >= C) {
      throw std::runtime_error("Target class index out of bounds");
    }
    probs.at(i, y) -= 1.0f;
  }

  // Normalize by batch
  for (size_t i = 0; i < B; ++i)
    for (size_t j = 0; j < C; ++j)
      probs.at(i, j) /= static_cast<float>(B);

  return probs;
}

double Conv2d::softmaxCrossEntropyLoss(const Tensor &logits,
                                       const Tensor &targets) {
  size_t B = logits.dim(0);
  size_t C = logits.dim(1);

  double loss = 0.0;

  for (size_t i = 0; i < B; ++i) {
    // find max logit for numerical stability
    float maxLogit = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < C; ++j) {
      maxLogit = std::fmax(maxLogit, logits.at(i, j));
    }
    float sumExp = 0.0f;
    for (size_t j = 0; j < C; ++j) {
      sumExp += std::exp(logits.at(i, j) - maxLogit);
    }
    size_t y = static_cast<size_t>(targets.flat(i));
    if (y >= C) {
      throw std::runtime_error("Target class index out of bounds");
    }
    loss -= std::log(std::exp(logits.at(i, y) - maxLogit) / sumExp);
  }
  return loss / static_cast<double>(B);
}

void Conv2d::testSoftmaxCrossEntropyBackwardPerfectPrediction() {
  Tensor logits({1, 3});
  logits.at(0, 0) = 10.0f;
  logits.at(0, 1) = -10.0f;
  logits.at(0, 2) = -10.0f;

  Tensor targets({1});
  targets.flat(0) = 0; // correct class

  Tensor grad = Conv2d::softmaxCrossEntropyBackward(logits, targets);

  const float eps = 1e-6f;

  for (size_t j = 0; j < 3; ++j) {
    float g = grad.at(0, j);
    if (std::abs(g) > eps) {
      std::cerr << "FAILED: expected gradient 0, got " << g << " at class " << j
                << std::endl;
      std::exit(1);
    }
  }

  std::cout << "PASSED: softmaxCrossEntropyBackward perfect prediction\n";
}

Tensor reluBackward(const Tensor &Z, const Tensor &dA) {
  size_t N = Z.noOfElements();
  Tensor dZ(Z.getShape());

  const float *z = Z.raw();
  const float *da = dA.raw();
  float *dz = dZ.raw();

  for (size_t i = 0; i < N; ++i)
    dz[i] = z[i] > 0.0f ? da[i] : 0.0f;

  return dZ;
}

Tensor Conv2d::flattenBackward(const Tensor &dOut, const Tensor &inputShape) {
  size_t batchSize = inputShape.dim(0);
  size_t channels = inputShape.dim(1);
  size_t height = inputShape.dim(2);
  size_t width = inputShape.dim(3);

  if (dOut.dim(0) != batchSize || dOut.dim(1) != channels * height * width) {
    throw std::runtime_error("Dimension mismatch in flattenBackward");
  }

  Tensor dInput(inputShape.getShape());
  for (size_t batch = 0; batch < batchSize; ++batch) {
    for (size_t channel = 0; channel < channels; ++channel) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          size_t outIndex = channel * (height * width) + h * width + w;
          dInput.at(batch, channel, h, w) = dOut.at(batch, outIndex);
        }
      }
    }
  }
  return dInput;
}

Tensor Conv2d::maxPool2dBackward(const Tensor &dOut, const Tensor &input,
                                 size_t poolSize, size_t stride) {
  size_t batchSize = input.dim(0);
  size_t channels = input.dim(1);
  size_t height = input.dim(2);
  size_t width = input.dim(3);

  size_t outHeight = (height - poolSize) / stride + 1;
  size_t outWidth = (width - poolSize) / stride + 1;

  if (dOut.dim(0) != batchSize || dOut.dim(1) != channels ||
      dOut.dim(2) != outHeight || dOut.dim(3) != outWidth) {
    throw std::runtime_error("Dimension mismatch in maxPool2dBackward");
  }

  Tensor dInput({batchSize, channels, height, width});
  Tensor::zeroTensor(dInput);
  for (size_t b = 0; b < batchSize; ++b) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t oy = 0; oy < outHeight; ++oy) {
        for (size_t ox = 0; ox < outWidth; ++ox) {

          float maxVal = -std::numeric_limits<float>::infinity();
          size_t maxY = 0, maxX = 0;

          for (size_t py = 0; py < poolSize; ++py) {
            for (size_t px = 0; px < poolSize; ++px) {

              size_t inY = oy * stride + py;
              size_t inX = ox * stride + px;

              float val = input.at(b, c, inY, inX);
              if (val > maxVal) {
                maxVal = val;
                maxY = inY;
                maxX = inX;
              }
            }
          }

          dInput.at(b, c, maxY, maxX) += dOut.at(b, c, oy, ox);
        }
      }
    }
  }
  return dInput;
}

void Conv2d::conv2dBackward(const Tensor &inputTensor,
                            const Tensor &dOutputTensor, Tensor &dInputTensor,
                            Tensor &dWeightTensor, Tensor &dBiasTensor) {
  // Dimensions
  const size_t batchSize = inputTensor.dim(0);
  const size_t inputChannels = inputTensor.dim(1);
  const size_t inputHeight = inputTensor.dim(2);
  const size_t inputWidth = inputTensor.dim(3);

  const size_t outputChannels = dOutputTensor.dim(1);
  const size_t outputHeight = dOutputTensor.dim(2);
  const size_t outputWidth = dOutputTensor.dim(3);

  const size_t kernelSize = kernel;
  const size_t strideSize = stride;
  const size_t paddingSize = padding;

  // Zero gradients
  Tensor::zeroTensor(dInputTensor);
  Tensor::zeroTensor(dWeightTensor);
  Tensor::zeroTensor(dBiasTensor);

  // Raw pointers for faster access
  const float *inputData = inputTensor.raw();
  const float *dOutputData = dOutputTensor.raw();
  const float *weightData = W.raw();

  float *dInputData = dInputTensor.raw();
  float *dWeightData = dWeightTensor.raw();
  float *dBiasData = dBiasTensor.raw();

  // Precomputed sizes
  const size_t inputImageSize = inputChannels * inputHeight * inputWidth;
  const size_t outputImageSize = outputChannels * outputHeight * outputWidth;
  const size_t weightKernelSizePerOutput =
      inputChannels * kernelSize * kernelSize;

#pragma omp parallel
  {
    Tensor localDWeight({outChannels, inChannels, kernelSize, kernelSize});
    Tensor localDBias({outChannels});
    Tensor::zeroTensor(localDWeight);
    Tensor::zeroTensor(localDBias);

    float *dWeightLocalData = localDWeight.raw();
    float *dBiasLocalData = localDBias.raw();

    const int batchSizeSigned = static_cast<int>(batchSize);
#pragma omp for
    for (int batchIdx = 0; batchIdx < batchSizeSigned; ++batchIdx) {
      const size_t batchIndex = static_cast<size_t>(batchIdx);
      const size_t inputBatchOffset = batchIndex * inputImageSize;
      const size_t outputBatchOffset = batchIndex * outputImageSize;

      for (size_t outputChannel = 0; outputChannel < outputChannels;
           ++outputChannel) {
        const size_t weightOutputOffset =
            outputChannel * weightKernelSizePerOutput;

        for (size_t outputY = 0; outputY < outputHeight; ++outputY) {
          for (size_t outputX = 0; outputX < outputWidth; ++outputX) {
            const size_t dOutputIndex =
                outputBatchOffset + outputChannel * outputHeight * outputWidth +
                outputY * outputWidth + outputX;

            const float gradOutputValue = dOutputData[dOutputIndex];

            // Bias gradient (thread-local accumulation)
            dBiasLocalData[outputChannel] += gradOutputValue;

            for (size_t inputChannel = 0; inputChannel < inputChannels;
                 ++inputChannel) {
              const size_t inputChannelOffset =
                  inputBatchOffset + inputChannel * inputHeight * inputWidth;

              const size_t weightInputOffset =
                  weightOutputOffset + inputChannel * kernelSize * kernelSize;

              for (size_t kernelY = 0; kernelY < kernelSize; ++kernelY) {
                const int inputY =
                    static_cast<int>(outputY * strideSize + kernelY) -
                    static_cast<int>(paddingSize);

                if (inputY < 0 || inputY >= static_cast<int>(inputHeight))
                  continue;

                for (size_t kernelX = 0; kernelX < kernelSize; ++kernelX) {
                  const int inputX =
                      static_cast<int>(outputX * strideSize + kernelX) -
                      static_cast<int>(paddingSize);

                  if (inputX < 0 || inputX >= static_cast<int>(inputWidth))
                    continue;

                  const size_t inputIndex =
                      inputChannelOffset +
                      static_cast<size_t>(inputY) * inputWidth +
                      static_cast<size_t>(inputX);

                  const size_t weightIndex =
                      weightInputOffset + kernelY * kernelSize + kernelX;

                  const float inputValue = inputData[inputIndex];
                  const float weightValue = weightData[weightIndex];

                  // Weight gradient (thread-local accumulation)
                  dWeightLocalData[weightIndex] += inputValue * gradOutputValue;
                  dInputData[inputIndex] += weightValue * gradOutputValue;
                }
              }
            }
          }
        }
      }
    }
#pragma omp critical
    {
      for (size_t i = 0; i < dWeightTensor.noOfElements(); ++i) {
        dWeightData[i] += dWeightLocalData[i];
      }
      for (size_t i = 0; i < dBiasTensor.noOfElements(); ++i) {
        dBiasData[i] += dBiasLocalData[i];
      }
    }
  }
}

Tensor Conv2d::reshapeToImage(const Tensor &X, size_t batchSize) {
  Tensor X_img({batchSize, 1, 28, 28});
  for (size_t bi = 0; bi < batchSize; ++bi)
    for (size_t h = 0; h < 28; ++h)
      for (size_t w = 0; w < 28; ++w)
        X_img.at(bi, 0, h, w) = X.at(bi, h * 28 + w);
  return X_img;
}

double Conv2d::trainBatch(Layer &fc1, Layer &fc2, const Tensor &X_img,
                          const Tensor &y, size_t batchSize,
                          double learningRate, double beta, size_t &correct,
                          double &fwdMs, double &bwdMs, double &updateMs,
                          BatchTimingStats &timing) {
  using clock = std::chrono::high_resolution_clock;

  // Forward
  auto t0 = clock::now();
  Tensor Z_conv = conv2dForward(*this, X_img);
  auto t1 = clock::now();
  Tensor A_conv = Z_conv;
  Tensor::relu(A_conv);
  auto t2 = clock::now();

  Tensor Z_pool = maxPool2dForward(A_conv, 2, 2);
  auto t3 = clock::now();
  Tensor Z_flat = flattenForward(Z_pool);
  auto t4 = clock::now();

  Tensor Z_fc1 = fc1.forward(Z_flat);
  auto t5 = clock::now();
  Tensor A_fc1 = Z_fc1;
  Tensor::relu(A_fc1);

  Tensor logits = fc2.forward(A_fc1);
  auto t6 = clock::now();

  double loss = softmaxCrossEntropyLoss(logits, y);
  correct = static_cast<size_t>(computeAccuracy(logits, y) * batchSize + 0.5);
  auto t7 = clock::now();

  // Backward
  Tensor dLogits = softmaxCrossEntropyBackward(logits, y);
  auto t8 = clock::now();

  Tensor dA_fc1 = fc2.backward(dLogits);
  auto t9 = clock::now();
  Tensor dZ_fc1 = reluBackward(Z_fc1, dA_fc1);
  auto t10 = clock::now();

  Tensor dZ_flat = fc1.backward(dZ_fc1);
  auto t11 = clock::now();
  Tensor dZ_pool = flattenBackward(dZ_flat, Z_pool);
  auto t12 = clock::now();

  Tensor dA_conv = maxPool2dBackward(dZ_pool, A_conv, 2, 2);
  auto t13 = clock::now();
  Tensor dZ_conv = reluBackward(Z_conv, dA_conv);
  auto t14 = clock::now();

  Tensor dX_img({batchSize, 1, 28, 28});
  Tensor dW_conv({outChannels, inChannels, kernel, kernel});
  Tensor db_conv({outChannels});

  conv2dBackward(X_img, dZ_conv, dX_img, dW_conv, db_conv);
  auto t15 = clock::now();

  // Update weights
  fc1.step(learningRate, beta);
  fc2.step(learningRate, beta);

  for (size_t i = 0; i < W.noOfElements(); ++i) {
    vW.flat(i) = static_cast<float>(beta * vW.flat(i) + dW_conv.flat(i));
    W.flat(i) -= static_cast<float>(learningRate * vW.flat(i));
  }

  for (size_t i = 0; i < b.noOfElements(); ++i) {
    vb.flat(i) = static_cast<float>(beta * vb.flat(i) + db_conv.flat(i));
    b.flat(i) -= static_cast<float>(learningRate * vb.flat(i));
  }
  auto t16 = clock::now();

  // Accumulate totals
  fwdMs += std::chrono::duration<double, std::milli>(t7 - t0).count();
  bwdMs += std::chrono::duration<double, std::milli>(t15 - t7).count();
  updateMs += std::chrono::duration<double, std::milli>(t16 - t15).count();

  // Detailed timing
  timing.conv2dFwdMs +=
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  timing.reluFwdMs +=
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  timing.poolFwdMs +=
      std::chrono::duration<double, std::milli>(t3 - t2).count();
  timing.flattenFwdMs +=
      std::chrono::duration<double, std::milli>(t4 - t3).count();
  timing.fc1FwdMs += std::chrono::duration<double, std::milli>(t5 - t4).count();
  timing.fc2FwdMs += std::chrono::duration<double, std::milli>(t6 - t5).count();
  timing.lossFwdMs +=
      std::chrono::duration<double, std::milli>(t7 - t6).count();

  timing.softmaxBwdMs +=
      std::chrono::duration<double, std::milli>(t8 - t7).count();
  timing.fc2BwdMs += std::chrono::duration<double, std::milli>(t9 - t8).count();
  timing.reluBwd1Ms +=
      std::chrono::duration<double, std::milli>(t10 - t9).count();
  timing.fc1BwdMs +=
      std::chrono::duration<double, std::milli>(t11 - t10).count();
  timing.flattenBwdMs +=
      std::chrono::duration<double, std::milli>(t12 - t11).count();
  timing.poolBwdMs +=
      std::chrono::duration<double, std::milli>(t13 - t12).count();
  timing.reluBwd2Ms +=
      std::chrono::duration<double, std::milli>(t14 - t13).count();
  timing.conv2dBwdMs +=
      std::chrono::duration<double, std::milli>(t15 - t14).count();
  timing.updateMs +=
      std::chrono::duration<double, std::milli>(t16 - t15).count();

  return loss;
}

double Conv2d::evaluateTestSet(Layer &fc1, Layer &fc2,
                               const MNISTDataset &testDataset) {
  size_t testSize = testDataset.labels.noOfElements();
  size_t testCorrect = 0;

  for (size_t i = 0; i < testSize; ++i) {
    Tensor X_test({1, 28 * 28});
    for (size_t j = 0; j < 28 * 28; ++j)
      X_test.at(0, j) = testDataset.images.at(i, j);

    Tensor X_test_img = reshapeToImage(X_test, 1);

    Tensor Z_c = conv2dForward(*this, X_test_img);
    Tensor::relu(Z_c);
    Tensor Z_p = maxPool2dForward(Z_c, 2, 2);
    Tensor Z_f = flattenForward(Z_p);
    Tensor Z_h = fc1.forwardNoCache(Z_f);
    Tensor::relu(Z_h);
    Tensor logits_test = fc2.forwardNoCache(Z_h);

    if (argmaxRow(logits_test, 0) ==
        static_cast<size_t>(testDataset.labels.flat(i)))
      ++testCorrect;
  }

  return static_cast<double>(testCorrect) / testSize;
}

void Conv2d::trainMNIST(Layer &fc1, Layer &fc2, const MNISTDataset &dataset,
                        const MNISTDataset &testDataset, double learningRate,
                        size_t batchSize, size_t epochs, double beta,
                        TrainingRunSummary *summary) {
  size_t trainSize = dataset.labels.noOfElements();
  std::vector<size_t> indices(trainSize);
  for (size_t i = 0; i < trainSize; ++i)
    indices[i] = i;

  std::mt19937 rng(42);
  std::cout << "Starting MNIST training...\n";

  double finalAvgLoss = 0.0;
  double finalTrainAcc = 0.0;
  double finalTestAcc = -1.0;

  if (summary != nullptr) {
    summary->epochStats.clear();
  }

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), rng);

    double epochLoss = 0.0;
    size_t totalCorrect = 0;
    size_t totalSamples = 0;
    size_t numBatches = 0;
    double fwdMs = 0.0, bwdMs = 0.0, updateMs = 0.0;
    BatchTimingStats timing{};

    for (size_t start = 0; start < trainSize; start += batchSize) {
      if (start + batchSize > trainSize)
        break;

      Tensor X({batchSize, 28 * 28});
      Tensor y({batchSize});
      loadBatch(dataset, indices, X, y, start);

      Tensor X_img = reshapeToImage(X, batchSize);

      size_t batchCorrect = 0;
      double loss =
          trainBatch(fc1, fc2, X_img, y, batchSize, learningRate, beta,
                     batchCorrect, fwdMs, bwdMs, updateMs, timing);

      epochLoss += loss;
      totalCorrect += batchCorrect;
      totalSamples += batchSize;
      numBatches++;
    }

    double avgLoss = epochLoss / numBatches;
    double trainAcc = static_cast<double>(totalCorrect) / totalSamples;
    std::cout << "Epoch " << epoch << " | Avg Loss: " << avgLoss
              << " | Train Acc: " << trainAcc << std::endl;
    std::cout << "  Forward: " << fwdMs / 1000.0
              << "s | Backward: " << bwdMs / 1000.0
              << "s | Update: " << updateMs / 1000.0 << "s" << std::endl;
    timing.print();

    double epochTestAcc = -1.0;
    if (testDataset.labels.noOfElements() > 0) {
      double testAcc = evaluateTestSet(fc1, fc2, testDataset);
      std::cout << "         Test Acc: " << testAcc << std::endl;
      finalTestAcc = testAcc;
      epochTestAcc = testAcc;
    }

    if (summary != nullptr) {
      EpochTrainingStats epochStats;
      epochStats.epochIndex = epoch;
      epochStats.avgLoss = avgLoss;
      epochStats.trainAcc = trainAcc;
      epochStats.forwardSeconds = fwdMs / 1000.0;
      epochStats.backwardSeconds = bwdMs / 1000.0;
      epochStats.updateSeconds = updateMs / 1000.0;
      epochStats.testAcc = epochTestAcc;
      epochStats.timing = timing;
      summary->epochStats.push_back(epochStats);
    }

    finalAvgLoss = avgLoss;
    finalTrainAcc = trainAcc;
  }

  if (summary != nullptr) {
    summary->finalAvgLoss = finalAvgLoss;
    summary->finalTrainAcc = finalTrainAcc;
    summary->finalTestAcc = finalTestAcc;
    summary->completedEpochs = epochs;
  }
}

void Conv2d::overfitSingleBatch(Layer &fc1, Layer &fc2, const Tensor &X_img,
                                const Tensor &y, double learningRate,
                                int steps) {
  size_t B = X_img.dim(0);

  std::cout << "Starting single-batch overfitting...\n";

  for (int step = 0; step < steps; ++step) {

    // FORWARD
    Tensor Z_conv = Conv2d::conv2dForward(*this, X_img);
    Tensor A_conv = Z_conv;
    Tensor::relu(A_conv);

    Tensor Z_pool = Conv2d::maxPool2dForward(A_conv, 2, 2);
    Tensor Z_flat = Conv2d::flattenForward(Z_pool);

    Tensor Z_fc1 = fc1.forward(Z_flat);
    Tensor A_fc1 = Z_fc1;
    Tensor::relu(A_fc1);

    Tensor logits = fc2.forward(A_fc1);

    double loss = Conv2d::softmaxCrossEntropyLoss(logits, y);
    Tensor dLogits = Conv2d::softmaxCrossEntropyBackward(logits, y);

    // BACKWARD
    Tensor dA_fc1 = fc2.backward(dLogits);
    Tensor dZ_fc1 = reluBackward(Z_fc1, dA_fc1);

    Tensor dZ_flat = fc1.backward(dZ_fc1);
    Tensor dZ_pool = Conv2d::flattenBackward(dZ_flat, Z_pool);

    Tensor dA_conv = Conv2d::maxPool2dBackward(dZ_pool, A_conv, 2, 2);
    Tensor dZ_conv = reluBackward(Z_conv, dA_conv);

    Tensor dX_img({B, 1, 28, 28});
    Tensor dW_conv(
        {this->outChannels, this->inChannels, this->kernel, this->kernel});
    Tensor db_conv({this->outChannels});

    this->conv2dBackward(X_img, dZ_conv, dX_img, dW_conv, db_conv);

    // UPDATE
    fc1.step(learningRate, 0.0);
    fc2.step(learningRate, 0.0);

    for (size_t i = 0; i < W.noOfElements(); ++i)
      W.flat(i) -= static_cast<float>(learningRate * dW_conv.flat(i));

    for (size_t i = 0; i < b.noOfElements(); ++i)
      b.flat(i) -= static_cast<float>(learningRate * db_conv.flat(i));

    // LOGGING
    if (step % 25 == 0) {
      double acc = computeAccuracy(logits, y);
      std::cout << "Step " << step << " | Loss: " << loss << " | Acc: " << acc
                << std::endl;
    }
  }
}
size_t Conv2d::argmaxRow(const Tensor &logits, size_t row) {
  size_t best = 0;
  float bestVal = logits.at(row, 0);

  for (size_t j = 1; j < logits.dim(1); ++j) {
    float val = logits.at(row, j);
    if (val > bestVal) {
      bestVal = val;
      best = j;
    }
  }
  return best;
}

double Conv2d::computeAccuracy(const Tensor &logits, const Tensor &labels) {
  size_t correct = 0;
  size_t total = logits.dim(0);

  for (size_t i = 0; i < total; ++i) {
    size_t predicted = argmaxRow(logits, i);
    size_t actual = static_cast<size_t>(labels.flat(i));
    if (predicted == actual) {
      ++correct;
    }
  }
  return static_cast<double>(correct) / static_cast<double>(total);
}
