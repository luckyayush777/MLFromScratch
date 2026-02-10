#include "conv2d.h"

#include <limits>
#include <stdexcept>

Tensor Conv2d::conv2dForward(const Conv2d &conv, const Tensor &input) {
  size_t batchSize = input.dim(0);
  size_t inputChannels = input.dim(1);
  size_t inputHeight = input.dim(2);
  size_t inputWidth = input.dim(3);

  // Safety check for negative dimensions
  if (inputHeight + 2 * conv.padding < (size_t)conv.kernel)
    throw std::runtime_error("Kernel larger than padded input");

  size_t outputHeight =
      (inputHeight + 2 * conv.padding - conv.kernel) / conv.stride + 1;
  size_t outputWidth =
      (inputWidth + 2 * conv.padding - conv.kernel) / conv.stride + 1;

  Tensor out({batchSize, static_cast<size_t>(conv.outChannels), outputHeight,
              outputWidth});

  for (size_t batch = 0; batch < batchSize; ++batch) {
    for (size_t outChannel = 0; outChannel < (size_t)conv.outChannels;
         ++outChannel) {
      for (size_t outputY = 0; outputY < outputHeight; ++outputY) {
        for (size_t outputX = 0; outputX < outputWidth; ++outputX) {
          double sum = 0.0;
          for (size_t inChannel = 0; inChannel < inputChannels; ++inChannel) {
            for (size_t kernelY = 0; kernelY < (size_t)conv.kernel; ++kernelY) {
              for (size_t kernelX = 0; kernelX < (size_t)conv.kernel;
                   ++kernelX) {

                const int inputY = static_cast<int>(outputY * conv.stride +
                                                    kernelY - conv.padding);
                const int inputX = static_cast<int>(outputX * conv.stride +
                                                    kernelX - conv.padding);

                const double inputVal = conv.getPaddedInput(
                    input, batch, inChannel, inputY, inputX);

                // Use 4D accessor for weights
                const double weightVal =
                    conv.W.at(outChannel, inChannel, kernelY, kernelX);
                sum += inputVal * weightVal;
              }
            }
          }
          sum += conv.b.flat(outChannel);
          out.at(batch, outChannel, outputY, outputX) = sum;
        }
      }
    }
  }
  return out;
}

double Conv2d::getPaddedInput(const Tensor &X, size_t batch, size_t channel,
                              int h, int w) const {
  const size_t height = X.dim(2);
  const size_t width = X.dim(3);

  if (h < 0 || w < 0 || h >= (int)height || w >= (int)width) {
    return 0.0; // zero padding
  }
  return X.at(batch, channel, h, w);
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

          double maxVal = -std::numeric_limits<double>::infinity();

          for (size_t py = 0; py < poolSize; ++py) {
            for (size_t px = 0; px < poolSize; ++px) {

              size_t inY = oy * stride + py;
              size_t inX = ox * stride + px;

              double val = input.at(b, c, inY, inX);
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
    double maxLogit = -std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < C; ++j)
      maxLogit = std::max(maxLogit, logits.at(i, j));

    double sumExp = 0.0;
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
    if(y >= C) {
      throw std::runtime_error("Target class index out of bounds");
    }
    probs.at(i, y) -= 1.0;
  }

  // Normalize by batch
  for (size_t i = 0; i < B; ++i)
    for (size_t j = 0; j < C; ++j)
      probs.at(i, j) /= static_cast<double>(B);

  return probs;
}


double Conv2d::softmaxCrossEntropyLoss(const Tensor &logits,
                                      const Tensor &targets) {
  size_t B = logits.dim(0);
  size_t C = logits.dim(1);

  double loss = 0.0;

  for(size_t i = 0; i < B; ++i){
    //find max logit for numerical stability
    double maxLogit = -std::numeric_limits<double>::infinity();
    for(size_t j = 0; j < C; ++j){
      maxLogit = std::max(maxLogit, logits.at(i, j));
    }
    double sumExp = 0.0;
    for(size_t j = 0; j < C; ++j){
      sumExp += std::exp(logits.at(i, j) - maxLogit);
    }
    size_t y = static_cast<size_t>(targets.flat(i));
    if(y >= C) {
      throw std::runtime_error("Target class index out of bounds");
    }
    loss -= std::log(std::exp(logits.at(i, y) - maxLogit) / sumExp);
  }
  return loss / static_cast<double>(B);
}

void Conv2d::testSoftmaxCrossEntropyBackwardPerfectPrediction() {
    Tensor logits({1, 3});
    logits.at(0, 0) = 10.0;
    logits.at(0, 1) = -10.0;
    logits.at(0, 2) = -10.0;

    Tensor targets({1});
    targets.flat(0) = 0;  // correct class

 
    Tensor grad = Conv2d::softmaxCrossEntropyBackward(logits, targets);

    const double eps = 1e-6;

    for (size_t j = 0; j < 3; ++j) {
        double g = grad.at(0, j);
        if (std::abs(g) > eps) {
            std::cerr << "FAILED: expected gradient 0, got "
                      << g << " at class " << j << std::endl;
            std::exit(1);
        }
    }

    std::cout << "PASSED: softmaxCrossEntropyBackward perfect prediction\n";
}

Tensor Conv2d::reluBackward(const Tensor &Z, const Tensor &dA) {
  if (Z.dim(0) != dA.dim(0) || Z.dim(1) != dA.dim(1) || Z.dim(2) != dA.dim(2) || Z.dim(3) != dA.dim(3)) {
    throw std::runtime_error("Dimension mismatch in reluBackward");
  }

  Tensor dZ(Z.shape());
  for (size_t i = 0; i < Z.noOfElements(); ++i) {
    dZ.flat(i) = (Z.flat(i) > 0.0) ? dA.flat(i) : 0.0;
  }
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

  Tensor dInput(inputShape.shape());
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

  if (dOut.dim(0) != batchSize || dOut.dim(1) != channels || dOut.dim(2) != outHeight || dOut.dim(3) != outWidth) {
    throw std::runtime_error("Dimension mismatch in maxPool2dBackward");
  }

  Tensor dInput({batchSize, channels, height, width});
  Tensor::zeroTensor(dInput);
  for (size_t b = 0; b < batchSize; ++b) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t oy = 0; oy < outHeight; ++oy) {
        for (size_t ox = 0; ox < outWidth; ++ox) {

          double maxVal = -std::numeric_limits<double>::infinity();
          size_t maxY = 0, maxX = 0;

          for (size_t py = 0; py < poolSize; ++py) {
            for (size_t px = 0; px < poolSize; ++px) {

              size_t inY = oy * stride + py;
              size_t inX = ox * stride + px;

              double val = input.at(b, c, inY, inX);
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


void Conv2d::conv2dBackward(
    const Tensor &input,
    const Tensor &dOut,
    Tensor &dInput,
    Tensor &dW,
    Tensor &db
) {
  size_t batchSize = input.dim(0);
  size_t inputChannels = input.dim(1);
  size_t inputHeight = input.dim(2);
  size_t inputWidth = input.dim(3);

  size_t outChannels = dOut.dim(1);
  size_t outputHeight = dOut.dim(2);
  size_t outputWidth = dOut.dim(3);

  // Zero gradients
  Tensor::zeroTensor(dInput);
  Tensor::zeroTensor(dW);
  Tensor::zeroTensor(db);

  for (size_t batch = 0; batch < batchSize; ++batch) {
    for (size_t outChannel = 0; outChannel < outChannels; ++outChannel) {

      for (size_t outputY = 0; outputY < outputHeight; ++outputY) {
        for (size_t outputX = 0; outputX < outputWidth; ++outputX) {

          double gradOut =
              dOut.at(batch, outChannel, outputY, outputX);

          // ---- bias gradient ----
          db.flat(outChannel) += gradOut;

          for (size_t inChannel = 0; inChannel < inputChannels; ++inChannel) {
            for (size_t kernelY = 0; kernelY < (size_t)kernel; ++kernelY) {
              for (size_t kernelX = 0; kernelX < (size_t)kernel; ++kernelX) {

                int inputY = static_cast<int>(
                    outputY * stride + kernelY - padding);
                int inputX = static_cast<int>(
                    outputX * stride + kernelX - padding);

                // Only propagate if inside input bounds
                if (inputY >= 0 && inputY < (int)inputHeight &&
                    inputX >= 0 && inputX < (int)inputWidth) {

                  double inputVal =
                      input.at(batch, inChannel, inputY, inputX);

                  double weightVal =
                      W.at(outChannel, inChannel, kernelY, kernelX);

                  // ---- weight gradient ----
                  dW.at(outChannel, inChannel, kernelY, kernelX) +=
                      inputVal * gradOut;

                  // ---- input gradient ----
                  dInput.at(batch, inChannel, inputY, inputX) +=
                      weightVal * gradOut;
                }
              }
            }
          }
        }
      }
    }
  }
}
