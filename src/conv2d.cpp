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
