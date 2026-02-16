#pragma once

#include <cmath>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <vector>

class Tensor {

private:
  std::vector<size_t> shape;
  size_t totalSize = 0;
  std::vector<double> data;

public:
  Tensor() = default;
  explicit Tensor(const std::vector<size_t> &shape_) : shape(shape_) {
    if (shape.empty()) {
      throw std::invalid_argument("Shape cannot be empty");
    }
    totalSize = 1;
    for (size_t d : shape) {
      if (d == 0) {
        throw std::invalid_argument(
            "Shape dimensions must be greater than zero");
      }
      totalSize *= d;
    }
    data.resize(totalSize, 0.0);
  }
  const std::vector<size_t> &getShape() const { return shape; }

  size_t noOfElements() const { return totalSize; }

  double *raw() { return data.data(); }

  const double *raw() const { return data.data(); }

  double &flat(size_t i) {
    if (i >= totalSize) {
      throw std::out_of_range("Tensor flat index out of range");
    }
    return data[i];
  }

  const double &flat(size_t i) const {
    if (i >= totalSize) {
      throw std::out_of_range("Tensor flat index out of range");
    }
    return data[i];
  }

  double &at(size_t i, size_t j) {
    if (shape.size() != 2) {
      throw std::logic_error("Tensor is not 2D");
    }
    size_t cols = shape[1];
    if (i >= shape[0] || j >= cols) {
      throw std::out_of_range("Tensor index out of range");
    }
    return data[i * cols + j];
  }

  const double &at(size_t i, size_t j) const {
    if (shape.size() != 2) {
      throw std::logic_error("Tensor is not 2D");
    }
    size_t cols = shape[1];
    if (i >= shape[0] || j >= cols) {
      throw std::out_of_range("Tensor index out of range");
    }
    return data[i * cols + j];
  }

  // tensor indexing for 4D tensors (for conv2d)
  double &at(size_t i, size_t j, size_t k, size_t l) {
    if (shape.size() != 4) {
      throw std::logic_error("Tensor is not 4D");
    }
    size_t channels = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];
    if (i >= shape[0] || j >= channels || k >= height || l >= width) {
      throw std::out_of_range("Tensor index out of range");
    }
    size_t index =
        channels * height * width * i + height * width * j + width * k + l;
    if (index >= totalSize) {
      throw std::out_of_range("Tensor index out of range");
    }
    return data[index];
  }

  const double &at(size_t i, size_t j, size_t k, size_t l) const {
    if (shape.size() != 4) {
      throw std::logic_error("Tensor is not 4D");
    }
    size_t channels = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];
    if (i >= shape[0] || j >= channels || k >= height || l >= width) {
      throw std::out_of_range("Tensor index out of range");
    }
    size_t index =
        channels * height * width * i + height * width * j + width * k + l;
    if (index >= totalSize) {
      throw std::out_of_range("Tensor index out of range");
    }
    return data[index];
  }

  size_t dim(size_t i) const {
    if (i >= shape.size()) {
      throw std::out_of_range("Shape index out of range");
    }
    return shape[i];
  }

  size_t ndim() const { return shape.size(); }
  static Tensor matmul(const Tensor &A, const Tensor &B) {
    if (A.ndim() != 2 || B.ndim() != 2) {
      throw std::logic_error("matmul requires 2D tensors");
    }
    if (A.dim(1) != B.dim(0)) {
      throw std::invalid_argument(
          "matmul shape mismatch: A columns must equal B rows");
    }

    size_t m = A.dim(0);
    size_t k = A.dim(1);
    size_t n = B.dim(1);

    Tensor out({m, n});

    const double *Adata = A.raw();
    const double *Bdata = B.raw();
    double *outData = out.raw();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(m); ++i) {

      double *outRow = &outData[i * n];

      for (size_t j = 0; j < n; ++j)
        outRow[j] = 0.0;

      for (size_t t = 0; t < k; ++t) {
        double a_val = Adata[i * k + t];
        const double *brow = &Bdata[t * n];

        for (size_t j = 0; j < n; ++j)
          outRow[j] += a_val * brow[j];
      }
    }

    return out;
  }

  static void addBias(Tensor &out, const Tensor &bias) {
    if (out.ndim() != 2 || bias.ndim() != 1) {
      throw std::logic_error(
          "addBias requires 2D output tensor and 1D bias tensor");
    }
    size_t m = out.dim(0);
    size_t n = out.dim(1);
    if (bias.dim(0) != n) {
      throw std::invalid_argument("addBias shape mismatch");
    }
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out.at(i, j) += bias.flat(j);
      }
    }
  }
  static void zeroTensor(Tensor &t) {
    for (size_t i = 0; i < t.noOfElements(); ++i) {
      t.flat(i) = 0.0;
    }
  }

  static Tensor linearForward(const Tensor &X, const Tensor &W,
                              const Tensor &b) {
    Tensor out = matmul(X, W);
    addBias(out, b);
    return out;
  }

  static void softmax(Tensor &Z) {
    size_t rows = Z.dim(0);
    size_t cols = Z.dim(1); // this rows and columns is purely temporary and
                            // doesnt make any sense for higher dimensions

    for (size_t i = 0; i < rows; ++i) {
      // subtract max for numerical stability
      double maxVal = Z.at(i, 0);
      for (size_t j = 0; j < cols; ++j) {
        maxVal = std::max(maxVal, Z.at(i, j));
      }
      double sumExp = 0.0;
      for (size_t j = 0; j < cols; ++j) {
        Z.at(i, j) = std::exp(Z.at(i, j) - maxVal);
        sumExp += Z.at(i, j);
      }

      for (size_t j = 0; j < cols; ++j) {
        Z.at(i, j) /= sumExp;
      }
    }
  }

  static double crossEntropyLoss(const Tensor &predictions,
                                 const Tensor &targets) {
    size_t B = predictions.dim(0);
    double loss = 0.0;

    for (size_t i = 0; i < B; i++) {
      size_t y = static_cast<size_t>(targets.flat(i));
      loss -= std::log(predictions.at(i, y) +
                       1e-15); // add small value to avoid log(0)
    }
    return loss / B;
  }

  static void softmaxCrossEntropyBackward(Tensor &predictions,
                                          const Tensor &labels) {
    size_t B = predictions.dim(0);
    // size_t C = predictions.dim(1);

    for (size_t i = 0; i < B; i++) {
      size_t y = static_cast<size_t>(labels.flat(i));
      predictions.at(i, y) -= 1.0;
    }
    // average over batch
    for (size_t i = 0; i < predictions.noOfElements(); ++i) {
      predictions.flat(i) /= B;
    }
  }

  static void linearBackward(const Tensor &X, const Tensor &dZ, Tensor &dw,
                             Tensor &db) {
    size_t B = X.dim(0);
    size_t D = X.dim(1);
    size_t C = dZ.dim(1);

    for (size_t i = 0; i < dw.noOfElements(); ++i) {
      dw.flat(i) = 0.0;
    }
    for (size_t j = 0; j < db.noOfElements(); ++j) {
      db.flat(j) = 0.0;
    }

    for (size_t i = 0; i < B; ++i) {
      for (size_t j = 0; j < D; ++j) {
        for (size_t k = 0; k < C; ++k) {
          dw.at(j, k) += X.at(i, j) * dZ.at(i, k);
        }
      }
    }

    for (size_t k = 0; k < C; ++k) {
      for (size_t i = 0; i < B; ++i) {
        db.flat(k) += dZ.at(i, k);
      }
    }
  }

  static void relu(Tensor &T) {
    for (size_t i = 0; i < T.noOfElements(); ++i)
      T.flat(i) = std::max(0.0, T.flat(i));
  }

  static Tensor transpose(const Tensor &input) {
    Tensor output({input.dim(1), input.dim(0)});
    if (input.ndim() != 2 || output.ndim() != 2) {
      throw std::logic_error("transpose requires 2D tensors");
    }
    size_t rows = input.dim(0);
    size_t cols = input.dim(1);
    if (output.dim(0) != cols || output.dim(1) != rows) {
      throw std::invalid_argument("transpose shape mismatch");
    }
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        output.at(j, i) = input.at(i, j);
      }
    }
    return output;
  }
};