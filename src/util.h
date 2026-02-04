#pragma once

class Tensor;







void dump_png(const Tensor &images, size_t index, const char *filename) {
  constexpr int W = 28;
  constexpr int H = 28;

  unsigned char buffer[W * H];

  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      double v = images.at(index, i * W + j);
      buffer[i * W + j] = static_cast<unsigned char>(v * 255.0);
    }
  }

  stbi_write_png(filename, W, H, 1, buffer, W);
}


void gradientDescent(const Matrix &x, const Matrix &y, double &slope,
                     double &intercept, double learningRate,
                     int maxIterations) {
  int numSamples = y.getNumRows();
  double prevLoss = std::numeric_limits<double>::infinity();
  const double threshold = 1e-6; // common-sense convergence threshold

  for (int iter = 0; iter < maxIterations; ++iter) {
    double slopeGradient = 0.0;
    double interceptGradient = 0.0;
    double loss = 0.0;

    for (int i = 0; i < numSamples; ++i) {
      double prediction = slope * x(i, 0) + intercept;
      double error = prediction - y(i, 0);

      loss += error * error;
      slopeGradient += error * x(i, 0);
      interceptGradient += error;
    }

    loss /= numSamples;
    slopeGradient *= (2.0 / numSamples);
    interceptGradient *= (2.0 / numSamples);

    if (std::abs(prevLoss - loss) < threshold) {
      std::cout << "Converged at iteration " << iter << "\n";
      break;
    }

    prevLoss = loss;
    slope -= learningRate * slopeGradient;
    intercept -= learningRate * interceptGradient;
  }
}


void gradientDescentMatricesOnly(const Matrix &x, const Matrix &y,
                                 double learningRate, int maxIterations) {
  int numSamples = y.getNumRows();
  double prevLoss = std::numeric_limits<double>::infinity();
  const double threshold = 1e-6;
  Matrix weights(2, 1);
  weights(0, 0) = 0.0;
  weights(1, 0) = 0.0; // weights(0,0) = slope, weights(1,0) = intercept

  Matrix gradients(2, 1);
  for (int iter = 0; iter < maxIterations; ++iter) {
    gradients(0, 0) = 0.0;
    gradients(1, 0) = 0.0;
    double loss = 0.0;

    for (int i = 0; i < numSamples; ++i) {
      double prediction = weights(0, 0) * x(i, 0) + weights(1, 0);
      double error = prediction - y(i, 0);

      loss += error * error;
      gradients(0, 0) += error * x(i, 0);
      gradients(1, 0) += error;
    }
    loss /= numSamples;
    gradients(0, 0) *= (2.0 / numSamples);
    gradients(1, 0) *= (2.0 / numSamples);
    if (std::abs(prevLoss - loss) < threshold) {
      std::cout << "Converged at iteration " << iter << "\n";
      break;
    }
    prevLoss = loss;
    weights(0, 0) -= learningRate * gradients(0, 0);
    weights(1, 0) -= learningRate * gradients(1, 0);
  }
  std::cout << "Slope: " << weights(0, 0) << ", Intercept: " << weights(1, 0)
            << std::endl;
}


void gradientDescentVectorized(const Matrix &x, const Matrix &y, double &slope,
                               double &intercept, double learningRate,
                               int maxIterations) {
  int numSamples = y.getNumRows();
  double prevLoss = std::numeric_limits<double>::infinity();
  Matrix weights(2, 1);
  weights(0, 0) = slope;
  weights(1, 0) = intercept;
  const double threshold = 1e-6;
  for (int iter = 0; iter < maxIterations; ++iter) {
    Matrix predictions = x * weights;
    Matrix errors = predictions - y;
    double loss = errors.dot(errors) / numSamples;

    Matrix gradients = x.transpose() * errors;
    gradients = gradients * (2.0 / numSamples);
    weights = weights - gradients * learningRate;
    if (std::abs(prevLoss - loss) < threshold) {
      std::cout << "Converged at iteration " << iter << "\n";
      break;
    }

    prevLoss = loss;
  }
  slope = weights(0, 0);
  intercept = weights(1, 0);
  std::cout << "Slope: " << slope << ", Intercept: " << intercept << std::endl;
}




Matrix relu(const Matrix &z) {
  int numRows = z.getNumRows();
  int numCols = z.getNumCols();
  Matrix output(numRows, numCols);
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      output(i, j) = std::max(0.0, z(i, j));
    }
  }
  return output;
}


// Forward declarations (defined later in this header)
void forwardPropagation(const Matrix &X, const Matrix &W1, const Matrix &W2,
                        Matrix &Z1, Matrix &A1, Matrix &A2);
double computeMSE(const Matrix &predictions, const Matrix &targets);
void backwardPropagation(const Matrix &X, const Matrix &Y, Matrix &W1,
                         Matrix &W2, const Matrix &Z1, const Matrix &A1,
                         const Matrix &A2, double learningRate, Matrix &dW1,
                         Matrix &dW2);
Matrix reluBackward(const Matrix &dA, const Matrix &Z) {
  Matrix dZ(Z.getNumRows(), Z.getNumCols());

  for (int i = 0; i < Z.getNumRows(); ++i) {
    for (int j = 0; j < Z.getNumCols(); ++j) {
      dZ(i, j) = (Z(i, j) > 0.0) ? dA(i, j) : 0.0;
    }
  }
  return dZ;
}
Matrix mseBackward(const Matrix &predictions, const Matrix &targets){
    int m = predictions.getNumRows();
    Matrix dA(predictions.getNumRows(), predictions.getNumCols());
    
    for (int i = 0; i < predictions.getNumRows(); ++i) {
        for (int j = 0; j < predictions.getNumCols(); ++j) {
        dA(i, j) = (2.0 / m) * (predictions(i, j) - targets(i, j));
        }
    }
    return dA;
}



size_t argmaxRow(const Tensor &logits, size_t row) {
  size_t best = 0;
  double bestVal = logits.at(row, 0);

  for (size_t j = 1; j < logits.dim(1); ++j) {
    double val = logits.at(row, j);
    if (val > bestVal) {
      bestVal = val;
      best = j;
    }
  }
  return best;
}

double computeAccuracy(const Tensor &logits, const Tensor &labels) {
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


void forwardPropagation(const Matrix &X, const Matrix &W1, const Matrix &W2,
                        Matrix &Z1, Matrix &A1, Matrix &A2) {
  Z1 = X * W1;
  A1 = relu(Z1);
  A2 = A1 * W2;
  std::cout << "Forward pass complete\n";
}

void sgdUpdate(Matrix &W, const Matrix &dW, double lr) {
  for (int i = 0; i < W.getNumRows(); ++i) {
    for (int j = 0; j < W.getNumCols(); ++j) {
      W(i, j) -= lr * dW(i, j);
    }
  }
}

double computeMSE(const Matrix &predictions, const Matrix &targets) {
  int m = predictions.getNumRows();
  double mse = 0.0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < predictions.getNumCols(); ++j) {
      double error = predictions(i, j) - targets(i, j);
      mse += error * error;
    }
  }
  return mse / m;
}

void backwardPropagation(const Matrix &X, const Matrix &Y, Matrix &W1,
                         Matrix &W2, const Matrix &Z1, const Matrix &A1,
                         const Matrix &A2, double learningRate, Matrix &dW1,
                         Matrix &dW2) {
  // dL/dA2
  Matrix dA2 = mseBackward(A2, Y);

  // dL/dW2 = A1^T * dA2
  dW2 = A1.transpose() * dA2;

  // dL/dA1 = dA2 * W2^T
  Matrix dA1 = dA2 * W2.transpose();

  Matrix dZ1 = reluBackward(dA1, Z1);

  // dL/dW1 = X^T * dZ1
  dW1 = X.transpose() * dZ1;

  sgdUpdate(W1, dW1, learningRate);
  sgdUpdate(W2, dW2, learningRate);
  std::cout << "Backward pass complete\n";
}
