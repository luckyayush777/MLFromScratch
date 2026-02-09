#include "layer.h"

#include <cmath>
#include <random>

Layer::Layer(size_t in, size_t out, std::mt19937 &rng)
    : W({in, out}), b({out}), dW({in, out}), db({out}) {

  double std = std::sqrt(2.0 / in);
  std::normal_distribution<double> dist(0.0, std);
  for (size_t i = 0; i < W.noOfElements(); ++i)
    W.flat(i) = dist(rng);
  for (size_t i = 0; i < b.noOfElements(); ++i)
    b.flat(i) = 0.0;
}

Tensor Layer::forward(const Tensor &X) {
  X_cache = X;
  return Tensor::linearForward(X, W, b);
}

Tensor Layer::backward(const Tensor &dY) {
  Tensor dX({X_cache.dim(0), X_cache.dim(1)});
  Tensor::linearBackward(X_cache, dY, dW, db);
  dX = Tensor::matmul(dY, Tensor::transpose(W));
  return dX;
}

void Layer::step(double learningRate) {
  for (size_t i = 0; i < W.noOfElements(); ++i)
    W.flat(i) -= learningRate * dW.flat(i);
  for (size_t i = 0; i < b.noOfElements(); ++i)
    b.flat(i) -= learningRate * db.flat(i);
}

Tensor Layer::forwardNoCache(const Tensor &X) {
  return Tensor::linearForward(X, W, b);
}
