#include "layer.h"

#include <cmath>
#include <random>

Layer::Layer(size_t in, size_t out, std::mt19937 &rng)
    : W({in, out}), b({out}), dW({in, out}), db({out}), vW({in, out}), vb({out}) {

  double std = std::sqrt(2.0 / in);
  std::normal_distribution<double> dist(0.0, std);
  for (size_t i = 0; i < W.noOfElements(); ++i)
    W.flat(i) = static_cast<float>(dist(rng));
  for (size_t i = 0; i < b.noOfElements(); ++i)
    b.flat(i) = 0.0f;
  for (size_t i = 0; i < vW.noOfElements(); ++i)
    vW.flat(i) = 0.0f;
  for (size_t i = 0; i < vb.noOfElements(); ++i)
    vb.flat(i) = 0.0f;
}

Tensor Layer::forward(const Tensor &X) {
  X_cache = X;
  return Tensor::linearForward(X, W, b);
}

Tensor Layer::backward(const Tensor &dY) {
  Tensor::linearBackward(X_cache, dY, dW, db);
  return Tensor::matmul(dY, Tensor::transpose(W));
}

void Layer::step(double lr, double beta)
{
    for (size_t i = 0; i < W.noOfElements(); ++i)
    {
        vW.flat(i) = static_cast<float>(beta * vW.flat(i) + dW.flat(i));
        W.flat(i) -= static_cast<float>(lr * vW.flat(i));
    }

    for (size_t i = 0; i < b.noOfElements(); ++i)
    {
        vb.flat(i) = static_cast<float>(beta * vb.flat(i) + db.flat(i));
        b.flat(i) -= static_cast<float>(lr * vb.flat(i));
    }
}


Tensor Layer::forwardNoCache(const Tensor &X) {
  return Tensor::linearForward(X, W, b);
}
