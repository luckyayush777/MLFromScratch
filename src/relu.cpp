
#include "relu.h"

static Tensor reluForward(const Tensor &Z) {
  Tensor output(Z.getShape());
  for (size_t i = 0; i < Z.noOfElements(); ++i) {
    output.flat(i) = std::max(0.0, Z.flat(i));
  }
  return output;
}

static Tensor reluBackward(const Tensor &Z, const Tensor &dA) {
  Tensor dZ(Z.getShape());
  for (size_t i = 0; i < Z.noOfElements(); ++i) {
    dZ.flat(i) = Z.flat(i) > 0.0 ? dA.flat(i) : 0.0;
  }
  return dZ;
}