#pragma once
#include"tensor.h"
struct Relu{
    static Tensor reluForward(const Tensor&Z);

    static Tensor reluBackward(const Tensor&Z, const Tensor& dA);
};
