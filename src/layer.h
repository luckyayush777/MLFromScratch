#pragma once
#include"tensor.h"

struct Layer{
    Tensor W;
    Tensor b;
    Tensor dW;
    Tensor db;
    Tensor X_cache;


    Layer(size_t in, size_t out):W({in,out}),b({out}),dW({in,out}),db({out}){
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 0.01);
        for (size_t i = 0; i < W.noOfElements(); ++i)
            W.flat(i) = dist(rng);
        for (size_t i = 0; i < b.noOfElements(); ++i)
            b.flat(i) = 0.0;
    }

    static Tensor forward(Layer& layer, const Tensor &X){
        layer.X_cache = X; //store input for backward pass
        return Tensor::linearForward(X, layer.W, layer.b);
    }


    Tensor backward(Layer& layer, const Tensor &dY){
        Tensor dX({layer.X_cache.dim(0), layer.W.dim(0)});
        Tensor::linearBackward(layer.X_cache, dY, layer.dW, layer.db);
        dX = Tensor::matmul(dY, Tensor::transpose(layer.W));    
        return dX;
    }

    void step(Layer& layer, double learningRate){
        for (size_t i = 0; i < layer.W.noOfElements(); ++i)
            layer.W.flat(i) -= learningRate * layer.dW.flat(i);
        for (size_t i = 0; i < layer.b.noOfElements(); ++i)
            layer.b.flat(i) -= learningRate * layer.db.flat(i);
    }
};