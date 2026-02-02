#include<iostream>
#include<vector>
#include<limits>
#include<cmath>
#include <stdexcept>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "matrix.h"
#include "mnist.h"
#include "tensor.h"


void dump_png(const Tensor& images, size_t index, const char* filename) {
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

void gradientDescent(const Matrix& x, const Matrix& y,
                     double& slope, double& intercept,
                     double learningRate, int maxIterations)
{
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

void gradientDescentMatricesOnly(const Matrix& x, const Matrix& y,
                     double learningRate, int maxIterations)
{
    int numSamples = y.getNumRows();
    double prevLoss = std::numeric_limits<double>::infinity();
    const double threshold = 1e-6;
    Matrix weights(2, 1);
    weights(0,0) = 0.0;
    weights(1,0) = 0.0; // weights(0,0) = slope, weights(1,0) = intercept

    Matrix gradients(2, 1);
    for (int iter = 0; iter < maxIterations; ++iter) {
        gradients(0,0) = 0.0;
        gradients(1,0) = 0.0;
        double loss = 0.0;

        for (int i = 0; i < numSamples; ++i) {
            double prediction = weights(0,0) * x(i, 0) + weights(1,0);
            double error = prediction - y(i, 0);

            loss += error * error;
            gradients(0,0) += error * x(i, 0);
            gradients(1,0) += error;
        }
        loss /= numSamples;
        gradients(0,0) *= (2.0 / numSamples);
        gradients(1,0) *= (2.0 / numSamples);
        if (std::abs(prevLoss - loss) < threshold) {
            std::cout << "Converged at iteration " << iter << "\n";
            break;
        }
        prevLoss = loss;
        weights(0,0) -= learningRate * gradients(0,0);
        weights(1,0) -= learningRate * gradients(1,0);

    }
    std::cout << "Slope: " << weights(0,0) << ", Intercept: " << weights(1,0) << std::endl;
}

void gradientDescentVectorized(const Matrix& x, const Matrix& y,
                     double& slope, double& intercept,
                     double learningRate, int maxIterations)
{
    int numSamples = y.getNumRows();
    double prevLoss = std::numeric_limits<double>::infinity();
    Matrix weights(2, 1);
    weights(0,0) = slope;
    weights(1,0) = intercept;
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
    slope = weights(0,0);
    intercept = weights(1,0);
    std::cout << "Slope: " << slope << ", Intercept: " << intercept << std::endl;
}

Matrix relu(const Matrix& z)
{
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

void forwardPropagation(
    const Matrix& X,
    const Matrix& W1,
    const Matrix& W2,
    Matrix& Z1,
    Matrix& A1,
    Matrix& A2)
{
    Z1 = X * W1;
    A1 = relu(Z1);
    A2 = A1 * W2;
    std::cout << "Forward pass complete\n";
}


Matrix reluBackward(const Matrix& dA, const Matrix& Z)
{
    Matrix dZ(Z.getNumRows(), Z.getNumCols());

    for (int i = 0; i < Z.getNumRows(); ++i) {
        for (int j = 0; j < Z.getNumCols(); ++j) {
            dZ(i, j) = (Z(i, j) > 0.0) ? dA(i, j) : 0.0;
        }
    }
    return dZ;
}

Matrix mseBackward(const Matrix& predictions,
                   const Matrix& targets)
{
    int m = predictions.getNumRows();
    Matrix dA(predictions.getNumRows(), predictions.getNumCols());

    for (int i = 0; i < predictions.getNumRows(); ++i) {
        for (int j = 0; j < predictions.getNumCols(); ++j) {
            dA(i, j) = (2.0 / m) * (predictions(i, j) - targets(i, j));
        }
    }
    return dA;
}

void sgdUpdate(Matrix& W, const Matrix& dW, double lr)
{
    for (int i = 0; i < W.getNumRows(); ++i) {
        for (int j = 0; j < W.getNumCols(); ++j) {
            W(i, j) -= lr * dW(i, j);
        }
    }
}

double computeMSE(const Matrix& predictions, const Matrix& targets)
{
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

void backwardPropagation(
    const Matrix& X,
    const Matrix& Y,
    Matrix& W1,
    Matrix& W2,
    const Matrix& Z1,
    const Matrix& A1,
    const Matrix& A2,
    double learningRate,
    Matrix& dW1,
    Matrix& dW2)
{
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

// Problem 1: Simple 2x2x1 network
void problem1()
{
    std::cout << "\nPROBLEM 1\n";
    
    Matrix X(2, 2);
    X(0, 0) = 1.0; X(0, 1) = 2.0;
    X(1, 0) = 3.0; X(1, 1) = 4.0;
    
    Matrix Y(2, 1);
    Y(0, 0) = 5.0;
    Y(1, 0) = 6.0;
    
    Matrix W1(2, 2);
    W1(0, 0) = 0.1; W1(0, 1) = 0.2;
    W1(1, 0) = 0.3; W1(1, 1) = 0.4;
    
    Matrix W2(2, 1);
    W2(0, 0) = 0.5;
    W2(1, 0) = 0.6;
    
    Matrix Z1(2, 2), A1(2, 2), A2(2, 1);
    Matrix dW1(2, 2), dW2(2, 1);

    double lr = 0.01;
    int iters = 1000;

    for (int i = 0; i < iters; ++i) {
        forwardPropagation(X, W1, W2, Z1, A1, A2);
        backwardPropagation(X, Y, W1, W2, Z1, A1, A2, lr, dW1, dW2);
        if(iters % 100 == 0){
            double loss = computeMSE(A2, Y);
            std::cout << "Iteration " << i << ", Loss: " << loss << "\n";
        }
    }
    
    std::cout << "Final predictions:\n";
    for (int i = 0; i < A2.getNumRows(); ++i) {
        std::cout << A2(i,0) << " (target " << Y(i,0) << ")\n";
    }

}

// Problem 2: Simple 2x3x1 network
void problem2()
{
    std::cout << "\nPROBLEM 2\n";
    
    Matrix X(3, 2);
    X(0, 0) = 1.0; X(0, 1) = 1.0;
    X(1, 0) = 2.0; X(1, 1) = 2.0;
    X(2, 0) = 3.0; X(2, 1) = 3.0;
    
    Matrix Y(3, 1);
    Y(0, 0) = 2.0;
    Y(1, 0) = 4.0;
    Y(2, 0) = 6.0;
    
    Matrix W1(2, 3);
    W1(0, 0) = 0.1; W1(0, 1) = 0.2; W1(0, 2) = 0.3;
    W1(1, 0) = 0.4; W1(1, 1) = 0.5; W1(1, 2) = 0.6;
    
    Matrix W2(3, 1);
    W2(0, 0) = 0.7;
    W2(1, 0) = 0.8;
    W2(2, 0) = 0.9;
    
    Matrix Z1(3, 3), A1(3, 3), A2(3, 1);
    Matrix dW1(2, 3), dW2(3, 1);
    
    forwardPropagation(X, W1, W2, Z1, A1, A2);
    double loss = computeMSE(A2, Y);
    std::cout << "Loss: " << loss << "\n";
    backwardPropagation(X, Y, W1, W2, Z1, A1, A2, 0.01, dW1, dW2);
}


int main() {
    auto dataset = loadMnist(
        "datasets/train-images-idx3-ubyte",
        "datasets/train-labels-idx1-ubyte"
    );

    constexpr size_t B = 32; // batch size
    constexpr size_t D = 28 * 28; // input dimension
    constexpr size_t C = 10; // number of classes
    double learningRate = 0.1;

    Tensor X({B, D});
    Tensor y({B});

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < D; ++j)
            X.at(i, j) = dataset.images.at(i, j);
        y.flat(i) = dataset.labels.flat(i);
    }


    Tensor W({D, C});
    Tensor b({C});

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.01);

    for (size_t i = 0; i < W.noOfElements(); ++i)
        W.flat(i) = dist(rng);
    for (size_t i = 0; i < b.noOfElements(); ++i)
        b.flat(i) = 0.0;

    Tensor dW({D, C});
    Tensor db({C});


    double prevLoss = std::numeric_limits<double>::infinity();
    const double threshold = 1e-6;
    const int maxSteps = 10000;

    for (int step = 0; step < maxSteps; ++step) {

        // forward
        Tensor logits = Tensor::linearForward(X, W, b);
        Tensor probs = logits;
        Tensor::softmax(probs);

        double loss = Tensor::crossEntropyLoss(probs, y);
        std::cout << "Step " << step << " | loss = " << loss << "\n";

        if(std::abs(prevLoss - loss) < threshold) {
            std::cout << "Converged at step " << step << "\n";
            break;
        }
        prevLoss = loss;

        // backward
        Tensor dZ = probs;
        Tensor::softmaxCrossEntropyBackward(dZ, y);
        Tensor::linearBackward(X, dZ, dW, db);

        // SGD update
        for (size_t i = 0; i < W.noOfElements(); ++i)
            W.flat(i) -= learningRate * dW.flat(i);
        for (size_t i = 0; i < b.noOfElements(); ++i)
            b.flat(i) -= learningRate * db.flat(i);
    }

    return 0;
}