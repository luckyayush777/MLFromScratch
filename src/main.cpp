#include<iostream>
#include<vector>
#include<limits>
#include<cmath>
#include <stdexcept>

#include "matrix.h"


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

void forwardPropagation(const Matrix& x, const Matrix& weights,
                        Matrix& output)
{
    output = relu(x * weights);
}




int main(){
    Matrix y(7, 1);
    y(0,0) = -7.0;
    y(1,0) = -5.0;
    y(2,0) = -3.0;
    y(3,0) = -1.0;
    y(4,0) = 1.0;
    y(5,0) = 3.0;       
    y(6,0) = 5.0;

    Matrix x(7, 2);
    x(0, 0) = -3.0; x(0, 1) = 1.0;  
    x(1, 0) = -2.0; x(1, 1) = 1.0;
    x(2, 0) = -1.0; x(2, 1) = 1.0;
    x(3, 0) = 0.0;  x(3, 1) = 1.0;
    x(4, 0) = 1.0;  x(4, 1) = 1.0;
    x(5, 0) = 2.0;  x(5, 1) = 1.0;
    x(6, 0) = 3.0;  x(6, 1) = 1.0;
    double slope = 0.0;
    double intercept = 0.0;

    gradientDescentVectorized(x, y, slope, intercept, 0.01, 1000);
    return 0;
}