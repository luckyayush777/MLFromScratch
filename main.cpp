#include<iostream>
#include<vector>
#include<limits>
#include<cmath>
#include <stdexcept>

class Matrix {
    int numRows;
    int numCols;
    std::vector<double> matrixData;


public:
    Matrix(int r, int c)
        : numRows(r), numCols(c), matrixData(r * c, 0.0)
    {
        if (r <= 0 || c <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
    }
    Matrix(const std::vector<double>& vals, int r, int c)
        : numRows(r), numCols(c), matrixData(vals)
    {
        if (r <= 0 || c <= 0 || vals.size() != r * c) {
            throw std::invalid_argument("Invalid matrix dimensions or data size");
        }
    }
    double& operator()(int r, int c) {
        if (r < 0 || r >= numRows || c < 0 || c >= numCols) {
            throw std::out_of_range("Index out of bounds");
        }
        return matrixData[r * numCols + c]; // row-major
    }
    const double& operator()(int r, int c) const {
        if (r < 0 || r >= numRows || c < 0 || c >= numCols) {
            throw std::out_of_range("Index out of bounds");
        }
        return matrixData[r * numCols + c];
    }
    Matrix matrixMultiply(const Matrix& other) const {
        if (numCols != other.numRows) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix result(numRows, other.numCols);
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < other.numCols; ++j) {
                for (int k = 0; k < numCols; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }
    Matrix transpose() const {
        Matrix result(numCols, numRows);
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    Matrix matrixAdd(const Matrix& other) const {
        if (numRows != other.numRows || numCols != other.numCols) {
            throw std::invalid_argument("Incompatible matrix dimensions for addition");
        }
        Matrix result(numRows, numCols);
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
        return result;
    }
    Matrix matrixSubtract(const Matrix& other) const {
        if (numRows != other.numRows || numCols != other.numCols) {
            throw std::invalid_argument("Incompatible matrix dimensions for subtraction");
        }
        Matrix result(numRows, numCols);
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                result(i, j) = (*this)(i, j) - other(i, j);
            }
        }
        return result;
    }
    Matrix matrixScalarMultiply(double scalar) const {
        Matrix result(numRows, numCols);
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    }
    Matrix matrixDotMultiply(const Matrix& other) const {
        if (numRows != other.numRows || numCols != other.numCols) {
            throw std::invalid_argument("Incompatible matrix dimensions for dot multiplication");
        }
        Matrix result(numRows, numCols);
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                result(i, j) = (*this)(i, j) * other(i, j);
            }
        }
        return result;
    }

    double dot(const Matrix& other) const {
        if (numRows != other.numRows || numCols != other.numCols) {
            throw std::invalid_argument("Incompatible matrix dimensions for dot product");
        }
        double result = 0.0;
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                result += (*this)(i, j) * other(i, j);
            }
        }
        return result;
    }

    // Operator overloads for cleaner syntax
    Matrix operator+(const Matrix& other) const {
        return matrixAdd(other);
    }

    Matrix operator-(const Matrix& other) const {
        return matrixSubtract(other);
    }

    Matrix operator*(double scalar) const {
        return matrixScalarMultiply(scalar);
    }

    Matrix operator*(const Matrix& other) const {
        return matrixMultiply(other);
    }

    friend Matrix operator*(double scalar, const Matrix& m) {
        return m.matrixScalarMultiply(scalar);
    }

    int getNumRows() const { return numRows; }
    int getNumCols() const { return numCols; }
};

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
    const double threshold = 1e-6; // common-sense convergence threshold
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
    const double threshold = 1e-6; // common-sense convergence threshold
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