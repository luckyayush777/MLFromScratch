#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <random>
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
        return matrixData[r * numCols + c];
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
    
    void randomInit(double min, double max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                (*this)(i, j) = dis(gen);
            }
        }
    }
    
    void print(const std::string& name) const {
        std::cout << name << " [" << numRows << "x" << numCols << "]:\n";
        for (int i = 0; i < numRows; ++i) {
            std::cout << "  ";
            for (int j = 0; j < numCols; ++j) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                          << (*this)(i, j);
                if (j < numCols - 1) std::cout << " ";
            }
            std::cout << "\n";
        }
    }
    
    double sum() const {
        double total = 0.0;
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                total += (*this)(i, j);
            }
        }
        return total;
    }
};