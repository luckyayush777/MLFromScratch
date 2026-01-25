#include<iostream>
#include<vector>
#include<limits>
#include<cmath>


#include <vector>
#include <stdexcept>

class Matrix {
    int rows;
    int cols;
    std::vector<double> data;

public:
    Matrix(int r, int c)
        : rows(r), cols(c), data(r * c, 0.0)
    {
        if (r <= 0 || c <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
    }
    Matrix(const std::vector<double>& vals, int r, int c)
        : rows(r), cols(c), data(vals)
    {
        if (r <= 0 || c <= 0 || vals.size() != r * c) {
            throw std::invalid_argument("Invalid matrix dimensions or data size");
        }
    }

    double& operator()(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[r * cols + c]; // row-major
    }

    const double& operator()(int r, int c) const {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[r * cols + c];
    }

    Matrix matrix_multiply(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }

    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    
    int num_rows() const { return rows; }
    int num_cols() const { return cols; }
};

void gradientDescent(const Matrix& x, const Matrix& y,
                     double& slope, double& intercept,
                     double learningRate, int maxIterations)
{
    int m = y.num_rows();
    double prevLoss = std::numeric_limits<double>::infinity();
    const double threshold = 1e-6; // common-sense convergence threshold

    for (int it = 0; it < maxIterations; ++it) {
        double slopeGradient = 0.0;
        double interceptGradient = 0.0;
        double loss = 0.0;

        for (int i = 0; i < m; ++i) {
            double prediction = slope * x(i, 0) + intercept;
            double error = prediction - y(i, 0);

            loss += error * error;
            slopeGradient += error * x(i, 0);
            interceptGradient += error;
        }

        loss /= m;
        slopeGradient *= (2.0 / m);
        interceptGradient *= (2.0 / m);

        if (std::abs(prevLoss - loss) < threshold) {
            std::cout << "Converged at iteration " << it << "\n";
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
    int m = y.num_rows();
    double prevLoss = std::numeric_limits<double>::infinity();
    const double threshold = 1e-6; // common-sense convergence threshold
    Matrix W(2, 1);
    W(0,0) = 0.0;
    W(1,0) = 0.0; // W(0,0) = slope, W(1,0) = intercept

    Matrix gradients(2, 1);
    for (int it = 0; it < maxIterations; ++it) {
        gradients(0,0) = 0.0;
        gradients(1,0) = 0.0;
        double loss = 0.0;
        
        
        for (int i = 0; i < m; ++i) {
            double prediction = W(0,0) * x(i, 0) + W(1,0);
            double error = prediction - y(i, 0);

            loss += error * error;
            gradients(0,0) += error * x(i, 0);
            gradients(1,0) += error;
        }
        loss /= m;
        gradients(0,0) *= (2.0 / m);
        gradients(1,0) *= (2.0 / m);
        if (std::abs(prevLoss - loss) < threshold) {
            std::cout << "Converged at iteration " << it << "\n";
            break;
        }
        prevLoss = loss;
        W(0,0) -= learningRate * gradients(0,0);
        W(1,0) -= learningRate * gradients(1,0);

    }
    std::cout << "Slope: " << W(0,0) << ", Intercept: " << W(1,0) << std::endl;
}

void gradientDescentVectorized(const Matrix& x, const Matrix& y,
                     double& slope, double& intercept,
                     double learningRate, int maxIterations)
{
    int m = y.num_rows();
    double prevLoss = std::numeric_limits<double>::infinity();
    Matrix W(2, 1);
    W(0,0) = slope;
    W(1,0) = intercept;
    const double threshold = 1e-6; // common-sense convergence threshold
    for (int it = 0; it < maxIterations; ++it) {
        Matrix predictions = x.matrix_multiply(W);
        Matrix errors(m, 1);
        for (int i = 0; i < m; ++i) {
            errors(i, 0) = predictions(i, 0) - y(i, 0);
        }

        double loss = 0.0;
        for (int i = 0; i < m; ++i) {
            loss += errors(i, 0) * errors(i, 0);
        }
        loss /= m;

        Matrix xTransposed = x.transpose();
        Matrix gradients = xTransposed.matrix_multiply(errors);
        for (int i = 0; i < gradients.num_rows(); ++i) {
            gradients(i, 0) *= (2.0 / m);
        }
        W(0,0) -= learningRate * gradients(0, 0);
        W(1,0) -= learningRate * gradients(1, 0);

        if (std::abs(prevLoss - loss) < threshold) {
            std::cout << "Converged at iteration " << it << "\n";
            break;
        }

        prevLoss = loss;
        slope = W(0,0);
        intercept = W(1,0);

    }
    slope = W(0,0);
    intercept = W(1,0);
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