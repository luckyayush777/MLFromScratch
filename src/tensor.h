#pragma once

#include<vector>
#include<stdexcept>
#include<numeric>


class Tensor {

    private :
    std::vector<size_t> shape;
    size_t total_size = 0;
    std::vector<double> data;

    public :
    Tensor() = default;
    explicit Tensor(const std::vector<size_t>& shape_) : shape(shape_){
        if(shape.empty()){
            throw std::invalid_argument("Shape cannot be empty");
        }
        total_size = 1;
        for(size_t d : shape){
            if(d == 0){
                throw std::invalid_argument("Shape dimensions must be greater than zero");
            }
            total_size *= d;
        }
        data.resize(total_size, 0.0);
    }
    const std::vector<size_t>& getShape() const {
        return shape;
    }

    size_t noOfElements() const {
        return total_size;
    }

    double* raw() {
        return data.data();
    }

    const double* raw() const {
        return data.data();
    }

    double& flat(size_t i) {
        if (i >= total_size) {
            throw std::out_of_range("Tensor flat index out of range");
        }
        return data[i];
    }

    const double& flat(size_t i) const {
        if (i >= total_size) {
            throw std::out_of_range("Tensor flat index out of range");
        }
        return data[i];
    }

    double& at(size_t i, size_t j) {
        if (shape.size() != 2) {
            throw std::logic_error("Tensor is not 2D");
        }
        size_t cols = shape[1];
        if (i >= shape[0] || j >= cols) {
            throw std::out_of_range("Tensor index out of range");
        }
        return data[i * cols + j];
    }

    const double& at(size_t i, size_t j) const {
        if (shape.size() != 2) {
            throw std::logic_error("Tensor is not 2D");
        }
        size_t cols = shape[1];
        if (i >= shape[0] || j >= cols) {
            throw std::out_of_range("Tensor index out of range");
        }
        return data[i * cols + j];
    }

};