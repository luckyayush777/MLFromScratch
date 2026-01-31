#pragma once

#include<string>
#include<vector>
#include<fstream>
#include <stdexcept>
#include"tensor.h"


inline uint32_t read_be_uint32(std::ifstream& f) {
    uint32_t x = 0;
    for (int i = 0; i < 4; ++i) {
        x = (x << 8) | f.get();
    }
    return x;
}

void readMnistImages(std::ifstream& img, Tensor& images, size_t numImages, size_t numRows, size_t numCols);
void readMnistLabels(std::ifstream& lbl, Tensor& labels, size_t numLabels);
struct MNISTDataset{
    Tensor images; // Shape : (flat form) [num_images, 28*28]   
    Tensor labels; //Shape : raw form [num_images] with values 0-9
};

MNISTDataset loadMnist(const std::string& imageFile, const std::string& labelFile){
    std::ifstream img(imageFile, std::ios::binary);
    std::ifstream lbl(labelFile,  std::ios::binary);
    if(!img.is_open() || !lbl.is_open()){
        throw std::runtime_error("Could not open MNIST data files");
    }
    // Read image file header
    uint32_t imgMagic = read_be_uint32(img);
    uint32_t numImages = read_be_uint32(img);
    uint32_t numRows = read_be_uint32(img);
    uint32_t numCols = read_be_uint32(img);

    if(imgMagic != 2051){
        throw std::runtime_error("Invalid MNIST image file magic number");
    }
    if(numRows != 28 || numCols != 28){
        throw std::runtime_error("MNIST images must be 28x28");
    }

    // Read label file header
    uint32_t lblMagic = read_be_uint32(lbl);
    uint32_t numLabels = read_be_uint32(lbl);
    if(lblMagic != 2049){
        throw std::runtime_error("Invalid MNIST label file magic number");
    }
    if(numLabels != numImages){
        throw std::runtime_error("Number of labels does not match number of images");
    }
    Tensor images({numImages, numRows * numCols});
    Tensor labels({numImages});

    readMnistImages(img, images, numImages, numRows, numCols);
    readMnistLabels(lbl, labels, numLabels);

    return {images, labels};
}

void readMnistImages(std::ifstream& img, Tensor& images, size_t numImages, size_t numRows, size_t numCols)
    {
    const size_t imageSize = numRows * numCols;

    for (size_t i = 0; i < numImages; ++i) {
        for (size_t j = 0; j < imageSize; ++j) {
            unsigned char pixel;
            if (!img.read(reinterpret_cast<char*>(&pixel), 1)) {
                throw std::runtime_error("Unexpected end of image file");
            }
            images.at(i, j) = static_cast<double>(pixel) / 255.0;
        }
    }
}

void readMnistLabels(std::ifstream& lbl, Tensor& labels, size_t numLabels)
{
    for (size_t i = 0; i < numLabels; ++i) {
        unsigned char y;
        if (!lbl.read(reinterpret_cast<char*>(&y), 1)) {
            throw std::runtime_error("Unexpected end of label file");
        }
        labels.flat(i) = static_cast<double>(y);
    }
}