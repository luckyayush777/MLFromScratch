#pragma once
#include <chrono>
#include <iostream>
#include <string>

class Timer {
public:
    explicit Timer(const std::string& name)
        : name_(name),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);

        std::cout << name_
                  << " took "
                  << duration.count() / 1000.0
                  << "s\n";
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};
