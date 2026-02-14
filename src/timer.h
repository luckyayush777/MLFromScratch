#pragma once
#include <chrono>
#include <functional>
#include <iostream>
#include <string>

class Timer {
public:
    explicit Timer(const std::string& name,
                   std::function<void(double)> onStop = {})
        : name_(name),
          onStop_(std::move(onStop)),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);

        double elapsedSeconds = duration.count() / 1000.0;

        std::cout << name_
                  << " took "
                  << elapsedSeconds
                  << "s\n";

        if (onStop_) {
            onStop_(elapsedSeconds);
        }
    }

private:
    std::string name_;
    std::function<void(double)> onStop_;
    std::chrono::high_resolution_clock::time_point start_;
};
