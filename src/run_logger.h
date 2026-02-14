#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "conv2d.h"
#include "layer.h"

struct TrainingRunLogConfig {
  std::string optimizationTitle;
  std::string logFilePath = "observations/training_runs.txt";

  size_t inputChannels = 1;
  size_t inputHeight = 28;
  size_t inputWidth = 28;
  size_t numClasses = 10;

  size_t trainSamples = 0;
  size_t testSamples = 0;

  double learningRate = 0.0;
  double beta = 0.0;
  size_t batchSize = 0;
  size_t epochs = 0;

  unsigned int seed = 42;
};

inline std::string currentTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);

  std::tm tmBuffer{};
#if defined(_WIN32)
  localtime_s(&tmBuffer, &nowTime);
#else
  localtime_r(&nowTime, &tmBuffer);
#endif

  std::ostringstream oss;
  oss << std::put_time(&tmBuffer, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

inline bool appendTrainingRunLog(const TrainingRunLogConfig &config,
                                 const Conv2d &conv1,
                                 const Layer &fc1,
                                 const Layer &fc2,
                                 const TrainingRunSummary &summary,
                                 double elapsedSeconds) {
  std::filesystem::path outputPath(config.logFilePath);
  if (outputPath.has_parent_path()) {
    std::filesystem::create_directories(outputPath.parent_path());
  }

  std::ofstream out(config.logFilePath, std::ios::app);
  if (!out.is_open()) {
    return false;
  }

  const size_t flattenedDim = config.inputChannels * config.inputHeight *
                              config.inputWidth;

  out << "============================================================\n";
  out << "Timestamp: " << currentTimestamp() << "\n";
  out << "Optimization Title: " << config.optimizationTitle << "\n";
  out << "\n";
  out << "[Training Time]\n";
  out << "Elapsed Seconds: " << std::fixed << std::setprecision(3)
      << elapsedSeconds << "\n";
  out << "\n";
  out << "[Data]\n";
  out << "Input: " << config.inputChannels << "x" << config.inputHeight
      << "x" << config.inputWidth << " (flattened: " << flattenedDim
      << ")\n";
  out << "Classes: " << config.numClasses << "\n";
  out << "Train Samples: " << config.trainSamples << "\n";
  out << "Test Samples: " << config.testSamples << "\n";
  out << "\n";
  out << "[Hyperparameters]\n";
  out << "Learning Rate: " << config.learningRate << "\n";
  out << "Momentum Beta: " << config.beta << "\n";
  out << "Batch Size: " << config.batchSize << "\n";
  out << "Epochs: " << config.epochs << "\n";
  out << "Seed: " << config.seed << "\n";
  out << "\n";
  out << "[Model]\n";
  out << "Layer 1: Conv2d"
      << " | in_ch=" << conv1.inChannels << " out_ch=" << conv1.outChannels
      << " kernel=" << conv1.kernel << " stride=" << conv1.stride
      << " padding=" << conv1.padding << "\n";
  out << "Layer 2: ReLU\n";
  out << "Layer 3: MaxPool2d | pool=2 stride=2\n";
  out << "Layer 4: Flatten\n";
  out << "Layer 5: Linear"
      << " | in=" << fc1.W.dim(0) << " out=" << fc1.W.dim(1) << "\n";
  out << "Layer 6: ReLU\n";
  out << "Layer 7: Linear"
      << " | in=" << fc2.W.dim(0) << " out=" << fc2.W.dim(1) << "\n";
  out << "Layer 8: Softmax + CrossEntropy\n";
  out << "\n";
  out << "[Final Metrics]\n";
  out << "Completed Epochs: " << summary.completedEpochs << "\n";
  out << "Final Avg Loss: " << summary.finalAvgLoss << "\n";
  out << "Final Train Accuracy: " << summary.finalTrainAcc << "\n";
  out << "Final Test Accuracy: " << summary.finalTestAcc << "\n";
  out << "\n";

  out << "[Epoch Logs]\n";
  for (const auto &epoch : summary.epochStats) {
    out << "Epoch " << epoch.epochIndex << " | Avg Loss: " << epoch.avgLoss
        << " | Train Acc: " << epoch.trainAcc << "\n";
    out << "  Forward: " << epoch.forwardSeconds
        << "s | Backward: " << epoch.backwardSeconds
        << "s | Update: " << epoch.updateSeconds << "s\n";
    if (epoch.testAcc >= 0.0) {
      out << "         Test Acc: " << epoch.testAcc << "\n";
    }
  }
  out << "\n";

  return true;
}
