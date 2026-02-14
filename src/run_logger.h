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

inline std::filesystem::path findProjectRootFromCurrentPath() {
  std::filesystem::path cursor = std::filesystem::current_path();

  while (!cursor.empty()) {
    const auto cmakePath = cursor / "CMakeLists.txt";
    const auto srcPath = cursor / "src";

    if (std::filesystem::exists(cmakePath) &&
        std::filesystem::exists(srcPath)) {
      return cursor;
    }

    if (cursor == cursor.root_path()) {
      break;
    }
    cursor = cursor.parent_path();
  }

  return {};
}

inline bool appendLogEntryToPath(const std::filesystem::path &outputPath,
                                 const std::string &entry) {
  if (outputPath.empty()) {
    return false;
  }

  if (outputPath.has_parent_path()) {
    std::filesystem::create_directories(outputPath.parent_path());
  }

  std::ofstream out(outputPath, std::ios::app);
  if (!out.is_open()) {
    return false;
  }

  out << entry;
  return true;
}

inline bool appendTrainingRunLog(const TrainingRunLogConfig &config,
                                 const Conv2d &conv1, const Layer &fc1,
                                 const Layer &fc2,
                                 const TrainingRunSummary &summary,
                                 double elapsedSeconds) {
  std::ostringstream log;

  const size_t flattenedDim =
      config.inputChannels * config.inputHeight * config.inputWidth;

  log << "============================================================\n";
  log << "Timestamp: " << currentTimestamp() << "\n";
  log << "Optimization Title: " << config.optimizationTitle << "\n";
  log << "\n";
  log << "[Training Time]\n";
  log << "Elapsed Seconds: " << std::fixed << std::setprecision(3)
      << elapsedSeconds << "\n";
  log << "\n";
  log << "[Data]\n";
  log << "Input: " << config.inputChannels << "x" << config.inputHeight << "x"
      << config.inputWidth << " (flattened: " << flattenedDim << ")\n";
  log << "Classes: " << config.numClasses << "\n";
  log << "Train Samples: " << config.trainSamples << "\n";
  log << "Test Samples: " << config.testSamples << "\n";
  log << "\n";
  log << "[Hyperparameters]\n";
  log << "Learning Rate: " << config.learningRate << "\n";
  log << "Momentum Beta: " << config.beta << "\n";
  log << "Batch Size: " << config.batchSize << "\n";
  log << "Epochs: " << config.epochs << "\n";
  log << "Seed: " << config.seed << "\n";
  log << "\n";
  log << "[Model]\n";
  log << "Layer 1: Conv2d"
      << " | in_ch=" << conv1.inChannels << " out_ch=" << conv1.outChannels
      << " kernel=" << conv1.kernel << " stride=" << conv1.stride
      << " padding=" << conv1.padding << "\n";
  log << "Layer 2: ReLU\n";
  log << "Layer 3: MaxPool2d | pool=2 stride=2\n";
  log << "Layer 4: Flatten\n";
  log << "Layer 5: Linear"
      << " | in=" << fc1.W.dim(0) << " out=" << fc1.W.dim(1) << "\n";
  log << "Layer 6: ReLU\n";
  log << "Layer 7: Linear"
      << " | in=" << fc2.W.dim(0) << " out=" << fc2.W.dim(1) << "\n";
  log << "Layer 8: Softmax + CrossEntropy\n";
  log << "\n";
  log << "[Final Metrics]\n";
  log << "Completed Epochs: " << summary.completedEpochs << "\n";
  log << "Final Avg Loss: " << summary.finalAvgLoss << "\n";
  log << "Final Train Accuracy: " << summary.finalTrainAcc << "\n";
  log << "Final Test Accuracy: " << summary.finalTestAcc << "\n";
  log << "\n";

  log << "[Epoch Logs]\n";
  for (const auto &epoch : summary.epochStats) {
    log << "Epoch " << epoch.epochIndex << " | Avg Loss: " << epoch.avgLoss
        << " | Train Acc: " << epoch.trainAcc << "\n";
    log << "  Forward: " << epoch.forwardSeconds
        << "s | Backward: " << epoch.backwardSeconds
        << "s | Update: " << epoch.updateSeconds << "s\n";
    if (epoch.testAcc >= 0.0) {
      log << "         Test Acc: " << epoch.testAcc << "\n";
    }
  }
  log << "\n";

  bool wroteConfiguredPath = appendLogEntryToPath(
      std::filesystem::path(config.logFilePath), log.str());

  const std::filesystem::path projectRoot = findProjectRootFromCurrentPath();
  bool wroteRootMirror = false;
  if (!projectRoot.empty()) {
    const std::filesystem::path rootMirrorPath =
        projectRoot / "observations" / "training_runs.txt";
    if (rootMirrorPath != std::filesystem::path(config.logFilePath)) {
      wroteRootMirror = appendLogEntryToPath(rootMirrorPath, log.str());
    }
  }

  return wroteConfiguredPath || wroteRootMirror;
}
