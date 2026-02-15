#pragma once

#include <string>

#include "conv2d.h"
#include "layer.h"
#include "run_logger.h"
#include "timer.h"

class TrainingRunLoggingSession {
public:
  TrainingRunLoggingSession(
      std::string optimizationTitle, Conv2d &convLayer, Layer &firstFcLayer,
      Layer &secondFcLayer, size_t numClasses, size_t trainSamples,
      size_t testSamples, double learningRate, double beta, size_t batchSize,
      size_t epochs, unsigned int seed = 42,
      const std::string &logFilePath = "observations/training_runs.txt")
      : convLayer_(convLayer), firstFcLayer_(firstFcLayer),
        secondFcLayer_(secondFcLayer),
        logConfig_(buildLogConfig(std::move(optimizationTitle), numClasses,
                                  trainSamples, testSamples, learningRate, beta,
                                  batchSize, epochs, seed, logFilePath)),
        timer_("Training run", [this](double elapsedSeconds) {
          bool logged =
              appendTrainingRunLog(logConfig_, convLayer_, firstFcLayer_,
                                   secondFcLayer_, runSummary_, elapsedSeconds);
          if (logged) {
            std::cout << "Training log updated at " << logConfig_.logFilePath
                      << "\n";
          } else {
            std::cerr << "Failed to write training log to "
                      << logConfig_.logFilePath << "\n";
          }
        }) {}

  TrainingRunSummary &summary() { return runSummary_; }

private:
  static TrainingRunLogConfig
  buildLogConfig(std::string optimizationTitle, size_t numClasses,
                 size_t trainSamples, size_t testSamples, double learningRate,
                 double beta, size_t batchSize, size_t epochs,
                 unsigned int seed, const std::string &logFilePath) {
    TrainingRunLogConfig config;
    config.optimizationTitle = std::move(optimizationTitle);
    config.logFilePath = logFilePath;
    config.inputChannels = 1;
    config.inputHeight = 28;
    config.inputWidth = 28;
    config.numClasses = numClasses;
    config.trainSamples = trainSamples;
    config.testSamples = testSamples;
    config.learningRate = learningRate;
    config.beta = beta;
    config.batchSize = batchSize;
    config.epochs = epochs;
    config.seed = seed;
    return config;
  }

  Conv2d &convLayer_;
  Layer &firstFcLayer_;
  Layer &secondFcLayer_;
  TrainingRunLogConfig logConfig_;
  TrainingRunSummary runSummary_;
  Timer timer_;
};
