#ifndef SAM3_TRACKER_CPP_H_
#define SAM3_TRACKER_CPP_H_

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <list>
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "util.h"

class Sam3Tracker {
  std::unique_ptr<Ort::Session> visionEncoder, decoder;
  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  Ort::RunOptions runOptionsEncoder;
  Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<float> inputTensorValuesFloat;
  std::vector<int64_t> inputShapeVision;
  std::vector<int64_t> outputShapeVision[3];
  std::vector<float> outputVision[3];
  std::vector<int64_t> outputShapeDecoder[3];
  std::vector<float> outputDecoder[3];
  std::vector<std::vector<float>> previousMasks;
  std::vector<std::string> cachedInputNamesVision, cachedOutputNamesVision;
  std::vector<std::string> cachedInputNamesDecoder, cachedOutputNamesDecoder;
  std::vector<const char*> ptrInputNamesVision, ptrOutputNamesVision;
  std::vector<const char*> ptrInputNamesDecoder, ptrOutputNamesDecoder;
  bool loadingModel = false;
  bool preprocessing = false;
  bool terminating = false;
 public:
  Sam3Tracker();
  ~Sam3Tracker();
  bool clearLoadModel();
  void clearVisionBatch();
  void clearDecoder();
  bool isDecoderEmpty();
  void clearPreviousMasks();
  void resizePreviousMasks(int previousMaskIdx);
  void terminatePreprocessing();
  bool loadModel(const std::string& visionPath, const std::string& decoderPath, int threadsNumber, const std::string device);
  void loadingStart();
  void loadingEnd();
  cv::Size getInputSize();
  cv::Size getMaskSize();
  bool preprocessImage(const cv::Mat& image);
  void preprocessingStart();
  void preprocessingEnd();
  void setPointsLabels(const std::vector<cv::Point2f>& points, const std::vector<int> &labels, std::vector<float> *inputPointValues, std::vector<int64_t> *inputLabelValues);
  void setDecorderTensorsEmbeddings(std::vector<Ort::Value> *inputTensors);
  void setDecorderTensorsPointsLabels(std::vector<float> &inputPointValues, std::vector<int64_t> &inputLabelValues, int numPoints, std::vector<Ort::Value> *inputTensors);
  void setDecorderTensorsMaskInput(const size_t maskInputSize, float *maskInputValues, float *hasMaskValues, std::vector<float> &previousMaskInputValues, std::vector<Ort::Value> *inputTensors);
  cv::Mat getMask(std::vector<float> &inputPointValues, std::vector<int64_t> &inputLabelValues, const cv::Size &imageSize, int previousMaskIdx, bool isNextGetMask);
};

#endif
