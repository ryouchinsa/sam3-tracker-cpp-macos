#include "sam3_tracker.h"
#include <opencv2/opencv.hpp>
#include <future>

Sam3Tracker::Sam3Tracker(){}
Sam3Tracker::~Sam3Tracker(){
  if(loadingModel){
    return;
  }
  if(preprocessing){
    return;
  }
  clearLoadModel();
  clearPreviousMasks();
}

bool Sam3Tracker::clearLoadModel(){
  try{
    Ort::Session* v = visionEncoder.release();
    Ort::Session* d = decoder.release();
    delete v;
    delete d;
    inputTensorValuesFloat.resize(0);
    inputShapeVision.resize(0);
    for(int i = 0; i < 3; i++){
      outputShapeVision[i].resize(0);
      outputVision[i].resize(0);
    }
    clearDecoder();
  }catch(Ort::Exception& e){
    return false;
  }
  return true;
}

void Sam3Tracker::clearDecoder(){
  for(int i = 0; i < 3; i++){
    outputShapeDecoder[i].resize(0);
    outputDecoder[i].resize(0);
  }
}

bool Sam3Tracker::isDecoderEmpty(){
  if(outputDecoder[0].size() == 0){
    return true;
  }
  return false;
}

void Sam3Tracker::clearPreviousMasks(){
  previousMasks.resize(0);
}

void Sam3Tracker::resizePreviousMasks(int previousMaskIdx){
  if(previousMasks.size() > previousMaskIdx + 1){
    previousMasks.resize(previousMaskIdx + 1);
  }
}

void Sam3Tracker::terminatePreprocessing(){
  runOptionsEncoder.SetTerminate();
  terminating = true;
}

bool Sam3Tracker::loadModel(const std::string& visionPath,  const std::string& decoderPath, int threadsNumber, const std::string device){
  try{
    loadingStart();
    if(!clearLoadModel()){
      loadingEnd();
      return false;
    }
    if(!modelExists(visionPath) || !modelExists(decoderPath)){
      loadingEnd();
      return false;
    }
    // Use global thread pool like Python's onnxruntime does
    Ort::ThreadingOptions threadingOptions;
    threadingOptions.SetGlobalIntraOpNumThreads(threadsNumber);
    threadingOptions.SetGlobalInterOpNumThreads(threadsNumber);
    // Replace the Env — must be done before session creation
    env = Ort::Env(threadingOptions, ORT_LOGGING_LEVEL_WARNING, "test");
    sessionOptions.SetIntraOpNumThreads(threadsNumber);
    sessionOptions.SetInterOpNumThreads(threadsNumber);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Disable per-session thread spinning — let global pool handle it
    sessionOptions.AddConfigEntry("session.intra_op.allow_spinning", "0");
    // Enable memory pattern optimization
    sessionOptions.EnableMemPattern();
    sessionOptions.EnableCpuMemArena();
    if(device.substr(0, 5) == "cuda:"){
      int gpuDeviceId = std::stoi(device.substr(5));
      OrtCUDAProviderOptions options;
      options.device_id = gpuDeviceId;
      sessionOptions.AppendExecutionProvider_CUDA(options);
    }
    auto futureVision = std::async(std::launch::async, [&](){
      return std::make_unique<Ort::Session>(env, visionPath.c_str(), sessionOptions);
    });
    auto futureDecoder = std::async(std::launch::async, [&](){
      return std::make_unique<Ort::Session>(env, decoderPath.c_str(), sessionOptions);
    });
    visionEncoder = futureVision.get();
    decoder       = futureDecoder.get();
    auto cacheIONames = [](Ort::Session* sess,
                           std::vector<std::string>& inNames,  std::vector<const char*>& inPtrs,
                           std::vector<std::string>& outNames, std::vector<const char*>& outPtrs){
      Ort::AllocatorWithDefaultOptions alloc;
      inNames.clear();
      for(size_t i = 0; i < sess->GetInputCount(); i++)
        inNames.push_back(sess->GetInputNameAllocated(i, alloc).get());
      outNames.clear();
      for(size_t i = 0; i < sess->GetOutputCount(); i++)
        outNames.push_back(sess->GetOutputNameAllocated(i, alloc).get());
      // Only build pointer vectors AFTER all strings are final — no more reallocation
      inPtrs.clear();
      for(auto& s : inNames)  inPtrs.push_back(s.c_str());
      outPtrs.clear();
      for(auto& s : outNames) outPtrs.push_back(s.c_str());
    };
    cacheIONames(visionEncoder.get(), cachedInputNamesVision, ptrInputNamesVision,
                                      cachedOutputNamesVision, ptrOutputNamesVision);
    cacheIONames(decoder.get(),       cachedInputNamesDecoder, ptrInputNamesDecoder,
                                      cachedOutputNamesDecoder, ptrOutputNamesDecoder);
    inputShapeVision = visionEncoder->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    inputTensorValuesFloat.assign(getShapeSize(inputShapeVision), 0.0f);
    for(int i = 0; i < 3; i++){
      outputShapeVision[i] = visionEncoder->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      outputVision[i].assign(getShapeSize(outputShapeVision[i]), 0.0f);
      printShape(outputShapeVision[i]);
    }
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    loadingEnd();
    return false;
  }
  if(terminating){
    loadingEnd();
    return false;
  }
  loadingEnd();
  return true;
}

void Sam3Tracker::loadingStart(){
  loadingModel = true;
}

void Sam3Tracker::loadingEnd(){
  loadingModel = false;
  terminating = false;
}

cv::Size Sam3Tracker::getInputSize(){
  return cv::Size((int)inputShapeVision[3], (int)inputShapeVision[2]);
}

cv::Size Sam3Tracker::getMaskSize(){
  return cv::Size((int)outputShapeVision[1][3], (int)outputShapeVision[1][2]);
}

bool Sam3Tracker::preprocessImage(const cv::Mat& image){
  try{
    preprocessingStart();
    if(image.size() != cv::Size((int)inputShapeVision[3], (int)inputShapeVision[2])){
      preprocessingEnd();
      return false;
    }
    if(image.channels() != 3){
      preprocessingEnd();
      return false;
    }
    // FAST: vectorized OpenCV ops matching Python's (img / 127.5 - 1.0).transpose(2,0,1)
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F, 1.0 / 127.5, -1.0); // bgr float, normalized
    // Split into B, G, R planes and reorder to R, G, B (CHW layout)
    std::vector<cv::Mat> channels(3);
    cv::split(imageFloat, channels);  // channels[0]=B, [1]=G, [2]=R
    int64_t planeSize = inputShapeVision[2] * inputShapeVision[3];
    // Copy R, G, B into CHW tensor (matching Python's channel order)
    std::memcpy(inputTensorValuesFloat.data() + 0 * planeSize,
                channels[2].ptr<float>(), planeSize * sizeof(float)); // R
    std::memcpy(inputTensorValuesFloat.data() + 1 * planeSize,
                channels[1].ptr<float>(), planeSize * sizeof(float)); // G
    std::memcpy(inputTensorValuesFloat.data() + 2 * planeSize,
                channels[0].ptr<float>(), planeSize * sizeof(float)); // B
    auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValuesFloat.data(), inputTensorValuesFloat.size(), inputShapeVision.data(), inputShapeVision.size());
    std::vector<Ort::Value> outputTensors;
    for(int i = 0; i < 3; i++){
      outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputVision[i].data(), outputVision[i].size(),
        outputShapeVision[i].data(), outputShapeVision[i].size()));
    }
    if(terminating){
      preprocessingEnd();
      return false;
    }
    runOptionsEncoder.UnsetTerminate();
    visionEncoder->Run(runOptionsEncoder,
      ptrInputNamesVision.data(),  &inputTensor, 1,
      ptrOutputNamesVision.data(), outputTensors.data(), outputTensors.size());
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    preprocessingEnd();
    return false;
  }
  preprocessingEnd();
  return true;
}

void Sam3Tracker::preprocessingStart(){
  preprocessing = true;
}

void Sam3Tracker::preprocessingEnd(){
  preprocessing = false;
  terminating = false;
}

void Sam3Tracker::setPointsLabels(const std::vector<cv::Point2f>& points, const std::vector<int> &labels, std::vector<float> *inputPointValues, std::vector<int64_t> *inputLabelValues){
  for(int i = 0; i < points.size(); i++){
    cv::Point2f point = points[i];
    (*inputPointValues).push_back(point.x);
    (*inputPointValues).push_back(point.y);
    (*inputLabelValues).push_back(labels[i]);
  }
}

void Sam3Tracker::setDecorderTensorsEmbeddings(std::vector<Ort::Value> *inputTensors){
  for(int i = 0; i < 3; i++){
    (*inputTensors).push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputVision[i].data(), outputVision[i].size(), outputShapeVision[i].data(), outputShapeVision[i].size()));
  }
}

void Sam3Tracker::setDecorderTensorsPointsLabels(std::vector<float> &inputPointValues, std::vector<int64_t> &inputLabelValues, int numPoints, std::vector<Ort::Value> *inputTensors){
  std::vector<int64_t> inputPointShape = {1, 1, numPoints, 2};
  std::vector<int64_t> inputLabelShape = {1, 1, numPoints};
  (*inputTensors).push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputPointValues.data(), 2 * numPoints, inputPointShape.data(), inputPointShape.size()));
  (*inputTensors).push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, inputLabelValues.data(), numPoints, inputLabelShape.data(), inputLabelShape.size()));
}

void Sam3Tracker::setDecorderTensorsMaskInput(const size_t maskInputSize, float *maskInputValues, float *hasMaskValues, std::vector<float> &previousMaskInputValues, std::vector<Ort::Value> *inputTensors){
  cv::Size maskSize = getMaskSize();
  std::vector<int64_t> maskInputShape = {1, 1, maskSize.height, maskSize.width},
  hasMaskInputShape = {1};
  if(hasMaskValues[0] == 1){
    (*inputTensors).push_back(Ort::Value::CreateTensor<float>(memoryInfo, previousMaskInputValues.data(), maskInputSize, maskInputShape.data(), maskInputShape.size()));
  }else{
    (*inputTensors).push_back(Ort::Value::CreateTensor<float>(memoryInfo, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
  }
  (*inputTensors).push_back(Ort::Value::CreateTensor<float>(memoryInfo, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
}

cv::Mat Sam3Tracker::getMask(std::vector<float> &inputPointValues, std::vector<int64_t> &inputLabelValues, const cv::Size &imageSize, int previousMaskIdx, bool isNextGetMask){
  std::vector<Ort::Value> inputTensors;
  setDecorderTensorsEmbeddings(&inputTensors);
  int numPoints = (int)inputLabelValues.size();
  setDecorderTensorsPointsLabels(inputPointValues, inputLabelValues, numPoints, &inputTensors);
  cv::Size maskSize = getMaskSize();
  const size_t maskInputSize = maskSize.height * maskSize.width;
  std::vector<float> previousMaskInputValues;
  resizePreviousMasks(previousMaskIdx);
  float maskInputValues[maskInputSize];
  memset(maskInputValues, 0, sizeof(maskInputValues));
  float hasMaskValues[] = {0};
  if(isNextGetMask){
  }else if(previousMaskIdx >= 0){
    hasMaskValues[0] = 1;
    previousMaskInputValues = previousMasks[previousMaskIdx];
  }
  setDecorderTensorsMaskInput(maskInputSize, maskInputValues, hasMaskValues, previousMaskInputValues, &inputTensors);
  try{
    Ort::RunOptions runOptionsDecoder;
    auto outputTensors = decoder->Run(runOptionsEncoder,
      ptrInputNamesDecoder.data(), inputTensors.data(), inputTensors.size(),
      ptrOutputNamesDecoder.data(), ptrOutputNamesDecoder.size());
    for(int i = 0; i < 3; i++){
      auto values = outputTensors[i].GetTensorMutableData<float>();
      outputShapeDecoder[i] = outputTensors[i].GetTensorTypeAndShapeInfo().GetShape();
      outputDecoder[i].assign(values, values + getShapeSize(outputShapeDecoder[i]));
      printShape(outputShapeDecoder[i]);
    }
    float iouScore = outputDecoder[1][0];
    float objectLogit = outputDecoder[2][0];
    objectLogit = 1 / (1 + exp(-objectLogit));
    std::cout<<iouScore<<std::endl;
    std::cout<<objectLogit<<std::endl;
    cv::Mat maskf((int)outputShapeDecoder[0][3], (int)outputShapeDecoder[0][4], CV_32F, outputDecoder[0].data());
    cv::Mat maskResized;
    cv::resize(maskf, maskResized, imageSize, 0, 0, cv::INTER_LINEAR);
    cv::Mat mask(imageSize.height, imageSize.width, CV_8UC1, cv::Scalar(0));
    mask.setTo(255, maskResized > 0);
    previousMasks.push_back(outputDecoder[0]);
    return mask;
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
  }
  cv::Mat mask;
  return mask;
}
