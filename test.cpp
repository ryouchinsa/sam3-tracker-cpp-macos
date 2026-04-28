#include <gflags/gflags.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include "sam3_tracker.h"

DEFINE_string(vision_encoder, "sam3/vision-encoder.onnx", "Path to the viion encoder model");
DEFINE_string(decoder, "sam3/decoder.onnx", "Path to the decoder model");
DEFINE_string(points, "", "Point prompt");
DEFINE_string(points_second, "", "Point prompt after the first point prompt");
DEFINE_string(image, "david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg", "Path to the image");
DEFINE_string(device, "cpu", "cpu or cuda:0(1,2,3...)");

int main(int argc, char** argv) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  Sam3Tracker sam3Tracker;
  std::chrono::steady_clock::time_point begin, end, begin_total, end_total; 
  std::cout<<"loadModel started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  bool successLoadModel = sam3Tracker.loadModel(FLAGS_vision_encoder, FLAGS_decoder, std::thread::hardware_concurrency(), FLAGS_device);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successLoadModel){
    std::cout<<"loadModel error"<<std::endl;
    return 1;
  }
  begin_total = std::chrono::steady_clock::now();
  std::cout<<"resize image started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  cv::Mat image = cv::imread(FLAGS_image, cv::IMREAD_COLOR);
  cv::Size imageSize = cv::Size(image.cols, image.rows);
  cv::Size inputSize = sam3Tracker.getInputSize();
  cv::resize(image, image, inputSize);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  std::cout<<"preprocessImage started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  bool successPreprocessImage = sam3Tracker.preprocessImage(image);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successPreprocessImage){
    std::cout<<"preprocessImage error"<<std::endl;
    return 1;
  }
  std::cout<<"getMask started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  int previousMaskIdx = -1;
  bool isNextGetMask = true;
  std::vector<float> inputPointValues;
  std::vector<int64_t> inputLabelValues;
  cv::Mat mask;
  auto [points, labels] = parse_point_prompts(FLAGS_points, imageSize, inputSize);
  sam3Tracker.setPointsLabels(points, labels, &inputPointValues, &inputLabelValues);
  mask = sam3Tracker.getMask(inputPointValues, inputLabelValues, imageSize, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask_point1.png", mask);
  inputPointValues.resize(0);
  inputLabelValues.resize(0);
  end = std::chrono::steady_clock::now();
  if(!FLAGS_points_second.empty()){
    begin = std::chrono::steady_clock::now();
    isNextGetMask = false;
    auto [points, labels] = parse_point_prompts(FLAGS_points_second, imageSize, inputSize);
    sam3Tracker.setPointsLabels(points, labels, &inputPointValues, &inputLabelValues);
    mask = sam3Tracker.getMask(inputPointValues, inputLabelValues, imageSize, previousMaskIdx, isNextGetMask);
    previousMaskIdx++;
    cv::imwrite("mask_point1_then_point2.png", mask);
    inputPointValues.resize(0);
    inputLabelValues.resize(0);
  }
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  end_total = std::chrono::steady_clock::now();
  std::cout << "predict sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end_total - begin_total).count()) / 1000000.0 <<std::endl;
  return 0;
}
