cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(Sam3TrackerCPP)
find_package(OpenCV CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)

add_library(sam3_tracker_cpp_lib SHARED sam3_tracker.h sam3_tracker.cpp util.h util.cpp)
if (APPLE)
  set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.dylib)
else()
  set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
endif()

target_include_directories(
  sam3_tracker_cpp_lib PUBLIC 
  ${ONNXRUNTIME_ROOT_DIR}/include
)

target_link_libraries(
  sam3_tracker_cpp_lib PUBLIC
  ${onnxruntime_lib}
  ${OpenCV_LIBS}
  gflags
)

add_executable(sam3_tracker_cpp_test test.cpp)
target_link_libraries(
  sam3_tracker_cpp_test PRIVATE
  sam3_tracker_cpp_lib
)