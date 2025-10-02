cmake_minimum_required(VERSION 3.0.2)
project(yolo_detection)

# C++ standardı
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROS paketleri
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)


if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV 4.10.0 not found!")
endif()
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ONNX Runtime
set(ONNXRUNTIME_ROOT_DIR /usr/local/onnxruntime)
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_ROOT_DIR}/lib
)

if(NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime library not found!")
endif()

# catkin_package
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs cv_bridge
)

# Include dizinleri
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_ROOT_DIR}/include
)

# Çalıştırılabilir dosya
add_executable(yolov11_detection src/yolov11_detection.cpp)
target_link_libraries(yolov11_detection
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIBRARY}
)
