cmake_minimum_required(VERSION 3.10)

# LANGUAGES CXX CUDA 添加c++ 和 cuda 编译器
project("trt" LANGUAGES CXX CUDA)  # 

set(CMAKE_BUILD_TYPE "Debug")

# add tensorRT
# include_directories("/usr/include/x86_64-linux-gnu")  # 将指定目录添加到编译器的头文件搜索路径之下
# link_directories("/usr/lib/x86_64-linux-gnu")  # 添加需要链接的库文件目录
include_directories("/TensorRT/include")
link_directories("/TensorRT/build/out")

# add CUDA
include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

# opencv
set(OpenCV_DIR "/workspace/opencv-4.6.0/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加自定义库
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)
message(STATUS "自定义头文件路径 ${CMAKE_SOURCE_DIR}/include")
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)


# 将源代码添加到此项目的可执行文件。
add_executable(${PROJECT_NAME} main.cpp ${SRCS})
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} nvinfer_plugin)
target_link_libraries(${PROJECT_NAME} nvonnxparser)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

