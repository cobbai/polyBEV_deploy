cmake_minimum_required(VERSION 3.12)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(tensorrt_ops LANGUAGES CXX CUDA)
#find_package(CUDA REQUIRED)
include(FindCUDAToolkit)

#execute_process(COMMAND nvcc --include-path /usr/local/cuda-11.4/targets/aarch64-linux/include --library-path /usr/local/cuda-11.4/targets/aarch64-linux/lib -lcuda ${CMAKE_CURRENT_SOURCE_DIR}/cudaComputeVersion.cu -o /tmp/cudaComputeVersion)
#execute_process(COMMAND /tmp/cudaComputeVersion
#        RESULT_VARIABLE CUDA_RETURN_CODE
#        OUTPUT_VARIABLE ARCH)
#execute_process(COMMAND rm /tmp/cudaComputeVersion)

if(NOT CMAKE_TENSORRT_PATH)
    message( "Need: -DCMAKE_TENSORRT_PATH=/path/to/TensorRT")

endif()

include_directories(common)
include_directories(/usr/local/cuda/include)
#include_directories(/usr/local/include)

message(STATUS "Linking TensorRT directory: ${CMAKE_TENSORRT_PATH}")
#link_directories(${CMAKE_TENSORRT_PATH}/lib)
#include_directories(${CMAKE_TENSORRT_PATH}/include)
link_directories(/opt/lib-target/aarch64-linux-gnu)
include_directories(/opt/include-target/aarch64-linux-gnu)

file(GLOB_RECURSE TENSORRT_OPS_SRCS common/*.cpp common/*.cu plugin/*/*.cu plugin/*/*.cpp)

set(SHARED_TARGET tensorrt_ops)
add_library(${SHARED_TARGET} SHARED ${TENSORRT_OPS_SRCS})
#target_compile_options(${SHARED_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_87>)

target_link_libraries(${SHARED_TARGET} PUBLIC cudnn nvinfer cublas)
set_target_properties(${SHARED_TARGET} PROPERTIES CUDA_ARCHITECTURES 87)

install(
        TARGETS ${SHARED_TARGET}
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
)
