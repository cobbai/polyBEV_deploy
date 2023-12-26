#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

void cuda_image_preprocess(cv::Mat& frame1, cv::Mat& frame2, float* dst, cudaStream_t stream);
