#include "preprocess.h"
#include "cuda_utils.h"



void cuda_image_preprocess(cv::Mat& frame1, cv::Mat& frame2, float* dst, cudaStream_t stream) {
  // padding
  cv::copyMakeBorder(frame1, frame1, 300, 50, 50, 50, cv::BORDER_CONSTANT, cv::Scalar(0));
  
  // dilate
  cv::dilate(frame1, frame1, cv::Mat());
  cv::dilate(frame2, frame2, cv::Mat());

  // input [1, 2, 1, 650, 400]
  float* input_blob = new float[1 * 2 * 1 * 650 * 400];
  
  // HCW --> CHW
  const int channels = frame1.channels();
  const int width = frame1.cols;
  const int height = frame1.rows;
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        // C: c * width * height
        // H: h * width
        // W: w
        input_blob[c * width * height + h * width + w] =
            frame1.at<cv::Vec3b>(h, w)[c];
      }
    }
  }

  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        // C: c * width * height
        // H: h * width
        // W: w
        input_blob[1 * 650 * 400 + c * width * height + h * width + w] =
            frame2.at<cv::Vec3b>(h, w)[c];
      }
    }
  }

  // 拷贝输入数据
  cudaMemcpyAsync(dst, input_blob,  (1 * 2 * 1 * 650 * 400) * sizeof(float),
                  cudaMemcpyHostToDevice, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

