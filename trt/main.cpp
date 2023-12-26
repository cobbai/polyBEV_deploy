#include <iostream>
#include <fstream>
#include <cassert>

#include "cuda_utils.h"
#include "utils.h"
#include "preprocess.h"


void inference(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output) {
    context.enqueueV2(gpu_buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], 1*4*650*400 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}


int main(int argc, char** argv) {

	cudaSetDevice(0);

    std::string wts_name = "";
    std::string engine_name = "";

    if (!parse_args(argc, argv, wts_name, engine_name)) {
        std::cerr << "arguments not right!" << std::endl;
        return -1;
    }
    
    // if (!wts_name.empty()) {
    //     serialize_engine_onnx(1, wts_name, engine_name);
    //     return 0;
    // }
	
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);

    // 创建 stream 流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // buffer
    float* gpu_buffers[2];
    float* cpu_output_buffer = nullptr;
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    // 获取数据 + preprocess
    cv::Mat frame_3030 = cv::imread("/workspace/BEVFormer_tensorrt/data/out_123/images/30_30/2488462.800000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat frame_4065 = cv::imread("/workspace/BEVFormer_tensorrt/data/out_123/images/40_65/2488462.800000.png", cv::IMREAD_GRAYSCALE);

    cuda_image_preprocess(frame_3030, frame_4065, gpu_buffers[0], stream);

    // inference
    auto start = std::chrono::system_clock::now();
    inference(*context, stream, (void**)gpu_buffers, cpu_output_buffer);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


	return 0;
}