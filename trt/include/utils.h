#pragma once

#include <dirent.h>
#include <fstream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <vector>
#include <cstring>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>

#include "logging.h"

using namespace nvinfer1;

static Logger gLogger;

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s") {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
    } else if (std::string(argv[1]) == "-d") {
        engine = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}


void serialize_engine_onnx(unsigned int max_batchsize, std::string& wts_name, std::string& engine_name){
    // auto tt = initLibNvInferPlugins(&gLogger, "");
    
    // build
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);

    //config
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    
    // network
    auto batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(batch);
    
    // onnx 创建网络
    auto parse = nvonnxparser::createParser(*network, gLogger);
    auto parsed = parse->parseFromFile(wts_name.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    assert(parsed);
    
    // TODO ...
    builder->setMaxBatchSize(max_batchsize);
    config->setMaxWorkspaceSize(16 * (1 << 40));

    nvinfer1::IHostMemory *engine = builder->buildSerializedNetwork(*network, *config);

    assert(engine);
    
    // save engine
    std::ofstream engine_file(engine_name, std::ios::binary);
    assert(engine_file.is_open() && "Failed open engine file");
    engine_file.write((char *)engine->data(), engine->size());
    engine_file.close();

    // delete
    engine->destroy();
    config->destroy();
    network->destroy();
    parse->destroy();
    builder->destroy();
}


void deserialize_engine(std::string &engine_name, nvinfer1::IRuntime** runtime, nvinfer1::ICudaEngine** engine, nvinfer1::IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char *serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = nvinfer1::createInferRuntime(gLogger);
    assert(*runtime);
    initLibNvInferPlugins(&gLogger, "");
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
    // 检查engine
    assert(engine->getNbBindings() == 3);

    const int image_index = engine->getBindingIndex("image");
    const int bev_embed_index = engine->getBindingIndex("bev_embed");
    const int seg_preds_index = engine->getBindingIndex("seg_preds");

    assert(image_index == 0);
    assert(seg_preds_index == 2);

    // Create GPU buffers on device
    // 申请device内存
    // CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    // CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));
    // *cpu_output_buffer = new float[kBatchSize * kOutputSize];

    // 获取模型输入尺寸并分配GPU内存
    nvinfer1::Dims input_dim = engine->getBindingDimensions(image_index);
    int input_size = 1;
    for (int j = 0; j < input_dim.nbDims; ++j) {
        input_size *= input_dim.d[j];
    }
    cudaMalloc((void**)gpu_input_buffer, input_size * sizeof(float));

    // 获取模型输出尺寸并分配GPU内存
    nvinfer1::Dims output_dim = engine->getBindingDimensions(seg_preds_index);
    int output_size = 1;
    for (int j = 0; j < output_dim.nbDims; ++j) {
        output_size *= output_dim.d[j];
    }
    cudaMalloc((void**)gpu_output_buffer, output_size * sizeof(float));

    // 给模型输出数据分配相应的CPU内存
    *cpu_output_buffer = new float[output_size];
}
