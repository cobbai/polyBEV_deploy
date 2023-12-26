#!/bin/bash

gpu_id=0
while getopts "d:" opt
do
  case $opt in
    d)
      gpu_id=$OPTARG
      ;;
    ?)
      echo "There is unrecognized parameter."
      exit 1
      ;;
  esac
done

echo "Running on the GPU: $gpu_id"

CUDA_VISIBLE_DEVICES=$gpu_id python tools/pth2onnx.py \
configs/bevformer/bevformer_small_trt_q.py \
checkpoints/pytorch/bevformer_small_epoch_24_ptq_max.pth \
--int8 \
--opset_version 13 \
--cuda
