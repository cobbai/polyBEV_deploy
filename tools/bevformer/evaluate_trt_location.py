import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import argparse
import torch
import mmcv
import copy
import numpy as np
from mmcv import Config
from mmdeploy.backend.tensorrt import load_tensorrt_plugin
import cv2
import sys

sys.path.append(".")
from det2trt.utils.tensorrt import (
    get_logger,
    create_engine_context,
    allocate_buffers,
    do_inference,
)
from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("trt_model", help="checkpoint file")
    args = parser.parse_args()
    return args


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def show_seg(labels):
    
    PALETTE = [[255, 255, 255], [220, 20, 60], [0, 0, 128], [0, 100, 0],
               [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
               [192, 192, 128], [64, 64, 128], [128, 0, 192], [192, 0, 64]]
    mask_colors = np.array(PALETTE)
    img = np.zeros((650, 400, 3))

    for index, mask_ in enumerate(labels):
        color_mask = mask_colors[index]
        mask_ = mask_.astype(bool)
        img[mask_] = color_mask

    # 这里需要水平翻转，因为这样才可以保证与在图像坐标系下，与习惯相同

    # img = np.flip(img, axis=0)
    # 可视化小车
    # car_img = np.where(car_img == [0, 0, 0], [255, 255, 255], car_img)
    # car_img = cv2.resize(car_img.astype(np.uint8), (16, 26))
    # img[300 - 13: 300 + 13, img.shape[1] // 2 - 8: img.shape[1] // 2 + 8, :] = car_img

    return img


def main():
    args = parse_args()
    load_tensorrt_plugin()

    trt_model = args.trt_model
    config_file = args.config
    TRT_LOGGER = get_logger(trt.Logger.INTERNAL_ERROR)

    engine, context = create_engine_context(trt_model, TRT_LOGGER)

    stream = cuda.Stream()

    config = Config.fromfile(config_file)
    if hasattr(config, "plugin"):
        import importlib
        import sys

        sys.path.append(".")
        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    output_shapes = config.output_shapes
    input_shapes = config.input_shapes
    default_shapes = config.default_shapes

    for key in default_shapes:
        if key in locals():
            raise RuntimeError(f"Variable {key} has been defined.")
        locals()[key] = default_shapes[key]

    import pickle
    with open("data/temp_data.pickle", "rb") as r:
        loader = pickle.load(r)

    prog_bar = mmcv.ProgressBar(len(loader))

    for data in loader:
        token = data["token"]
        img = data["img"].numpy()
        can_bus = data["can_bus"]

        output_shapes_ = {}
        for key in output_shapes.keys():
            shape = output_shapes[key][:]
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            output_shapes_[key] = shape

        input_shapes_ = {}
        for key in input_shapes.keys():
            shape = input_shapes[key][:]
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            input_shapes_[key] = shape

        # 申请内存空间
        inputs, outputs, bindings = allocate_buffers(
            engine, context, input_shapes=input_shapes_, output_shapes=output_shapes_
        )

        for inp in inputs:
            if inp.name == "image":
                inp.host = img.reshape(-1).astype(np.float32)
            # elif inp.name == "prev_bev":
            #     inp.host = prev_bev.reshape(-1).astype(np.float32)
            # elif inp.name == "use_prev_bev":
            #     inp.host = use_prev_bev.reshape(-1).astype(np.float32)
            elif inp.name == "can_bus":
                inp.host = can_bus.reshape(-1).astype(np.float32)
            # elif inp.name == "lidar2img":
            #     inp.host = lidar2img.reshape(-1).astype(np.float32)
            else:
                raise RuntimeError(f"Cannot find input name {inp.name}.")

        trt_outputs, t = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        trt_outputs = {
            out.name: out.host.reshape(*output_shapes_[out.name]) for out in trt_outputs
        }

        prev_bev = trt_outputs.pop("bev_embed")

        semantic = trt_outputs["seg_preds"]
        semantic = onehot_encoding(torch.tensor(semantic)).cpu().numpy()
        mmcv.imwrite(show_seg(semantic.squeeze()), "data/" + str(token) + ".png")
        

        for _ in range(len(img)):
            prog_bar.update()


if __name__ == "__main__":
    main()
