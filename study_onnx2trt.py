import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np
import os, requests


# Download checkpoint and test image
urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
    'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png']
names = ['srcnn.pth', 'face.png']
for url, name in zip(urls, names):
    if not os.path.exists(name):
        open(name, 'wb').write(requests.get(url).content)

class DynamicTRTResize(torch.autograd.Function):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def symbolic(g, input, size_tensor, align_corners = False):
        """Symbolic function for creating onnx op."""
        return g.op(
            'Test::DynamicTRTResize',
            input,
            size_tensor,
            align_corners_i=align_corners)

    @staticmethod
    def forward(g, input, size_tensor, align_corners = False):
        """Run forward."""
        size = [size_tensor.size(-2), size_tensor.size(-1)]
        return interpolate(
            input, size=size, mode='bicubic', align_corners=align_corners)
    

class StrangeSuperResolutionNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, size_tensor):
        x = DynamicTRTResize.apply(x, size_tensor)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model():
    torch_model = StrangeSuperResolutionNet()

    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


model = init_torch_model()
factor = torch.rand([1, 1, 512, 512], dtype=torch.float)

input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img), factor).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("face_torch.png", torch_output)

x = torch.randn(1, 3, 256, 256)

dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'factor': {
            0: 'batch1',
            2: 'height1',
            3: 'width1'
        },
        'output': {
            0: 'batch2',
            2: 'height2',
            3: 'width2'
        },
    }

with torch.no_grad():
    torch.onnx.export(
        model, (x, factor),
        "srcnn3.onnx",
        opset_version=11,
        input_names=['input', 'factor'],
        output_names=['output'],
        dynamic_axes=dynamic_axes)
    

# 目前生成的 srcnn3.onnx 文件缺少DynamicTRTResize算子，用c++实现在 
# /mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/dynamic_resize 目录下。
# 主要继承自 nvinfer1::IPluginV2DynamicExt 和 nvinfer1::IPluginCreator
# 方法实现好后，重新build一下mmdeploy生成 libmmdeploy_tensorrt_ops.so


# 测试 python api 是否添加成功
import tensorrt as trt
from mmdeploy.backend.tensorrt import load_tensorrt_plugin
import ctypes
# load_tensorrt_plugin()
# ctypes.CDLL("/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so")
ctypes.CDLL("/workspace/BEVFormer_tensorrt/TensorRT/build/libtensorrt_ops.so")

def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]
print(get_plugin_names())
print(len(get_plugin_names()))

# 
from mmdeploy.backend.tensorrt.utils import from_onnx

engine = from_onnx(
    'srcnn3.onnx',
    'srcnn3',
    input_shapes=dict(
        input=dict(
            min_shape=[1, 3, 256, 256],
            opt_shape=[1, 3, 256, 256],
            max_shape=[1, 3, 256, 256]),
        factor=dict(
            min_shape=[1, 1, 256, 256],
            opt_shape=[1, 1, 512, 512],
            max_shape=[1, 1, 1024, 1024])))

from mmdeploy.backend.tensorrt import TRTWrapper

trt_model = TRTWrapper('srcnn3.engine', ['output'])

factor = torch.rand([1, 1, 768, 768], dtype=torch.float)
trt_output = trt_model.forward(dict(input=x.cuda(), factor=factor.cuda()))
torch_output = model.forward(x, factor)
assert np.allclose(
    trt_output['output'].cpu().numpy(),
    torch_output.cpu().detach(),
    rtol=1e-3,
    atol=1e-5)