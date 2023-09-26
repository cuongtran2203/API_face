import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
from core.models.retinaface import RetinaFace
from core.config import cfg_mnet
from core.prior_box  import PriorBox
from core.ultils import decode,decode_landm,py_cpu_nms
from core.face_detect import *
#####Load model########
cfg=cfg_mnet
model_path="src/core/weights/mobilenet0.25_Final.pth"
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, model_path, True)
net.eval()
input_names = ["input"]
output_names = ["bbox", "confidence", "landmark"]
inputs = torch.randn(1, 3,640,640)

dynamic_axes = {"input": {0: "None", 2: "None", 3: "None"}, "bbox": {1: "None"}, "confidence": {1: "None"}, "landmark": {1: "None"}}

torch_out = torch.onnx.export(net, inputs,"Face_Detector.onnx", export_params=True, verbose=False,
                                input_names=input_names, output_names=output_names, opset_version=11,
                                dynamic_axes=dynamic_axes)