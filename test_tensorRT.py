import torch
from torchvision import models
import torch_tensorrt

model = models.densenet121().cuda()
with torch.no_grad():
    data = torch.rand(128, 3, 224, 224).cuda()
    jit_model = torch.jit.trace(model, data)
    torch.jit.save(jit_model, "test_densenet121.jit.pt")


jit_model = torch.jit.load("test_densenet121.jit.pt")

# input = torch.rand(16, 3, 224, 224).cuda()

inputs = torch_tensorrt.Input(min_shape=[1, 3, 224, 224],
                              opt_shape=[16, 3, 224, 224],
                              max_shape=[64, 3, 224, 224],
                              dtype=torch.float32)

trt_model = torch_tensorrt.compile(
    jit_model,
    inputs=[torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float32)],
    enabled_precisions={torch.float32},  # Enable Float32
    truncate_long_and_double=True
)

trt_model()