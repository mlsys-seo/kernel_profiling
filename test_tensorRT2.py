import torch
from torchvision import models
import torch_tensorrt
import time
import numpy as np



def benchmark(model, input_shape=(1024, 1, 224, 224), dtype='fp32', nwarmup=3, nruns=10):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            print(i)
            features = model(input_data)
          
          
model = models.resnet50().cuda()
with torch.no_grad():
    data = torch.rand(128, 3, 224, 224).cuda()
    jit_model = torch.jit.trace(model, data)
    # torch.jit.save(jit_model, "test_resnet50.jit.pt")
          
# jit_model = torch.jit.load("test_resnet50.jit.pt")
print('load end!!!')

for i in range(5):
    output = jit_model(data)
print('jit model end')
trt_model_fp32 = torch_tensorrt.compile(jit_model, inputs=[torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float32)],
    enabled_precisions = torch.float32, # Run with FP32
    truncate_long_and_double=False
)
print('compile end!!')

benchmark(trt_model_fp32, input_shape=(128, 3, 224, 224), nruns=5)
