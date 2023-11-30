import torch
import torch.nn as nn
from torchvision import models

import argparse
import random
import pdb

class Event_record_custom():
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
    def record_start(self):
        self.start.record(check_graph_time=torch.cuda.is_current_stream_capturing())
    def record_end(self):
        self.end.record(check_graph_time=torch.cuda.is_current_stream_capturing())
    def __enter__(self):
        self.record_start()
    def __exit__(self, exc_type, exc_value, traceback):
        self.record_end()
    def get_time(self):
        return self.start.elapsed_time(self.end)
    
    
class Event_record():
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
    def record_start(self):
        self.start.record()
    def record_end(self):
        self.end.record()
    def __enter__(self):
        self.record_start()
    def __exit__(self, exc_type, exc_value, traceback):
        self.record_end()
    def get_time(self):
        return self.start.elapsed_time(self.end)



def model_init(model):
    if 'densenet' in model:
        if model == 'densenet121':
            model = models.densenet121().cuda()
        elif model == 'densenet201':
            model = models.densenet201().cuda()
    elif 'mobilenet' in model:
        if model == 'mobilenetv30.5':
            model = models.mobilenet_v3_large(width_mult=0.5).cuda()
        elif model == 'mobilenetv31.0':
            model = models.mobilenet_v3_large(width_mult=1.0).cuda()
    elif 'resnet' in model:
        if model == 'resnet18':
            model = models.resnet18().cuda()
        elif model == 'resnet50':
            model = models.resnet50().cuda()
        
    elif 'efficientnet' in model:
        if model == 'efficientnet_v2_l':
            model = models.efficientnet_v2_l().cuda()
        elif model == 'efficientnet_v2_m':
            model = models.efficientnet_v2_m().cuda()
    
    elif 'convnext' in model:
        if model == 'convnext_tiny':
            model = models.convnext_tiny().cuda()
        elif model == 'convnext_small':
            model = models.convnext_small().cuda()
        elif model == 'convnext_base':
            model = models.convnext_base().cuda()
        elif model == 'convnext_large':
            model = models.convnext_large().cuda()
        
    elif 'shufflenet' in model:
        if model == 'shufflenet_v2_x0_5':
            model = models.shufflenet_v2_x0_5().cuda()
        elif model == 'shufflenet_v2_x1_0':
            model = models.shufflenet_v2_x1_0().cuda()
        elif model == 'shufflenet_v2_x1_5':
            model = models.shufflenet_v2_x1_5().cuda()
        elif model == 'shufflenet_v2_x2_0':
            model = models.shufflenet_v2_x2_0().cuda()
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--do_train_or_infer')
parser.add_argument('--mps_percent')
args = parser.parse_args()

model = args.model
batch_size = args.batch_size
do_train_or_infer = args.do_train_or_infer

# autonvtx(model)
model = model_init(model)
dummy_input = torch.randn(batch_size, 3, 224, 224).cuda()

try:
    print("====================")
    print(f"model: {args.model}")
    print(f"batch_size: {args.batch_size}")
    print(f"task: {do_train_or_infer}")
    print("====================")

    # CUDA Graph Capture
    stream1 = torch.cuda.Stream()
    
    total_record = Event_record()
    
    # Train
    if do_train_or_infer == "train":
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), 0.1, capturable=True)
        # WarmUp
        stream1.wait_stream(torch.cuda.current_stream())
        for _ in range(3):
            with torch.cuda.stream(stream1):
                output = model(dummy_input)
                loss = criterion(output, output)
                loss.backward()
                optimizer.step()
        torch.cuda.current_stream().wait_stream(stream1)
        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stream1.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream1):
                output = model(dummy_input)
                loss = criterion(output, output)
                loss.backward()
                optimizer.step()
            torch.cuda.current_stream().wait_stream(stream1)

    # Inference
    elif do_train_or_infer == "infer":
        # WarmUp
        with torch.no_grad():
            stream1.wait_stream(torch.cuda.current_stream())
            for _ in range(3):
                with torch.cuda.stream(stream1):
                    output = model(dummy_input)
        torch.cuda.current_stream().wait_stream(stream1)
        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad():
                stream1.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream1):
                    output = model(dummy_input)
            torch.cuda.current_stream().wait_stream(stream1)



    # Replay
    duration = []
    
    for _ in range(10):
        with total_record:
            graph.replay()
        torch.cuda.synchronize()
        duration.append(total_record.get_time())
    avg_duration = sum(duration) / len(duration)
        

    if do_train_or_infer == "train":
        with open("./data/train_duration_with_mps.txt", "a") as f:
            f.write(f"{args.model},{args.batch_size},{args.mps_percent}%,{avg_duration}\n")
    elif do_train_or_infer == "infer":
        with open("./data/infer_duration_with_mps.txt", "a") as f:
            f.write(f"{args.model},{args.batch_size},{args.mps_percent}%,{avg_duration}\n")
              

except Exception as e:
    print(e)
    with open("model_single_error.txt", "a") as f:
        f.write(f"{torch.cuda.get_device_name(0)}_{do_train_or_infer}_{args.model}_{args.batch_size}\n")
        f.write(f"{e}\n\n")
