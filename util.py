import types

import torch
import torch.optim as optim
import torchvision.models as models

def get_model_by_name(model_name):
    return eval(f"models.{model_name}()").to("cuda")

def get_data_by_name(data_name, batch_size):
    if(data_name == "imagenet"):
        return torch.randn([batch_size, 3, 224, 224]).to("cuda")
class Event_record():
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
    
def train_warmup(stream, model, input, criterion, optimizer, iter_num=4):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(iter_num):
            outputs = model(input)
            loss = criterion(outputs, outputs)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(stream)
    
def train_capture(stream, model, input, criterion, optimizer, iter_num=4):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outputs = model(input)
            loss = criterion(outputs, outputs)
            loss.backward()
            optimizer.step()
        torch.cuda.current_stream().wait_stream(stream)
    return graph

def infer_warmup(stream, model, input, iter_num=4):
    with torch.no_grad():
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(iter_num):
                outputs = model(input)
        torch.cuda.current_stream().wait_stream(stream)
    return outputs

def infer_capture(stream, model, input, iter_num=4, event=None):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.no_grad():
                if event != None:
                    with event:
                        model(input)
                else:
                    model(input)
        torch.cuda.current_stream().wait_stream(stream)
    return graph

def idx_to_str(idx):
    str_list = ["forward1", "backward1", "update1", "forward2", "backward2", "update2"]
    return str_list[idx]

def get_start_section(train_events:list, infer_event):
    for idx in range(len(train_events)):
        if train_events[idx].start.elapsed_time(infer_event.start) < 0:
            start_section = idx - 1
            return idx_to_str(start_section)

def get_end_section(train_events, infer_event):
    for idx in range(len(train_events)):
        if infer_event.end.elapsed_time(train_events[idx].end) > 0:
            end_section = idx
            return idx_to_str(end_section)
        
###monkey patch###
import torch.nn.functional as F
def copy_func(f, name=None):
    return types.FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)

class Adaptive_avg_pool2d_module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = copy_func(F.adaptive_avg_pool2d)
    def forward(self, x: torch.Tensor, output_size) -> torch.Tensor:
        # print("adaptive_avg_pool2d : start!!")
        return self.func(x, output_size)

class Relu_module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = copy_func(F.relu)
    def forward(self, x: torch.Tensor, inplace=False) -> torch.Tensor:
        # print("relu : start!!")
        return self.func(x, inplace=False)

class Silu_module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = copy_func(F.selu)
    def forward(self, x: torch.Tensor, inplace=False) -> torch.Tensor:
        return self.func(x, inplace=False)
    
class Hardswish_module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = copy_func(F.hardswish)
    def forward(self, x: torch.Tensor, inplace=False) -> torch.Tensor:
        # print("hardswish : start!!")
        return self.func(x, inplace=False)

class Dropout_module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = copy_func(F.dropout)
    def forward(self, x, p=0.5, training=True, inplace=False) -> torch.Tensor:
        # print("hardswish : start!!")
        return self.func(x, p, training, inplace=False)

class Layernorm_module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = copy_func(F.layer_norm)
    def forward(self, input, normalized_shape, weight=None, bias=None, eps=1e-05) -> torch.Tensor:
        # print("layernorm : start!!")
        return self.func(input, normalized_shape, weight, bias, eps)

class Add_module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, input, another_input) -> torch.Tensor:
        return torch.add(input, another_input)
    
class Cat_module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, tensors, dim=0, *, out=None) -> torch.Tensor:
        return torch.cat(tensors, dim, out)
    
# class Concat_module(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.func = copy_func(torch._C._VariableFunctions.cat)
#     def forward(self, tensors, dim=0,*,out=None) -> torch.Tensor:
#         print("Concat : start!!")
#         return self.func(tensors, dim=0,out=None)
import pickle
def monkeypatch_func_to_module():
    F.layer_norm = Layernorm_module()
    F.relu = Relu_module()
    F.silu = Silu_module()
    F.adaptive_avg_pool2d = Adaptive_avg_pool2d_module()
    F.hardswish = Hardswish_module()
    F.dropout = Dropout_module()
    # torch._C._VariableFunctions.cat = Concat_module()    
    
    
#sleep
import cuda_graph_utils.sleep_ops
class Sleep_controller():
    def __init__(self):
        self.placeholder = torch.tensor([0]).cuda()
    def sleep_change(self, delay):
        self.placeholder[0] = int(delay * 1000000 * 1.54)
    def sleep(self):
        cuda_graph_utils.sleep_ops.sleep(self.placeholder)