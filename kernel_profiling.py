import types

import torch.nn as nn
import torch
import torch.optim as optim
# import torchvision.models as models
# from transformers import AutoTokenizer
# from transformers import BertModel

from util import *

import argparse 

def get_forward_pre_hook(event, label):
    def forward_pre_hook(m, input):
        torch.cuda.nvtx.range_push(f'f_{label}')
        event.record_start()
    return forward_pre_hook

def get_forward_post_hook(event, label):
    def forward_post_hook(m, input, output):
        event.record_end()
        torch.cuda.nvtx.range_pop()
    return forward_post_hook

def get_backward_pre_hook(event, label):
    def backward_pre_hook(m, input):
        torch.cuda.nvtx.range_push(f'b_{label}')
        event.record_start()
    return backward_pre_hook

def get_backward_post_hook(event, label):
    def backward_post_hook(m, input, output):
        event.record_end()
        torch.cuda.nvtx.range_pop()
    return backward_post_hook

parent_name = None
idx = 0
def traversal_all_layers(module):
    global parent_name
    global idx
    for block_name, m in module._modules.items():
        
        type_name = str(type(m))
        #layer
        if not isinstance(m, nn.Sequential) \
            and "torch.nn.modules" in type_name:
            forward_event = Event_record()
            backward_event = Event_record()
            m.register_forward_pre_hook(get_forward_pre_hook(forward_event, str(m)))
            m.register_forward_hook(get_forward_post_hook(forward_event, str(m)))
            m.register_full_backward_pre_hook(get_backward_pre_hook(backward_event, str(m)))
            m.register_full_backward_hook(get_backward_post_hook(backward_event, str(m)))
            forward_events.append([parent_name, str(m), forward_event])
            # backward_events.append(backward_event)
        else:
            if("densenet" in model_name):
                if("block" in str(block_name) or "trans" in str(block_name)):
                    parent_name = str(block_name)
            elif("resnet" in model_name):
                if("layer" in str(block_name)):
                    parent_name = str(block_name)
            elif("convnext" in model_name):
                if("CNBlock" in str(m)):
                    parent_name = "NULL"#str(m)
            elif("efficient" in model_name):
                name = str(m).split('\n')[0][:-1]
                if("Sequential" == name):
                    pass
                if("Conv2dNormActivation" == name):
                    idx = idx + 1
                    parent_name = name
                if("FusedMBConv" == name):
                    parent_name = name+str(idx)
            elif("mobile" in model_name):
                name = str(m).split('\n')[0][:-1]
                if "InvertedResidual" == name:
                    parent_name = name + block_name
            traversal_all_layers(m)
            
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--do_train')
args = parser.parse_args()

model_name = args.model_name.lower()
batch_size = args.batch_size

record = Event_record()
forward_events = []
backward_events = []
monkeypatch_func_to_module()
train_stream = torch.cuda.Stream()
infer_stream = torch.cuda.Stream()

# model = get_model_by_name(model_name)
import my_models as models
model = eval(f"models.{model_name}()").to("cuda")
traversal_all_layers(model)
inputs = get_data_by_name("imagenet", batch_size)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), 0.1, capturable=True)

if args.do_train == "True":
    # train
    train_warmup(train_stream, model, inputs, criterion, optimizer)
    graph = train_capture(train_stream, model, inputs, criterion, optimizer)
else:
    # infer
    infer_warmup(infer_stream, model, inputs)
    graph = infer_capture(infer_stream, model, inputs)

for _ in range(1):
    graph.replay()
torch.cuda.synchronize()

for parent, name, event in forward_events:
    print(f"{parent}/{name}/{event.get_time()}")