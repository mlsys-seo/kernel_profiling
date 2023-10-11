import types

import torch.nn as nn
import torch
import torch.optim as optim
# import torchvision.models as models
# from transformers import AutoTokenizer
# from transformers import BertModel

from util import *
import my_models as models
import argparse 

def get_forward_pre_hook(label):
    def forward_pre_hook(m, input):
        torch.cuda.nvtx.range_push(f'f_{label}')
    return forward_pre_hook

def get_forward_post_hook(label):
    def forward_post_hook(m, input, output):
        torch.cuda.nvtx.range_pop()
    return forward_post_hook

def get_backward_pre_hook(label):
    def backward_pre_hook(m, input):
        torch.cuda.nvtx.range_push(f'b_{label}')
    return backward_pre_hook

def get_backward_post_hook(label):
    def backward_post_hook(m, input, output):
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
            m.register_forward_pre_hook(get_forward_pre_hook(str(m)))
            m.register_forward_hook(get_forward_post_hook(str(m)))
            m.register_full_backward_pre_hook(get_backward_pre_hook(str(m)))
            m.register_full_backward_hook(get_backward_post_hook(str(m)))
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

monkeypatch_func_to_module()
stream = torch.cuda.Stream()

model = eval(f"models.{model_name}()").to("cuda")
inputs = get_data_by_name("imagenet", batch_size)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), 0.1, capturable=True)

print("============================")
if args.do_train == "True":
    print(f"train")
else:
    print("infer")
print(f"model: {model_name}")
print(f"batch_size: {batch_size}")
print("============================\n\n")

if args.do_train == "True":
    # train
    train_warmup(stream, model, inputs, criterion, optimizer, iter_num=2)
    traversal_all_layers(model)
    train_warmup_update_nvtx(stream, model, inputs, criterion, optimizer, iter_num=1)

else:
    # infer
    infer_warmup(stream, model, inputs, iter_num=2)
    traversal_all_layers(model)
    infer_warmup(stream, model, inputs, iter_num=1)
