import sys
import time
import torch
import torchvision.models as models
import torch_tensorrt


def convert_to_tensorrt(model, dummy_input):
    model.eval()

    script_model = torch.jit.trace(model, dummy_input)
    min_shape = dummy_input[:int(dummy_input.size()[0]/2)].size()
    opt_shape = dummy_input.shape
    max_shape = torch.cat((dummy_input,dummy_input), 0).size()

    compile_spec = {
        "inputs": [
            torch_tensorrt.Input(min_shape=min_shape,
                                 opt_shape=opt_shape,
                                 max_shape=max_shape)
        ],
        "enabled_precisions": torch.float,
    }
    print("compile start...")
    trt_model = torch_tensorrt.compile(script_model, **compile_spec)
    print("compile end...")

    return trt_model


def get_model(model, is_train=False):
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

    if is_train:
        model.train()
    else:
        model.eval()

    return model


if __name__ == "__main__":
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])

    print(f"model : {model_name}")
    print(f"batch size : {batch_size}")

    model = get_model(model_name)
    dummy_input = torch.rand(batch_size, 3, 224, 224).cuda()
    print("Prepare a model...")

    # warming up for cuda graph capture
    for _ in range(3):
        output = model(dummy_input)
    print("Warm-up the model...")

    # run a tensorRT model
    trt_model = convert_to_tensorrt(model, dummy_input)
    print("Run a TensorRT model...")
    trt_output = trt_model(dummy_input)

    time.sleep(0.1)    


    for i in range(5):
        print(f"iter : {i+1}")
        trt_output = trt_model(dummy_input)
    
    # capture and run a cuda graph
    # g = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(g):
    #     o = model(dummy_input)

    # print("Run a CUDA Graph model...")
    # g.replay()
    # torch.cuda.synchronize()