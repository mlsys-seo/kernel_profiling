import my_models as models
import torch


model = models.convnext_tiny().cuda()
# model = models.efficientnet_v2_l().cuda()
# model = models.shufflenet_v2_x0_5().cuda()


batch_size = 1
input = torch.randn(batch_size, 3, 224, 224).cuda()


output = model(input)