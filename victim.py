import torch
from torch import nn
import torchvision
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import torch.nn.functional as F
import numpy as np
import math

my_device = torch.device('cuda:0')

'''
通过model预测x，输出logits
保持输入类型和输出类型一致（numpy的array / torch的tensor）
'''
def predict(x, model, batch_size, device):
    if isinstance(x, np.ndarray):
        batch_amount = math.ceil(x.shape[0] / batch_size)
        batch_logits = []
        with torch.no_grad():
            for i in range(batch_amount):
                x_now = torch.as_tensor(x[i*batch_size : (i+1)*batch_size], device=device)
                batch_logits.append(model(x_now).detach().cpu().numpy())
        logits = np.vstack(batch_logits)
        return logits
    else:
        return model(x)


class Model(nn.Module):
    def __init__(self, device=my_device, batch_size=500):
        self.cnn = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT).to(device).eval()
        self.device = device
        self.batch_size = batch_size

    def forward(self, x):
        return predict(x=x, model=self.cnn, batch_size=self.batch_size, device=self.device)
