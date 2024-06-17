import warnings

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
                x_now = torch.as_tensor(x[i*batch_size : (i+1)*batch_size], device=device, dtype=torch.float32)
                batch_logits.append(model(x_now).detach().cpu().numpy())
        logits = np.vstack(batch_logits)
        return logits
    else:
        return model(x)


class Model(nn.Module):
    def __init__(self, device=my_device, batch_size=500, defense='None'):
        super(Model, self).__init__()
        self.cnn = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT).to(device).eval()
        self.arch = 'wide_resnet50_2'
        self.device = device
        self.batch_size = batch_size
        self.defense = defense

        # AAA parameters
        self.dev = 0.5
        self.attractor_interval = 6
        self.reverse_step = 0.7

        # RND parameters
        self.n_in = 0.03
        self.n_out = 0.3

    def forward(self, x):
        if self.defense == 'None':
            return predict(x=x, model=self.cnn, batch_size=self.batch_size, device=self.device)
        elif self.defense =='AAASine':
            return self.aaa_sine_forward(x)
        elif self.defense == 'inRND' or self.defense == 'outRND' or self.defense == 'inoutRND':
            return self.rnd_forward(x)
        else:
            warnings.warn('no such defense method')


    def aaa_sine_forward(self, x):
        with torch.no_grad():
            logits = predict(x=x, model=self.cnn, batch_size=self.batch_size, device=self.device)
            if isinstance(logits, np.ndarray):
                logits = torch.as_tensor(logits, device=self.device)
        logits_ori = logits.detach()

        # 每个logit中选择最大的两个。value就是这两个的值，index_ori就是这两个的下标，即类别。
        value, index_ori = torch.topk(logits_ori, k=2, dim=1)

        margin_ori = value[:, 0] - value[:, 1]
        attractor = ((margin_ori / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
        target = margin_ori - self.reverse_step * self.attractor_interval * torch.sin(
            (1 - 2 / self.attractor_interval * (margin_ori - attractor)) * torch.pi)
        gap_to_target = target - margin_ori
        logits_ori[torch.arange(logits_ori.shape[0]), index_ori[:, 0]] += gap_to_target

        logits_ret = logits_ori.detach().cpu()
        if isinstance(x, np.ndarray):
            logits_ret = logits_ret.numpy()
        return logits_ret


    def rnd_forward(self, x):
        if self.defense=='inRND' or self.defense=='inoutRND':
            noise_in = np.random.normal(scale=self.n_in, size=x.shape)
        else:
            noise_in = np.zeros(shape=x.shape)

        logits = predict(x=np.clip(x + noise_in, 0, 1), model=self.cnn, batch_size=self.batch_size, device=self.device)

        if self.defense=='outRND' or self.defense=='inoutRND':
            noise_out = np.random.normal(scale=self.n_out, size=logits.shape)
        else:
            noise_out = np.zeros(shape=logits.shape)

        return logits + noise_out














































