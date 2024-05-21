import os
import random
import numpy as np


def load_imagenet(model, amount, random_seed=0):
    arch = model.arch
    data_path = 'data/imagenet_%s_x_seed=%d.npy' % (arch, random_seed)
    label_path = 'data/imagenet_%s_y_seed=%d.npy' % (arch, random_seed)

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        with open('data/val.txt','r') as f:
            lines = f.read().split('\n')
        labels = {}
        for line in lines:
            if ' ' not in line:
                continue
            file, label = line.split(' ')
            labels[file] = int(label)
        data = []
        files = os.listdir('data/ILSVRC2012_img_val')
        label = np.zeros((amount, 1000), dtype=np.uint8)
        label_done = []
        random.seed(random_seed)

        for i in random.sample(range(len(files)), len(files)):
            file = files[i]
            val = labels[file]
            if val in label_done:
                continue





    else:
        x_test = np.load(data_path)
        y_test = np.load(label_path)
    return x_test[:amount], y_test[:amount]