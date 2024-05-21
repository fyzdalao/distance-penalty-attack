import os
import random
import numpy as np
import PIL.Image

def sample_imagenet(model, amount=2000, random_seed=0, need_right_prediction=False):
    with open('data/val.txt', 'r') as f:
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
    random.seed(random_seed)

    for i in random.sample(range(len(files)), len(files)):
        file = files[i]
        val = labels[file]
        img = np.array(PIL.Image.open(
            'data/ILSVRC2012_img_val' + '/' + file).convert('RGB').resize((224, 224))) \
                  .astype(np.float32).transpose((2, 0, 1)) / 255

        # 保证采样的所有数据都是预测正确的数据
        if need_right_prediction:
            prd = model(img[np.newaxis, ...]).argmax(1)
            if prd != val:
                continue

        label[len(data), val] = 1
        data.append(img)

        if len(data) == amount:
            break

    x_test = np.array(data)
    y_test = np.array(label)
    return x_test, y_test



def sample_imagenet_every_class(model, random_seed=0, need_right_prediction=True):
    arch = model.arch
    data_path = 'data/imagenetEvery_%s_x_seed=%d.npy' % (arch, random_seed)
    label_path = 'data/imagenetEvery_%s_y_seed=%d.npy' % (arch, random_seed)

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
        label = np.zeros((1000, 1000), dtype=np.uint8)
        label_done = []
        random.seed(random_seed)

        for i in random.sample(range(len(files)), len(files)):
            file = files[i]
            val = labels[file]
            if val in label_done:
                continue
            img = np.array(PIL.Image.open(
                'data/ILSVRC2012_img_val' + '/' + file).convert('RGB').resize((224, 224))) \
                      .astype(np.float32).transpose((2, 0, 1)) / 255

            #保证采样的所有数据都是预测正确的数据
            if need_right_prediction:
                prd = model(img[np.newaxis, ...]).argmax(1)
                if prd != val:
                    continue

            label[len(data), val] = 1
            data.append(img)
            label_done.append(val)
            print('selecting samples in different classes...', len(label_done), '/', 1000, end='\r')
            if len(label_done) == 1000:
                break
        x_test = np.array(data)
        y_test = np.array(label)
        np.save(data_path, x_test)
        np.save(label_path, y_test)
    else:
        x_test = np.load(data_path)
        y_test = np.load(label_path)
    return x_test, y_test


def margin_loss(y, logits):
    a = (logits * y)
    rest = logits - a
    margin = a.max(1) - rest.max(1)
    return margin




