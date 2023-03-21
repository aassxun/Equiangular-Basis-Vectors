import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import presets
import transforms
from torch.utils.data.dataloader import default_collate
import torch
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle as pkl

import sys
import timm
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F
import time
from torchmetrics import Metric

CFG = {
    'root_dir': '',
    'seed': 42, 
    'resize_size': 256,
    'crop_size': 224,
    'valid_bs': 64,
    'num_workers': 8,
    'device': 'cuda',
    'num_classes': 1000,
    #'model_name': 'swin_tiny_patch4_window7_224',
    'model_name': 'resnet50',
    'pkl_pth': 'eq_1000_1000.pkl',
    'model_path': 'EBV_ResNet_1000_SGD_105.pth', 
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_img(path):
    img = Image.open(path).convert('RGB')
    return img

import pickle as pkl
class ImageNetDataset(Dataset):
    def __init__(self, root, part='train', transforms=None):
        self.part = part
        self.transforms = transforms
        self.images = []
        self.labels = []
        self.labels = []
        if part == 'train':
            mycsv = pd.read_csv('./imagenet_train.csv')
        else:
            mycsv = pd.read_csv('./imagenet_val.csv')
        for i in range(len(mycsv['image_id'])):
            self.images.append('/data1/dataset'+mycsv['image_id'][i][3:])
            self.labels.append(int(mycsv['label'][i]))

    def __len__(self):
        return len(self.labels)
        #return 128

    def __getitem__(self, index):
        image = get_img(self.images[index])
        if self.transforms is not None:
            image = self.transforms(image)
            # image = self.transforms(image=image)['image']
        return image, self.labels[index]


def gram(X):
    # ensure correct input shape
    X = X.view(X.size(0), -1)
    return X @ X.T


def centering_mat(n):
    H = torch.eye(n, device=CFG['device']) - torch.ones(n,n, device=CFG['device']) / n
    return H


def centered_gram(X):
    K = gram(X)
    m = K.shape[0]
    H = centering_mat(m)
    return H @ K @ H


def unbiased_hsic_xy(X,Y):
    n = X.shape[0]
    assert n > 3
    v_i = torch.ones(n,1, device=CFG['device'])
    K = centered_gram(X)
    L = centered_gram(Y)
    KL = K @ L
    iK = v_i.T @ K
    Li = L @ v_i
    iKi = iK @ v_i
    iLi = v_i.T @ Li

    a = torch.trace(KL)
    b = iKi * iLi / ((n-1)*(n-2))
    c = iK @ Li * 2 / (n-2)

    outv = (a + b - c) / (n*(n-3))
    return outv.long().item()


class MinibatchCKA(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("_xx", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_xy", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_yy", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, X, Y):
        self._xx += unbiased_hsic_xy(X,X)
        self._xy += unbiased_hsic_xy(X,Y)
        self._yy += unbiased_hsic_xy(Y,Y)
    def compute(self):
        xx, xy, yy = self._xx, self._xy, self._yy
        return xy / (torch.sqrt(xx) * torch.sqrt(yy))


class HookedCache:
    def __init__(self, model, target):
        self.model = model
        self.target = target
        
        self.clear()
        self._extract_target()
        self._register_hook()

    @property
    def value(self):
        return self._cache
    def clear(self):
        self._cache = None
    def _extract_target(self):
        for name, module in self.model.named_modules():
            if name == self.target:
                self._target = module
                return
    def _register_hook(self):
        def _hook(module, in_val, out_val):
             self._cache = out_val
        self._target.register_forward_hook(_hook)


def get_simmat_from_metrics(metrics):
    vals = []
    for i, ckas in enumerate(metrics):
        for j, cka in enumerate(ckas):
            z = cka.compute().item()
            vals.append((i,j,z))

    sim_mat = torch.zeros(i+1,j+1)
    for i,j,z in vals:
        sim_mat[i,j] = z
    
    sim_mat = torch.where(torch.eye(sim_mat.size(0), dtype=torch.long).bool(), torch.ones_like(sim_mat), sim_mat)
    sim_mat = torch.where(torch.isnan(sim_mat)|torch.isinf(sim_mat), torch.zeros_like(sim_mat), sim_mat)
    sim_mat = sim_mat**4
    return sim_mat.numpy()


def make_pairwise_metrics(mod1_hooks, mod2_hooks):
    metrics = []
    for i_ in mod1_hooks:
        metrics.append([])
        for j_ in mod2_hooks:
            metrics[-1].append(MinibatchCKA().to(CFG['device']))
    return metrics


def update_metrics(mod1_hooks, mod2_hooks, metrics):
    for i, hook1 in enumerate(mod1_hooks):
        for j, hook2 in enumerate(mod2_hooks):
            cka = metrics[i][j]
            X,Y = hook1.value, hook2.value
            cka.update(X,Y)


class Net(nn.Module):
    def __init__(self, model_name="resnet50"):
        super(Net, self).__init__()
        self.backbone = timm.create_model(model_name=CFG['model_name'], num_classes=CFG['num_classes'], pretrained=False)

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        return x

if __name__ == '__main__':
        
    seed_everything(CFG['seed'])

    device = torch.device(CFG['device'])

    model = Net()
    model.to(device)
    d = torch.load(CFG['model_path'], map_location='cpu')
    state_dict = OrderedDict()
    for k, v in d.items():
        state_dict[k.replace('module.', '')] = v
    model.load_state_dict(state_dict)
    model = model.backbone
    
    #d = torch.load('SwinT_baseline.pth', map_location='cpu')
    d = torch.load('ResNet_baseline.pth', map_location='cpu')
    state_dict = OrderedDict()
    for k, v in d.items():
        state_dict[k.replace('module.', '')] = v
    model2 = timm.create_model(model_name=CFG['model_name'], num_classes=CFG['num_classes'], pretrained=True).to(device)
    model2.load_state_dict(state_dict)

    val_dataset = ImageNetDataset(CFG['root_dir'], 'val', presets.ClassificationPresetEval(
                crop_size=CFG['crop_size'], resize_size=CFG['resize_size']
            ))
    val_loader = DataLoader(val_dataset, CFG['valid_bs'], num_workers=CFG['num_workers'], pin_memory=False, drop_last=True)

    #ResNet
    names = ['conv1']
    for i, layer in enumerate([3, 4, 6, 3]):
        for j in range(layer):
            names.append(f'layer{i+1}.{j}.conv1')
            names.append(f'layer{i+1}.{j}.conv2')
            names.append(f'layer{i+1}.{j}.conv3')
    names.append('fc')

    #Swin
    '''names = ['patch_embed']
    for i, layer in enumerate([2, 2, 6, 2]):
        for j in range(layer):
            names.append(f'layers.{i}.blocks.{j}.norm1')
            names.append(f'layers.{i}.blocks.{j}.norm2')
            names.append(f'layers.{i}.blocks.{j}.mlp')
    names.append('head')'''

    hooks = []
    hooks2 = []
    for name in names:
        hook = HookedCache(model, name)
        hooks.append(hook)
        hook = HookedCache(model2, name)
        hooks2.append(hook)
    metrics1 = make_pairwise_metrics(hooks, hooks)
    metrics2 = make_pairwise_metrics(hooks, hooks2)
    metrics3 = make_pairwise_metrics(hooks2, hooks2)

    model.eval()
    model2.eval()
    with torch.inference_mode():
        for it, (data, label) in tqdm(enumerate(val_loader), total=len(val_loader), ncols=100):
            data = data.to(CFG['device'])
            output = model(data)
            output2 = model2(data)

            update_metrics(hooks, hooks, metrics1)
            update_metrics(hooks, hooks2, metrics2)
            update_metrics(hooks2, hooks2, metrics3)

            for hook in hooks:
                hook.clear()
            for hook in hooks2:
                hook.clear()
    sim_mat1 = get_simmat_from_metrics(metrics1)
    sim_mat2 = get_simmat_from_metrics(metrics2)
    sim_mat3 = get_simmat_from_metrics(metrics3)
    pkl.dump((sim_mat1, sim_mat2, sim_mat3), open('cache2.pkl', 'wb'))

    '''size_t = 20
    size_l = 16
    plt.imshow(sim_mat1, cmap='plasma')
    plt.xlabel('Layers EBV', fontdict={'size': size_l})
    plt.ylabel('Layers EBV', fontdict={'size': size_l})
    plt.title('EBV vs EBV', fontdict={'size': size_t})
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig('1.png')
    
    plt.imshow(sim_mat2, cmap='plasma')
    plt.xlabel('Layers FC', fontdict={'size': size_l})
    plt.ylabel('Layers EBV', fontdict={'size': size_l})
    plt.title('EBV vs FC', fontdict={'size': size_t})
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig('2.png')
    
    plt.imshow(sim_mat3, cmap='plasma')
    plt.xlabel('Layers FC', fontdict={'size': size_l})
    plt.ylabel('Layers FC', fontdict={'size': size_l})
    plt.title('FC vs FC', fontdict={'size': size_t})
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig('3.png')
    #plt.show()
    '''