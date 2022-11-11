import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import presets
import transforms
from torch.utils.data.dataloader import default_collate
import torch

import sys
import timm
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.distributed as dist
import torch.multiprocessing as mp
# from torchvision import transforms
import time
import pandas as pd
from torch.nn.parallel import DataParallel
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply
import torchvision
import torch.nn.functional as F

CFG = {
    'root_dir': '/',
    'seed': 42,  # 719,42,68
    'resize_size': 256, #val
    'crop_size': 224, #train
    'epochs': 105, #100+5
    'warmup_epochs': 5,
    'train_bs': 1024,
    'valid_bs': 1024,
    'lr': 0.5,
    'weight_decay': 2e-5,
    'lr_warmup_decay': 0.01,
    'num_workers': 32,
    'accum_iter': 1,
    'verbose_step': 1,
    'device': 'cuda:0',
    'num_classes': 100,
    'model_name': 'resnet50', 
    'pkl_pth': 'eq_100_1000.pkl',
    'info': 'EBV_ResNet_dim100_SGD_epoch105', #log name
    'ifval': False,
    'model_path': 'EBV_ResNet_dim100_SGD_epoch105.pth'
}
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("./" + CFG['info'] + ".txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_img(path):
    # im_bgr = cv2.imread(path)
    # im_rgb = im_bgr[:, :, ::-1]
    # return im_rgb
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

    def __getitem__(self, index):
        image = get_img(self.images[index])
        if self.transforms is not None:
            image = self.transforms(image)
            # image = self.transforms(image=image)['image']
        return image, self.labels[index]

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, d, device, scheduler=None, schd_batch_update=False):
    model.train()
    
    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device)#.long()
        
        with autocast():
            image_preds = model(imgs)

            # loss = loss_fn(image_preds, pt_image_labels, ng_image_labels)
            loss = loss_fn((image_preds@d.t()/0.07), image_labels)
            scaler.scale(loss).backward()
            
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01
            
            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()
            
            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                
                pbar.set_description(description)
  
    if scheduler is not None and not schd_batch_update:
        scheduler.step()

def valid_one_epoch(epoch, model, loss_fn, val_loader, d, device, scheduler=None, schd_loss_update=False):
    model.eval()
    
    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=100)
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds = model(imgs)  # batch_size * 50
        
        image_preds = image_preds@d.t()#.abs()  #bs *200
        
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    ans = (image_preds_all == image_targets_all).mean()
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))
    logger.info(' Epoch: ' + str(epoch) + ' validation accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()
    return ans


def test_one_epoch(model, val_loader, d, device):
    model.eval()
    
    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=100)
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds = model(imgs)  # batch_size * 50
        
        image_preds = image_preds@d.t()#.abs()  #bs *200
        
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    ans = (image_preds_all == image_targets_all).mean()
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))

    return ans


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
    model = nn.DataParallel(model)
    model.to(device)

    train_dataset = ImageNetDataset(CFG['root_dir'], 'train', presets.ClassificationPresetTrain(
                crop_size=CFG['crop_size'],
                auto_augment_policy="ta_wide",
                random_erase_prob=0.1,
            ),)

    mixupcutmix = torchvision.transforms.RandomChoice([
        transforms.RandomMixup(num_classes=1000, p=1.0, alpha=0.2),
        transforms.RandomCutmix(num_classes=1000, p=1.0, alpha=1.0)
    ])
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))

    train_loader = DataLoader(train_dataset, 
                            batch_size=CFG['train_bs'], 
                            num_workers=CFG['num_workers'], 
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=collate_fn,
                            )
                            
    val_dataset = ImageNetDataset(CFG['root_dir'], 'val', presets.ClassificationPresetEval(
                crop_size=CFG['resize_size'], resize_size=CFG['resize_size']
            ))
    val_loader = DataLoader(val_dataset, CFG['valid_bs'], num_workers=CFG['num_workers'], pin_memory=True)

    scaler = GradScaler()

    optimizer = torch.optim.SGD(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'], momentum=0.9)
    # optimizer = torch.optim.Adam(parameters, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    # optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs'] - CFG['warmup_epochs']
    )
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=CFG['lr_warmup_decay'], total_iters=CFG['warmup_epochs']
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[CFG['warmup_epochs']]
    )

    loss_tr = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)


    d = pkl.load(open(CFG['pkl_pth'], 'rb')).data#.detach().cpu()
    d = F.normalize(d).to(device)

    
    if CFG['ifval'] == True:
        model.load_state_dict(torch.load(CFG['model_path']))
        with torch.no_grad():
            answer = test_one_epoch(model, val_loader, d, device)
        print(answer)
        logger.info('#Val Resize size = {}: {:.5f}'.format(CFG['resize_size'], answer))
        exit()

    best_answer = 0.0
    for epoch in range(CFG['epochs']):
        print(optimizer.param_groups[0]['lr'])
        train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, d, device, scheduler=scheduler,
                                schd_batch_update=False)
                
        answer = 0.0
        with torch.no_grad():
            # if (epoch<100 and epoch%10==9) or (epoch<130 and epoch>=100 and epoch%5==4) or epoch>=130:
            if epoch%1==0:
                answer = valid_one_epoch(epoch, model, loss_fn, val_loader, d, device, scheduler=None, schd_loss_update=False)
            if answer > best_answer:
                torch.save(model.state_dict(), CFG['info'] + '.pth'.format(epoch))
        if answer > best_answer:
            best_answer = answer
        # time.sleep(10)
        # torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
    del model, optimizer, train_loader, val_loader, scaler, scheduler
    print(best_answer)
    logger.info('BEST-TEST-ACC: ' + str(best_answer))
    torch.cuda.empty_cache()
