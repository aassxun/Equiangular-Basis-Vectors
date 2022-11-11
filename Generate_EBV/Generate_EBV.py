import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import pickle as pkl
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda')

#任意两个基向量夹角cos绝对值的最大值
#The maximum value of the absolute cosine value of the angle between any two basis vectors
threshold = 0.002

#类别数量，也可以理解为需要生成基向量的数量
#Number of categories, which can also be interpreted as the number of basis vectors N that need to be generated, num_cls >= N
num_cls = 1000

#基向量的维度
#Dimension for basis vectors
dim = 1000

#由于显存不够，所以需要切片优化。每次切片的大小。 slice_size<num_cls
#Slicing optimization is required due to insufficient memory
slice_size = 130   

#优化的step数量，达到threshold会立即退出
#Optimize step numbers
step = 100000        
lr = 1e-3 #learning rate
save_name = 'eq_1000_1000.pkl'  #pkl_name: eq_dim_numcls


def main():
    dtype = torch.float32

    basis_vec = nn.Parameter(F.normalize(torch.randn((num_cls, dim), dtype=dtype, device=device)))
    optim = torch.optim.SGD([basis_vec], lr=lr)
    pbar = tqdm(range(step), total=step)
    for _ in pbar:
        basis_vec.data.copy_(F.normalize(basis_vec.data))
        mx = threshold
        for i in range(math.ceil(num_cls / slice_size)):
            start = slice_size * i
            end = min(num_cls, slice_size * (i + 1))
            e = F.one_hot(torch.arange(start, end, device=device), num_cls)
            m = (basis_vec[start:end] @ basis_vec.T).abs() - e
            mx = max(mx, m.max().item())

            loss = F.relu_(m - threshold).sum()
            loss.backward()

        if mx <= threshold + 0.0001:
            pkl.dump(basis_vec.data, open(save_name, 'bw'))
            return
        optim.step()
        optim.zero_grad()
        pbar.set_description(f'{mx:.4f}')


if __name__ == '__main__':
    if not os.path.exists(save_name):
        seed = 42
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        main()
    else:
        basis_vec = pkl.load(open(save_name, 'rb'))
        print(basis_vec.shape)
        device = torch.device('cuda')
        m_max = 0.
        m_min = 1.
        ada = 0.
        for j in tqdm(range(num_cls)):
            m = (basis_vec[j] @ basis_vec.T).abs() - F.one_hot(torch.tensor(j, device=device), num_classes=num_cls)
            ada += m.sum().item()/(num_cls - 1)
            if m.max().cpu() > m_max:
                m_max = m.max().cpu()
            if m.min().cpu() < m_min:
                m_min = m.min().cpu()
        ada /= num_cls
        print(ada)
        print(m_max, m_min)
