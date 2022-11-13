import pickle as pkl

_base_ = '../upernet_r50_512x512_80k_ade20k.py'

work_dir = './work_dirs/ebv_seg/upernet_r50_512x512_80k_ade20k_2'
ebv_dict = pkl.load(open('configs/upernet/ebv/eq_150_150.pkl', 'rb'))
ebv_dim = ebv_dict.size(1)
crop_size = (512, 512)

model = dict(
    decode_head=dict(
        type='UPerEBVHead',
        ebv_dim=150,
        t=0.07,
        ebv_dict=ebv_dict),
    auxiliary_head=dict(
        type='FCNEBVHead',
        ebv_dim=150,
        t=0.07,
        ebv_dict=ebv_dict),
    #test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

dataset_type = 'ADE20KDataset'
data_root = '/data1/dataset/ADEChallengeData2016/'
data = dict(
    #samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root),
    val=dict(
        type=dataset_type,
        data_root=data_root),
    test=dict(
        type=dataset_type,
        data_root=data_root))
