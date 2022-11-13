_base_ = './upernet_r50_512x512_80k_ade20k.py'

work_dir = './work_dirs/ebv_seg/upernet_r101_512x512_160k_ade20k'

runner = dict(max_iters=160000)
checkpoint_config = dict(interval=16000)
evaluation = dict(interval=16000)
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
