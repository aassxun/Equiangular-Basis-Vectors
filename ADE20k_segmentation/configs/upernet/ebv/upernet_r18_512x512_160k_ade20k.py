_base_ = './upernet_r50_512x512_80k_ade20k.py'

work_dir = './work_dirs/ebv_seg/upernet_r18_512x512_160k_ade20k'

runner = dict(max_iters=160000)
checkpoint_config = dict(interval=16000)
evaluation = dict(interval=16000)
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150),
    auxiliary_head=dict(in_channels=256, num_classes=150))