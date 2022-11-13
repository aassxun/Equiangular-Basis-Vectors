_base_ = './upernet_r18_512x512_160k_ade20k.py'

work_dir = './work_dirs/ebv_seg/upernet_r18_512x512_160k_ade20k_0.14'

model = dict(
    decode_head=dict(t=0.14),
    auxiliary_head=dict(t=0.14),
    #test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)