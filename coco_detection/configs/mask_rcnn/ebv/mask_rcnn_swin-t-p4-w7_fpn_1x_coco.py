import pickle as pkl

_base_ = "../../swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py"

work_dir = './work_dirs/ebv_det/mask_rcnn_swin-t-p4-w7_fpn_1x_coco'
ebv_dict = pkl.load(open('configs/mask_rcnn/ebv/eq_81_81.pkl', 'rb'))
ebv_dim = ebv_dict.size(1)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type="EBVBBoxHead",
            t=0.07,
            ebv_dict=ebv_dict,
            ebv_dim=ebv_dim
        ),
        mask_head=dict(
            type='EBVMaskHead',
            t=0.07,
            ebv_dict=ebv_dict,
            ebv_dim=ebv_dim
        )
    )
)
dataset_type = 'CocoDataset'
data_root = '/data1/dataset/coco/'
data = dict(
    #samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'))