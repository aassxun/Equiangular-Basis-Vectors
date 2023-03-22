# Equiangular-Basis-Vectors

EBVs pretrain on ImageNet-1K can be download at: https://github.com/aassxun/Equiangular-Basis-Vectors/releases/download/untagged-72ef569681e789dd1bb8/EBV_ResNet_dim1000_SGD_epoch605.pth

Data for the 10w classes classification task can be download from: https://github.com/aassxun/Equiangular-Basis-Vectors/releases/download/10w_classes/10w_classes.zip  

It contains 120w images for training and 60w images for validation. (Resize to 224*224)

## Environment

Python 3.8.11  
Pytorch 1.11.0  
torchvision 0.12.0  
numpy 1.22.4
timm 0.6.11

## Generate EBVs

    Please refer to /Generate_EBV/Generate_EBV.py

## Experiments on ImageNet

    Training logs and code can be found at: /ImageNet_Validation_Experiment/ and /ImageNet_Ablation_Study/Log/
    
    The random seed for all the experiments is 42.

## Experiments on COCO

    All the code and training logs can be found at: /coco_detection/work_dirs/ebv_det/
    
    The random seeds for each experiment are exactly the same as the corresponding experimental settings in MMdetection.

## Experiments on ADE20K

    All the code and training logs can be found at: /ADE20k_segmentation/work_dirs/ebv_seg/
    
    The random seeds for each experiment are exactly the same as the corresponding experimental settings in MMSegmentation.
