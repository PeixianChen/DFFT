## Efficient Decoder-free  Object Detection with Transformers
![image](https://github.com/Anonymous-px/ID2445_DFFT/blob/main/DFFT_wholenet.jpg)

This code is based on [mmdetection](https://github.com/open-mmlab/mmdetection)

## Results 
| Backbone | box mAP | GFLOPs | 
| :---: | :---: | :---: | :---: | 
| DFFT_{NANO} | 42.8 | 42 |
| DFFT_{TINY} | 43.5 | 57 |
| DFFT_{SMALL} | 44.5 | 62 |
| DFFT_{MEDIUM} | 45.7 | 67 | 
| DFFT_{LARGE} | 46.0 | 101 |

## Models
[ImageNet Pretrained](https://drive.google.com/drive/folders/1_uOAf6wvGhsIsPlHfQ635jY3SVgyZ2cu?usp=sharing)

[MS COCO Detection](https://drive.google.com/drive/folders/17ZQ57eu11beaHIR9oN-yI_CkAmOupGQo?usp=sharing)

# Runs
```bash
python3 setup.py develop && ./tools/dist_train.sh configs/dfft/dfft.py [num_gpus] \
--cfg-options model.pretrained=pretrained.pth
```
