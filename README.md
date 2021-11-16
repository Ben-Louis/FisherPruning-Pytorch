# FisherPruning-Pytorch
An implementation of &lt;Group Fisher Pruning for Practical Network Compression> based on pytorch and mmcv 

---


### Main Functions

- Pruning for fully-convolutional structures, 
  such as one-stage detectors; (copied from the [official code](https://github.com/jshilong/FisherPruning))
  
- Pruning for networks combining convolutional layers and fully-connected layers, such as faster-RCNN and ResNet;

- Pruning for networks which involve group convolutions, such as ResNeXt and RegNet.

### Usage

#### Requirements

```text
torch
torchvision
mmcv / mmcv-full
mmcls 
mmdet 
```
#### Compatibility
This code is tested with 

```text
pytorch=1.3
torchvision=0.4
cudatoolkit=10.0
mmcv-full==1.3.14
mmcls=0.16 
mmdet=2.17
```

and 

```text
pytorch=1.8
torchvision=0.9
cudatoolkit=11.1
mmcv==1.3.16
mmcls=0.16 
mmdet=2.17
```

#### Data

Download [ImageNet](https://image-net.org/download.php) and [COCO](https://cocodataset.org/), 
then extract them and organize the folders as 

  ```
  - detection
    |- tools
    |- configs
    |- data
    |   |- coco
    |   |   |- train2017
    |   |   |- val2017
    |   |   |- test2017
    |   |   |- annotations
    |
  - classification
    |- tools
    |- configs
    |- data
    |   |- imagenet
    |   |   |- train
    |   |   |- val
    |   |   |- test 
    |   |   |- meta
    |
  - ...
  ```

#### Commands

e.g. Classification
```bash
cd classification
```
1. Pruning
   ```bash
   # single GPU
   python tools/train.py configs/xxx_pruning.py --gpus=1
   # multi GPUs (e.g. 4 GPUs)
   python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/xxx_pruning.py --launch pytorch
   ```
   
2. Fine-tune
   
   In the config file, modify the `deploy_from` to the pruned model, and modify the `samples_per_gpu` to 256/#GPUs. Then
   ```bash
   # single GPU
   python tools/train.py configs/xxx_finetune.py --gpus=1
   # multi GPUs (e.g. 4 GPUs)
   python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/xxx_finetune.py --launch pytorch
   ```
   
3. Test

   In the config file, add the attribute `load_from` to the finetuned model. Then
   ```bash
   python tools/test.py configs/xxx_finetune.py --metrics=accuracy
   ```
   
The commands for pruning and finetuning of detection models are similar to that of classification models. 
Instructions will be added soon.

## Acknowledgments

My project acknowledges the official code [FisherPruning](https://github.com/jshilong/FisherPruning).