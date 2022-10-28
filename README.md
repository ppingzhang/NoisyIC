NoisyIC

Official Pytorch Implementation for "Learning-based Compression for Noisy Images in the Wild", TCSVT 2022

## Installation
git clone https://github.com/ppingzhang/NoisyIC.git

conda create -n NoisyIC python=3.8 
conda activate NoisyIC
pip install -r requirement.txt

Note: torch, GPU and CUDA version need match!

## Dataset
###How to get the dataset:

Flickr3k: [download1](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) or [download2](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

SSID: [download](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)

```
the structure of image path
├── flickr3k
│   ├── 9964318083_6199f7ee20_b.jpg
│   ├── 9964619774_c10a0480df_b.jpg
│   ├── ...
├── SSID
│   ├── img0000.png
│   ├── img0000_gt.png
│   ├── img0001.png
│   ├── img0001_gt.png
├── ...
```

> Note: Your can originze your own dataset following above data structure. 
> Or you can create your own rule to find the imags in the "data/dataset_load.py"


## Run the training code
Pretrained model can be download through this [link]()
``python main.py --mode=train --train_img_dataset='./dataset/flickr3k' --train_real_dataset='./dataset/SSID' --model=MainCodec --lmbda=1`` #[1, 5, 20, 50]

## run the testing code
``python3 main.py --mode=test --model=MainCodec --test_dataset_de=/data/zpp/NIC/data/test_dataset/Gaussian_Noise/Kodak/15/    --test_dataset_gt=/data/zpp/NIC/data/KODAK_PNG/ --ckpt=''``

## test a single image
``python3 demo.py --ckpt=test --img_path="./ckpt.pth.tar" --img_save_path="./xx_decode.jpg" --bin_save_path="./xx_bitstream.bin" ``


## BibTeX
```
@ARTICLE{Learning2022Zhang,
  author={Zhang, Pingping and Wang, Meng and Chen, Baoliang and Lin, Rongqun and Wang, Xu and Wang, Shiqi and Kwong, Sam},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Learning-based Compression for Noisy Images in the Wild}, 
  year={2022},
  volume={},
  number={},
  pages={1-1}}
```