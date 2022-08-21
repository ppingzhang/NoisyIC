



**1. Train on your dataset**

You can train on your dataset through chaning the path in "./data/dataset_load.py".

If you do not use SSID for training, you need modify the "./data/dataset_noise_mix.py".

**2. run the training code**

python main.py --mode=train --train_dataset='flicker' --model=MainCodec --lmbda=1 #[1, 5, 20, 50]

**2. run the testing code**

python3 main.py --mode=test --model=MainCodec --test_dataset_gt=./test_dataset_path/ --ckpt=''

