import argparse

parser = argparse.ArgumentParser(description="End2End Image Compression Settings")


parser.add_argument('--seed', type=float, default=1, help='Set random seed for reproducibility')
parser.add_argument('--image_size', type=int, nargs=2, default=(256, 256), help='Size of the patches to be cropped (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: %(default)s)')

parser.add_argument('--train_img_dataset', default="/data/zpp/NIC/dataset/flicker", type=str, help='Training dataset') 
parser.add_argument('--train_real_dataset', default="/data/zpp/NIC/dataset/SSID_path", type=str, help='Training dataset') 

parser.add_argument('--test_dataset_de', default="./", type=str, help='Testing dataset:noisy image')
parser.add_argument('--test_dataset_gt', default="./", type=str, help='Testing dataset:clean image')

parser.add_argument('--model', default="MainCodec", type=str, help='')
parser.add_argument('--label_str', default="basic", type=str, help='')
parser.add_argument('--mode', default="train", type=str, help='mode of model')
parser.add_argument('--ckpt', default="", type=str, help='checkpoint')
parser.add_argument('--restore', action='store_true', default=False, help='Restore model')

parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs (default: %(default)s)')
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float, help='Learning rate (default: %(default)s)')
parser.add_argument('-n', '--num-workers', type=int, default=3, help='Dataloaders threads (default: %(default)s)')
parser.add_argument('--lmbda', dest='lmbda', type=int, default=1, help='Bit-rate distortion parameter (default: %(default)s), [1-3-6-9]')
parser.add_argument('--test-batch-size', type=int, default=64, help='Test batch size (default: %(default)s)')
parser.add_argument('--aux-learning-rate', default=1e-3, help='Auxiliary loss learning rate (default: %(default)s)')
parser.add_argument('--cuda', action='store_false', default=True, help='Use cuda')
parser.add_argument('--save', action='store_false', default=True, help='Save model to disk')
parser.add_argument('--fix_d', action='store_true', default=False)

parser.add_argument('--print_num', type=int, default=100, help='print interval')
parser.add_argument('--clip_max_norm', default=1.0, type=float, help='=0.1, gradient clipping max norm') 

args = parser.parse_args()
