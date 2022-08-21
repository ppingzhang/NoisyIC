import os
import glob
from PIL import Image


######################################### load training data ################################

def SIDD_patch_load():

    dataset_dir = '../dataset/SSID_path/'
    gt_list = glob.glob(dataset_dir + '*_gt.png')
    # print(gt_list)

    de_list = []
    for gt_path in gt_list:
        de_list.append(gt_path.replace("_gt.png", ".png"))
#
    return gt_list, de_list


def load_train_dataset(dataname_list):
    dataset_dict = {
        'flicker': "../dataset/flicker/",
        'div2k': "",
        'bsd500': "/",
        'coco': "",
        'live': "",
        'koniq': "",
        'ssid': "",
        'others':""
    }

    dataname_list = dataname_list.lower()
    dataset_list = dataname_list.split('+')
    imagelist = []
    for dataset in dataset_list:
        if dataset in dataset_dict:
            dataset_dir = dataset_dict[dataset]
            if dataset == 'sidd':
                for parent, dirnames, filenames in os.walk(dataset_dir):
                    for filename in filenames:
                        if 'noisy_' in filename.lower():
                            if filename.lower().endswith(
                                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                                imagelist.append(os.path.join(parent, filename))
            else:

                for parent, dirnames, filenames in os.walk(dataset_dir):
                    for filename in filenames:
                        if filename.lower().endswith(
                                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                            imagelist.append(os.path.join(parent, filename))
        else:
            raise Exception("[!] {dataset} not exitd".format(dataset=dataset_dir))
    
    return imagelist

