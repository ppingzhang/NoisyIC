import os
import glob
from PIL import Image


def SIDD_patch_load(dataset_dir):
    """Load SIDD patches

    :param dataset_dir: _description_
    :type dataset_dir: _type_
    :return: _description_
    :rtype: _type_
    """ 
    gt_list = glob.glob(dataset_dir + '/*_gt.png')
    
    de_list = []
    for gt_path in gt_list:
        de_list.append(gt_path.replace("_gt.png", ".png"))

    return gt_list, de_list


def load_train_dataset(dataset_dir):
    """load training data

    :param dataset_dir: the image directory
    :type dataset_dir: str
    :return: image list
    :rtype: list
    """
    imagelist = []
    if 'sidd' in dataset_dir:
        for parent, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                # noisy_XX.jpg 
                if 'noisy_' in filename.lower():
                    if filename.lower().endswith(
                            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                        imagelist.append(os.path.join(parent, filename))
    else:
        # Note: you can change the rule to find your images！
        for parent, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                if filename.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    imagelist.append(os.path.join(parent, filename))
    
    return imagelist

'''
def load_train_dataset(dataname_list):
    """load training data

    :param dataname_list: e.g., "flicker+div2k"
    :type dataname_list: str
    :return: image list
    :rtype: list
    """
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
                        # noisy_XX.jpg 
                        if 'noisy_' in filename.lower():
                            if filename.lower().endswith(
                                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                                imagelist.append(os.path.join(parent, filename))
            else:
                # Note: you can change the rule to find your images！
                for parent, dirnames, filenames in os.walk(dataset_dir):
                    for filename in filenames:
                        if filename.lower().endswith(
                                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                            imagelist.append(os.path.join(parent, filename))
        else:
            raise Exception("[!] {dataset} not exitd".format(dataset=dataset_dir))
    
    return imagelist
'''
