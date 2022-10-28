import glob

def data_load(de_img_path='', gt_img_path=''):
    
    # you can rewrite it according to your testdataset.
    data_list = glob.glob(gt_img_path+"*real.png")
    data_list.sort()
    noisy_data_list = []
    for ii in data_list:
        noisy_data_list.append(ii.replace("real", "mean"))
    return noisy_data_list, data_list

def data_load_same_name(de_img_path='', gt_img_path=''):
    
    # you can rewrite it according to your testdataset.
    data_list = glob.glob(gt_img_path+"/*.png")
    data_list.sort()
    noisy_data_list = glob.glob(gt_img_path+"/*.png")
    noisy_data_list.sort()
    return noisy_data_list, data_list