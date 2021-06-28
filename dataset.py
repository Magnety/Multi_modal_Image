import os
import argparse
import random
import shutil
from shutil import copyfile
from misc import printProgressBar


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config):

    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)

    filenames = os.listdir(config.origin_data_path)
    data_list = []
    GT_list = []

    for filename in filenames:
        data_list.append(filename)
        """ext = os.path.splitext(filename)[-1]
        if ext =='.jpg':
            filename = filename.split('_')[-1][:-len('.jpg')]
            data_list.append('ISIC_'+filename+'.jpg')
            GT_list.append('ISIC_'+filename+'_segmentation.png')"""

    num_total = len(data_list)
    num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_valid = int((config.valid_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_test = num_total - num_train - num_valid

    print('\nNum of train set : ',num_train)
    print('\nNum of valid set : ',num_valid)
    print('\nNum of test set : ',num_test)

    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train):
        idx = Arange.pop()
        src = config.origin_data_path+'/'+ data_list[idx]+'/img'
        jpg_names = os.listdir(src)
        for jpg_name in jpg_names:
            img_src = config.origin_data_path+'/'+ data_list[idx]+'/img'+'/'+jpg_name
            img_dst = config.train_path+'/'+data_list[idx]+'_'+jpg_name
            copyfile(img_src, img_dst)
            mask_src = config.origin_data_path+'/'+ data_list[idx]+'/mask'+'/'+jpg_name
            mask_dst = config.train_GT_path+'/'+data_list[idx]+'_'+jpg_name
            copyfile(mask_src, mask_dst)

        printProgressBar(i + 1, num_train, prefix = 'Producing train set:', suffix = 'Complete', length = 50)
        

    for i in range(num_valid):
        idx = Arange.pop()
        src = config.origin_data_path + '/' + data_list[idx] + '/img'
        jpg_names = os.listdir(src)
        for jpg_name in jpg_names:
            img_src = config.origin_data_path + '/' + data_list[idx] + '/img' + '/' + jpg_name
            img_dst = config.valid_path + '/' + data_list[idx] + '_' + jpg_name
            copyfile(img_src, img_dst)
            mask_src = config.origin_data_path + '/' + data_list[idx] + '/mask' + '/' + jpg_name
            mask_dst = config.valid_GT_path + '/' + data_list[idx] + '_' + jpg_name
            copyfile(mask_src, mask_dst)

        printProgressBar(i + 1, num_valid, prefix = 'Producing valid set:', suffix = 'Complete', length = 50)

    for i in range(num_test):
        idx = Arange.pop()
        src = config.origin_data_path + '/' + data_list[idx] + '/img'
        jpg_names = os.listdir(src)
        for jpg_name in jpg_names:
            img_src = config.origin_data_path + '/' + data_list[idx] + '/img' + '/' + jpg_name
            img_dst = config.test_path + '/' + data_list[idx] + '_' + jpg_name
            copyfile(img_src, img_dst)
            mask_src = config.origin_data_path + '/' + data_list[idx] + '/mask' + '/' + jpg_name
            mask_dst = config.test_GT_path + '/' + data_list[idx] + '_' + jpg_name
            copyfile(mask_src, mask_dst)


        printProgressBar(i + 1, num_test, prefix = 'Producing test set:', suffix = 'Complete', length = 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # data path
    parser.add_argument('--origin_data_path', type=str, default=r'G:\Thoyroid\ThyroidProject\Data\finallcroplabeledjpg\total\sort_by_patient')
    parser.add_argument('--origin_GT_path', type=str, default='../ISIC/dataset/ISIC2018_Task1_Training_GroundTruth')
    
    parser.add_argument('--train_path', type=str, default='./dataset/Thyroid/train/')
    parser.add_argument('--train_GT_path', type=str, default='./dataset/Thyroid/train_GT/')
    parser.add_argument('--valid_path', type=str, default='./dataset/Thyroid/valid/')
    parser.add_argument('--valid_GT_path', type=str, default='./dataset/Thyroid/valid_GT/')
    parser.add_argument('--test_path', type=str, default='./dataset/Thyroid/test/')
    parser.add_argument('--test_GT_path', type=str, default='./dataset/Thyroid/test_GT/')

    config = parser.parse_args()
    print(config)
    main(config)