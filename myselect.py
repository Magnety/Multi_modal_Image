from PIL import Image
import os
import shutil
import numpy as np
gt_path = './dataset/Thyroid/train_GT/'
img_path = './dataset/Thyroid/train'
gt_names = os.listdir(gt_path)

if not os.path.isdir(gt_path):
    os.makedirs(gt_path)
if not os.path.isdir(img_path):
    os.makedirs(img_path)
for gt_name in gt_names:
    gt = Image.open(gt_path+'/'+gt_name)
    gt = np.copy(gt)
    if np.sum(gt)>0:
        shutil.move(gt_path+'/'+gt_name,gt_path+'/'+gt_name)
        shutil.move(img_path + '/' + gt_name, img_path + '/' + gt_name)
        #print(gt_name)
    #print(gt)



