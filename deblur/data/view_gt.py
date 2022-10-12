from os import listdir, path
import os
import numpy as np
import imageio
import cv2
import skimage
import math
import time
import glob
from scipy import ndimage
from scipy.io import savemat
import csv

test_gt_dir = 'C:\\Users\\user\\deblurring_dataset\\ours\\dataset_1248\\test_gt_crops\\'
test_gt_dir = 'Z:\\data\\deblurring_dataset\\test_gt_crops\\'
for path in glob.glob('{}*.npy'.format(test_gt_dir)):
    img_name = path.split('\\')[-1].split('.')[0]
    print(img_name)
    a = np.load(path)
    a = np.log2(1+a*5000)/np.log2(1+5000)
    for i in range(4):
        imageio.imwrite('gt_visualize\\{}_{}.png'.format(img_name,i),np.uint8(a[i]*255))