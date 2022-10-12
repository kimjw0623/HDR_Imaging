from os import listdir, path
import os
from tkinter import image_names
import numpy as np
import imageio
import cv2
import skimage
import math
import time
import glob
import rawpy

# Generate Raw Bayer Images from HDR Images

gamma = 2.2
gain = 200

def dng2ldrs(hdrs): 
    # hdrs: shape=(4,H,W,3)
    # ldrs: shape=(4,H,W,3)
    hdrs = hdrs * (pow(2, 14)-1)
    ldrs = np.clip(hdrs, 0, (pow(2, 14)-1))
    ldrs = np.around(ldrs-0.5)/(pow(2, 14)-1)
    ldrs = np.clip(ldrs,0,1)
    return ldrs

def ldrs2raw(ldrs):
    exposure,h,w,_ = ldrs.shape
    ldrs = ldrs[:,:h//4*4,:w//4*4,:]
    raw_bayer = np.zeros((h//4*4,w//4*4))

    for level in range(exposure):
        offset_h = level//2
        offset_w = level%2
        raw_bayer[offset_h::4,offset_w::4] = ldrs[level,offset_h::4,offset_w::4,0]
        raw_bayer[2+offset_h::4,offset_w::4] = ldrs[level,2+offset_h::4,offset_w::4,1]
        raw_bayer[offset_h::4,2+offset_w::4] = ldrs[level,offset_h::4,2+offset_w::4,1]
        raw_bayer[2+offset_h::4,2+offset_w::4] = ldrs[level,2+offset_h::4,2+offset_w::4,2]

    return raw_bayer

def raw2ldrs(raw,file_dir):
    h,w = raw.shape
    ldr = np.zeros((h // 2, w // 2))
    print(ldr.shape)
    
    for level in range(4):
        offset_h = level//2
        offset_w = level%2
        ldr[::2,::2] = raw[offset_h::4,offset_w::4]
        ldr[1::2,::2] = raw[2+offset_h::4,offset_w::4]
        ldr[::2,1::2] = raw[offset_h::4,2+offset_w::4]
        ldr[1::2,1::2] = raw[2+offset_h::4,2+offset_w::4]
        
        np.save("{}bayers\\ldr_{}.npy".format(file_dir,level), ldr)

image_list = ['0050','0200','0800','3200']
ldr_group = np.zeros((4,3024,4032,3))

for idx,image_name in enumerate(image_list):
    print(image_name)
    image = np.array(rawpy.imread(f'imgs/500-iso{image_name}.dng').postprocess(no_auto_bright=True, output_bps=16, output_color=rawpy.ColorSpace(2)))
    image = image/(pow(2,16)-1)
    ldr_group[idx,:,:] = image

ldr_group = dng2ldrs(ldr_group)
np.save('ldr_group.npy',ldr_group)
raw_image = ldrs2raw(ldr_group)
np.save('test0.npy',raw_image)
imageio.imwrite('bayer.png',np.uint8(raw_image*255))