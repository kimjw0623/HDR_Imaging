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
from matplotlib import pyplot as plt
import rawpy

#img = imageio.imread('imgs/20220614_150826.dng',format = 'RAW-FI')
img = np.array(rawpy.imread('imgs/20220614_152355.dng').postprocess(no_auto_bright=True, output_bps=16, output_color=rawpy.ColorSpace(0)))
    
img = np.uint8((img/65535)*255)
color = ('r','g','b')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()