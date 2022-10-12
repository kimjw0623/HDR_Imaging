from distutils.log import error
from os import listdir, path
import os
from more_itertools import difference
import numpy as np
import imageio
import cv2
from skimage.measure import block_reduce
import math
from scipy import ndimage
import time
import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# peak value of each scene
hdrv_dict = {'bridge': 1.8486079524125767, 'bridge_2': 8.885517928962994, 'exhibition_area': 2086.6565167462386, 'hallway': 9236.556273307686, 'hallway2': 15244.469862891738, 'river': 27.730643114939657, 'students': 7894.0296937251, 'window': 31398.238802635064}
foot_dict = {'beerfest_lightshow_01': 60.717923174513146, 'beerfest_lightshow_02': 478.1178264212101, 'beerfest_lightshow_02_reconstruction_update_2015': 482.3360310006649, 'beerfest_lightshow_03': 35.939507230122885, 'beerfest_lightshow_04': 192.90464248502158,
             'beerfest_lightshow_04_reconstruction_update_2015': 204.72225637469143, 'beerfest_lightshow_05': 104.28169161081314, 'beerfest_lightshow_06': 351.03759219292965, 'beerfest_lightshow_07': 220.76632392406464, 'bistro_01': 174.7414870104253, 'bistro_02': 223.58788572711708, 'bistro_03': 86.55792842191808, 'carousel_fireworks_01': 364.19813082267046, 'carousel_fireworks_02': 498.7218083817829, 'carousel_fireworks_03': 493.635499359459, 'carousel_fireworks_04': 141.84006071090698, 'carousel_fireworks_05': 433.41774162204786, 'carousel_fireworks_06': 119.5637951222333, 'carousel_fireworks_07': 495.4070549533792, 'carousel_fireworks_08': 432.31907334229237, 'carousel_fireworks_09': 100.95291816104542, 'cars_closeshot': 429.8834358989329, 'cars_fullshot': 383.78515438580405, 'cars_longshot': 455.5022449028201, 'fireplace_01': 470.3036923457066, 'fireplace_02': 432.8508772777632, 'fishing_closeshot': 499.80078125, 'fishing_longshot': 353.30032906658073, 'hdr_testimage': 265.8702720006307, 'poker_fullshot': 233.58810709635418, 'poker_travelling_slowmotion': 318.52470106801803, 'showgirl_01': 396.09499288342664, 'showgirl_02': 367.95990511888635, 'smith_hammering': 476.6622160231531, 'smith_welding': 496.81805453153356}

scene_dict = foot_dict

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

dataset_name = 'HDR_Camera_Footage'#'HDRV' 
method_dir = 'C:\\Users\\user\\deblurring_dataset\\source\\{}\\'.format(dataset_name)
save_dir = 'C:\\Users\\user\\deblurring_dataset\\source_thumbnail\\{}\\'.format(dataset_name)
createFolder(save_dir)
peak_dict = {}
get_peak = 0
for data_dir in glob.glob('{}*\\'.format(method_dir)):
    scene_name = data_dir.split('\\')[-2]
    print(scene_name)
    if scene_name == 'poker_travelling_slowmotion':
        scene_dir = save_dir + scene_name + '\\'
        createFolder(scene_dir)
        print(scene_dir)
        scene_peak = 0
        cnt = 0
        for img_name_dir in glob.glob('{}*.exr'.format(data_dir)):
            img_name = img_name_dir.split('\\')[-1][:-4]
            hdr = imageio.imread(img_name_dir)
            if get_peak:
                resized = cv2.resize(hdr, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                resized = cv2.resize(resized, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                resized = cv2.resize(resized, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                hdr_peak = np.amax(resized)
                scene_peak += hdr_peak
                cnt += 1
            else:
                hdr = (hdr/scene_dict[scene_name])  
                hdr = np.clip(hdr,0,1)
                mu = 5000
                ldr = np.log2(1+mu*hdr)/np.log2(1+mu)
                # #imageio.imsave('{}\\mu40000_{}.png'.format(method_dir,img_name[:-4].split('\\')[-1]),(gt_ldr*255.0).astype('uint8'))
                imageio.imsave('{}mu5000_{}.png'.format(scene_dir,img_name),(ldr*255.0).astype('uint8'))
    if get_peak:
        peak_dict[scene_name] = scene_peak/cnt
    
print(peak_dict)