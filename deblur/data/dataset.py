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

gamma = 2.2
gain = 200

foot_dict = {'beerfest_lightshow_01': 60.717923174513146, 'beerfest_lightshow_02': 478.1178264212101, 'beerfest_lightshow_02_reconstruction_update_2015': 482.3360310006649, 'beerfest_lightshow_03': 35.939507230122885, 'beerfest_lightshow_04': 192.90464248502158,
             'beerfest_lightshow_04_reconstruction_update_2015': 204.72225637469143, 'beerfest_lightshow_05': 104.28169161081314, 'beerfest_lightshow_06': 351.03759219292965, 'beerfest_lightshow_07': 220.76632392406464, 'bistro_01': 174.7414870104253, 'bistro_02': 223.58788572711708, 'bistro_03': 86.55792842191808, 'carousel_fireworks_01': 364.19813082267046, 'carousel_fireworks_02': 498.7218083817829, 'carousel_fireworks_03': 493.635499359459, 'carousel_fireworks_04': 141.84006071090698, 'carousel_fireworks_05': 433.41774162204786, 'carousel_fireworks_06': 119.5637951222333, 'carousel_fireworks_07': 495.4070549533792, 'carousel_fireworks_08': 432.31907334229237, 'carousel_fireworks_09': 100.95291816104542, 'cars_closeshot': 429.8834358989329, 'cars_fullshot': 383.78515438580405, 'cars_longshot': 455.5022449028201, 'fireplace_01': 470.3036923457066, 'fireplace_02': 432.8508772777632, 'fishing_closeshot': 499.80078125, 'fishing_longshot': 353.30032906658073, 'hdr_testimage': 265.8702720006307, 'poker_fullshot': 233.58810709635418, 'poker_travelling_slowmotion': 318.52470106801803, 'showgirl_01': 396.09499288342664, 'showgirl_02': 367.95990511888635, 'smith_hammering': 476.6622160231531, 'smith_welding': 496.81805453153356}


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def hdr2ldr(hdrs): 
    # hdrs: shape=(4,H,W,3)
    # ldrs: shape=(4,H,W,3)
    hdrs = hdrs * (pow(2, 14)-1)
    ldrs = np.clip(hdrs, 0, (pow(2, 14)-1))
    ldrs = np.around(ldrs-0.5)/(pow(2, 14)-1)
    ldrs = np.clip(ldrs,0,1)
    return ldrs

def ldrs2raw(ldrs):
    _,h,w,_ = ldrs.shape
    ldrs = ldrs[:,:h//4*4,:w//4*4,:]
    raw_bayer = np.zeros((h//4*4,w//4*4))

    #level= 0
    for idx in range(4):
        offset_h = idx//2
        offset_w = idx%2
        raw_bayer[offset_h::4,offset_w::4] = ldrs[idx,offset_h::4,offset_w::4,0]
        raw_bayer[2+offset_h::4,offset_w::4] = ldrs[idx,2+offset_h::4,offset_w::4,1]
        raw_bayer[offset_h::4,2+offset_w::4] = ldrs[idx,offset_h::4,2+offset_w::4,1]
        raw_bayer[2+offset_h::4,2+offset_w::4] = ldrs[idx,2+offset_h::4,2+offset_w::4,2]
        #level += 1
    return raw_bayer

def ldrs2dual(ldrs): #RGGB, large exp first
    # option 1:
    # R3 G3
    # G3 B3
    # R0 G0
    # G0 B0
    # option 2:
    # G3 R3 
    # B3 G3 
    # G0 R0 
    # B0 G0 
    
    exposure,h,w,_ = ldrs.shape
    ldrs = ldrs[:,:h//4*4,:w//4*4,:]
    raw_bayer = np.zeros((h//4*4,w//4*4))

    option = True

    if option:
        raw_bayer[0::4,0::2] = ldrs[2,0::4,0::2,0] #R3
        raw_bayer[0::4,1::2] = ldrs[2,0::4,1::2,1] #G3
        raw_bayer[1::4,0::2] = ldrs[2,1::4,0::2,1] #G3
        raw_bayer[1::4,1::2] = ldrs[2,1::4,1::2,2] #B3

        raw_bayer[2::4,0::2] = ldrs[0,2::4,0::2,0] #R0
        raw_bayer[2::4,1::2] = ldrs[0,2::4,1::2,1] #G0
        raw_bayer[3::4,0::2] = ldrs[0,3::4,0::2,1] #G0
        raw_bayer[3::4,1::2] = ldrs[0,3::4,1::2,2] #B0
    
    else: 
        raw_bayer[0::4,0::2] = ldrs[2,0::4,0::2,1] #G3
        raw_bayer[0::4,1::2] = ldrs[2,0::4,1::2,0] #R3
        raw_bayer[1::4,0::2] = ldrs[2,1::4,0::2,2] #B3
        raw_bayer[1::4,1::2] = ldrs[2,1::4,1::2,1] #G3

        raw_bayer[2::4,0::2] = ldrs[0,2::4,0::2,1] #G0
        raw_bayer[2::4,1::2] = ldrs[0,2::4,1::2,0] #B0
        raw_bayer[3::4,0::2] = ldrs[0,3::4,0::2,2] #B0
        raw_bayer[3::4,1::2] = ldrs[0,3::4,1::2,1] #G0

    return raw_bayer

def raw2ldrs(raw):
    h,w = raw.shape
    ldr = np.zeros((4,h//2,w//2))
    for level in range(4):
        offset_h = level//2
        offset_w = level%2
        ldr[level,:,:] = raw[offset_h::2,offset_w::2]
    return ldr

def exrread(file_name):
    try:
        return imageio.imread('%s%s_%06d.exr'%(dataset_dir,short_name,(cur_frame_num)))
    except:
        print('filenotfound')

# Classify scene type
f = open('scene_info.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
num_line = 0
static_scene_list = []
dynamic_scene_list = []
for line in rdr:
    if num_line>1 and num_line < 19:
        if int(line[2]):
            dynamic_scene_list.append(line[0])
        else:
            static_scene_list.append(line[0])
    num_line += 1
f.close()  
print(static_scene_list)
print(dynamic_scene_list)

test_scene = ['cars_fullshot','poker_fullshot','carousel_fireworks_03']

imageio.plugins.freeimage.download()

count = 0
dataset_name = 'froehlich'#'technicolor'#'hdrv'#
scene_name = 'poker_travelling_slowmotion'#'exhibition_area'#'Market3_1920x1080p_50_hf_709'##'fireplace_01'#
exp_type = '1248'
get_ldr_images = 0
get_gt_hdr = 1 # get hdr gt if True
frame_skip = 0

dataset_list = {'froehlich':0, 'hdrv':1, 'technicolor':2}
dataset_num = dataset_list[dataset_name]
dataset_dir_list = ['HDR_Camera_Footage','HDRV','Technicolor_dataset']

method_dir = 'C:\\Users\\user\\deblurring_dataset\\source\\{}\\'.format(dataset_dir_list[dataset_num])
for data_dir in glob.glob('{}*\\'.format(method_dir)):
    scene_name = data_dir.split('\\')[-2]
    print(scene_name)
    if scene_name in dynamic_scene_list:
        frame_skip = 4
    elif scene_name in static_scene_list:
        frame_skip = 10
    else:
        continue
    dataset_dir = 'C:\\Users\\user\\deblurring_dataset\\source\\{}\\{}\\'.format(dataset_dir_list[dataset_num],scene_name)
    for img_name_dir in glob.glob('{}*.exr'.format(dataset_dir)):
        # read frame number
        if dataset_num in [0,2]:
            frame_num = int(img_name_dir.split('\\')[-1].split('_')[-1][:-4])
        elif dataset_num == 1:
            frame_num = int(img_name_dir.split('\\')[-1].split('.')[-2])
        
        if frame_num%frame_skip != 0:
            continue
        
        # image name w/ frame number
        img_name = img_name_dir.split('\\')[-1][:-4]
        
        # image name w/o frame number
        if dataset_num == 2:
            short_name = img_name_dir.split('\\')[-1][:-10]
        else:
            short_name = img_name_dir.split('\\')[-1][:-11]
        
        # ground-truth HDR frame
        target_frame = imageio.imread('{}{}.exr'.format(dataset_dir,img_name))
        gt_frame = target_frame.copy()
        img_h,img_w,_ = target_frame.shape
        cur_frame_num = frame_num
        exp0 = target_frame.copy()
        
        # get peak value
        target_frame_peak = foot_dict[scene_name]
        # resized = cv2.resize(target_frame, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        # resized = cv2.resize(resized, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        # resized = cv2.resize(resized, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        # target_frame_peak = np.amax(resized)
        
        # GT dataset
        target_frame = target_frame/target_frame_peak
        
        if get_gt_hdr:
            if scene_name in test_scene:
                np.save('Z:\\data\\deblurring_dataset\\test_gt_hdr\\{}.npy'.format(img_name),target_frame)
            else:
                np.save('Z:\\data\\deblurring_dataset\\gt_hdr\\{}.npy'.format(img_name),target_frame)
            continue
        
        # generate HDR with various exposure time, if FileNotFoundError, break for loop
        try:
            if dataset_name == 'froehlich':
                if exp_type=='1248':
                    exp1 = exp0.copy()
                    for i in range(1):
                        cur_frame_num += 1
                        exp1 += imageio.imread('%s%s_%06d.exr'%(dataset_dir,short_name,(cur_frame_num)))
                    
                    exp2 = exp1.copy()
                    for i in range(2):
                        cur_frame_num += 1
                        exp2 += imageio.imread('%s%s_%06d.exr'%(dataset_dir,short_name,(cur_frame_num)))
                        
                    exp3 = exp2.copy()
                    for i in range(4):
                        cur_frame_num += 1
                        exp3 += imageio.imread('%s%s_%06d.exr'%(dataset_dir,short_name,(cur_frame_num)))   
                elif exp_type=='1416':
                    exp1 = exp0.copy()
                    for i in range(3):
                        cur_frame_num += 1
                        exp1 += imageio.imread('%s%s_%06d.exr'%(dataset_dir,short_name,(cur_frame_num)))
                    
                    exp2 = exp1.copy()
                    exp3 = exp1.copy()
                    for i in range(12):
                        cur_frame_num += 1
                        exp2 += imageio.imread('%s%s_%06d.exr'%(dataset_dir,short_name,(cur_frame_num)))
            elif dataset_name == 'hdrv':
                if exp_type=='1248':
                    exp1 = exp0.copy()
                    for i in range(1):
                        cur_frame_num += 1
                        exp1 += imageio.imread('%s%s.%06d.exr'%(dataset_dir,short_name,(cur_frame_num)))
                    
                    exp2 = exp1.copy()
                    for i in range(2):
                        cur_frame_num += 1
                        exp2 += imageio.imread('%s%s.%06d.exr'%(dataset_dir,short_name,(cur_frame_num)))
                        
                    exp3 = exp2.copy()
                    for i in range(4):
                        cur_frame_num += 1
                        exp3 += imageio.imread('%s%s.%06d.exr'%(dataset_dir,short_name,(cur_frame_num)))  
            elif dataset_name == 'technicolor':
                if exp_type=='1248':
                    exp1 = exp0.copy()
                    for i in range(1):
                        cur_frame_num += 1
                        exp1 += imageio.imread('%s%s_%05d.exr'%(dataset_dir,short_name,(cur_frame_num)))
                    
                    exp2 = exp1.copy()
                    for i in range(2):
                        cur_frame_num += 1
                        exp2 += imageio.imread('%s%s_%05d.exr'%(dataset_dir,short_name,(cur_frame_num)))
                        
                    exp3 = exp2.copy()
                    for i in range(4):
                        cur_frame_num += 1
                        exp3 += imageio.imread('%s%s_%05d.exr'%(dataset_dir,short_name,(cur_frame_num)))
        except FileNotFoundError:
            break
        
        print('asd')
        
        hdrs = np.clip(np.stack((exp0,exp1,exp2,exp3),axis=0)/target_frame_peak,0,1) # shape=(4,H,W,3)
        gts = np.clip(np.stack((target_frame, target_frame*2, target_frame*4, target_frame*8), axis=0),0, 1) # shape=(4,H,W,3)
        
        save_dir = 'C:\\Users\\user\\deblurring_dataset\\ours\\dataset_{}\\'.format(exp_type)
        scene_dir = save_dir + 'exp_ldr_visualization\\' + scene_name + '\\'
        createFolder(scene_dir)
        if get_ldr_images:
            for i in range(4):
                output = np.log2(1+hdrs[i]*5000)/np.log2(1+5000)
                imageio.imwrite('{}{}_{}.png'.format(scene_dir, img_name, i),np.uint8(output*255))
        else:    
            ldrs = hdr2ldr(hdrs) # shape=(4,H,W,3)
            output = ldrs2raw(ldrs) # shape=(H,W)
            output = np.clip(output,0,1)
            frame_gap = output.copy()
            print(output.shape)
            print(gts.shape)
            
            np.save('C:\\Users\\user\\deblurring_dataset\\ours\\dataset_1248\\bayer\\{}.npy'.format(img_name),output)
            # np.save('C:\\Users\\user\\deblurring_dataset\\ours\\dataset_1248\\gt\\{}.npy'.format(img_name),gts)
            
        # output = np.log2(1+output*5000)/np.log2(1+5000)
        # imageio.imwrite('C:\\Users\\user\\deblurring_dataset\\ours\\dataset_{}\\bayer_visualization\\{}.png'.format(exp_type, img_name),np.uint8(output*255))
        
        # exps = raw2ldrs(output)
        # #print(exps[0,100:110,500:504]*255)
        # for i in range(4):
        #     imageio.imwrite('C:\\Users\\user\\deblurring_dataset\\ours\\dataset_{}\\exp_visualization\\{}_{}.png'.format(exp_type, img_name, i),np.uint8(exps[i]*255))
    
    # Get frame gap
    # raw_exps = raw2ldrs(frame_gap)
    # for i in range(3):
    #     output = np.clip(raw_exps[i+1]-raw_exps[i],0,1)
    #     output = np.log2(1+output*5000)/np.log2(1+5000)
    #     imageio.imwrite('C:\\Users\\user\\deblurring_dataset\\ours\\dataset_{}\\frame_gap\\{}_{}.png'.format(exp_type, img_name, i),np.uint8((output)*255))