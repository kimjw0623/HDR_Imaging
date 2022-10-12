from os import listdir, path
import os
import numpy as np
import imageio
import glob
import rawpy
from tqdm import tqdm 

image_name_list = ['0050','0200','0800','3200'] #['20220614_133940','20220614_133934','20220614_133930','20220614_133909']#,'2','3','4']

real_dataset_dir = 'C:/Users/user/hdr_experiment/iso/'
dslr_dir = 'C:\\Users\\user\\samsung-hdr-demosaic-code\\deblur\\experiment\\dslr_images\\'
dslr_view_dir = 'C:\\Users\\user\\samsung-hdr-demosaic-code\\deblur\\experiment\\dslr_images_view\\'
new_dir = 'Z:\\data\\raw_image\\exptime\\scene8\\'
new_view_dir = 'Z:\\data\\raw_image\\exptime\\scene8-visual\\'
    
files = glob.glob(os.path.join(new_dir, '*.CR2'))[:-7]

def ldrs2raw(ldrs):
    _,h,w,_ = ldrs.shape
    ldrs = ldrs[:,:h//4*4,:w//4*4,:]
    raw_bayer = np.zeros((h//4*4,w//4*4))

    for idx in range(4):
        offset_h = idx//2
        offset_w = idx%2
        raw_bayer[offset_h::4,offset_w::4] = ldrs[idx,offset_h::4,offset_w::4,0]
        raw_bayer[2+offset_h::4,offset_w::4] = ldrs[idx,2+offset_h::4,offset_w::4,1]
        raw_bayer[offset_h::4,2+offset_w::4] = ldrs[idx,offset_h::4,2+offset_w::4,1]
        raw_bayer[2+offset_h::4,2+offset_w::4] = ldrs[idx,2+offset_h::4,2+offset_w::4,2]
    return raw_bayer

def generate_raw(file_):
    frame_target = [1,2,4,8]
    max_frame = max(frame_target)
    # print(max_frame)
    filename = os.path.split(file_)[-1]
    demosaiced_img = np.array(rawpy.imread(os.path.join(new_dir,filename)).postprocess(
        gamma = (1,1),
        demosaic_algorithm=None,
        no_auto_bright=True, 
        output_bps=16,
        output_color=rawpy.ColorSpace(0))) # gamma = (1,1)
    demosaiced_img = np.float32(demosaiced_img/65535)
    # print(demosaiced_img[1000,1000])
    h,w,c = demosaiced_img.shape
    frame_group = np.zeros((max_frame,h,w,c),np.float32)
    frame_group[0,:,:,:] = demosaiced_img
    ldr_group = np.zeros((4,h,w,c),np.float32)
    ldr_group[0,:,:,:] = demosaiced_img
    
    origin_filename = os.path.splitext(os.path.split(file_)[-1])[0]
    origin_filenumber = int(origin_filename[-4:])
    
    for i in range(1,max_frame):
        temp_filename = 'GO5A' + str(origin_filenumber + i) + '.CR2'
        temp_demosaiced_img = np.array(rawpy.imread(os.path.join(new_dir,temp_filename)).postprocess(
            gamma = (1,1),
            demosaic_algorithm=None,
            no_auto_bright=True, 
            output_bps=16,
            output_color=rawpy.ColorSpace(0)))
        temp_demosaiced_img = np.float32(temp_demosaiced_img/65535)
        frame_group[i,:,:,:] = temp_demosaiced_img
    
    for i,frame in enumerate(frame_target):
        sum_frame = np.float32(np.zeros((h,w,c)))
        for idx in range(frame):
            sum_frame += frame_group[idx,:,:,:]
        ldr_group[i,:,:,:] = sum_frame
        
    result = ldrs2raw(ldr_group)
    result = np.clip(result,0,1)
    mu = 5000
    ldr = np.log2(1+mu*result)/np.log2(1+mu)
    imageio.imwrite(f'{new_view_dir}{origin_filename}.png',np.uint8(ldr*255))
    print(result.shape)
    np.save(f'{new_dir}{origin_filename}.npy',result)
    
def combine_raw(file_,num_frame):
    filename = os.path.split(file_)[-1]
    demosaiced_img = np.array(rawpy.imread(os.path.join(new_dir,filename)).postprocess(
        gamma = (1,1),
        demosaic_algorithm=None,
        no_auto_bright=True, 
        output_bps=16,
        output_color=rawpy.ColorSpace(0))) # gamma = (1,1)
    demosaiced_img = np.float32(demosaiced_img)
    origin_filename = os.path.splitext(os.path.split(file_)[-1])[0]
    origin_filenumber = int(origin_filename[-4:])
    
    try:
        if num_frame > 1:
            for i in range(1,num_frame):
                temp_filename = 'GO5A' + str(origin_filenumber + i) + '.CR2'
                temp_demosaiced_img = np.array(rawpy.imread(os.path.join(new_dir,temp_filename)).postprocess(
                    gamma = (1,1),
                    demosaic_algorithm=None,
                    no_auto_bright=True, 
                    output_bps=16,
                    output_color=rawpy.ColorSpace(0)))
                demosaiced_img += np.float32(temp_demosaiced_img)
    except: # filenotfound
        return
        
    demosaiced_img = np.clip(demosaiced_img,0,65535)
    demosaiced_img = demosaiced_img/65535
    demosaiced_img = np.clip(demosaiced_img,0,1)
    demosaiced_img = np.log2(1+demosaiced_img*5000)/np.log2(1+5000)
    demosaiced_img = np.uint8(demosaiced_img*255)
    
    imageio.imwrite(f'{new_view_dir}{origin_filename}-{num_frame}.png',demosaiced_img)
    
from joblib import Parallel, delayed
num_cores = 10
# Parallel(n_jobs=num_cores)(delayed(combine_raw)(file_,frame_num) for file_ in tqdm(files) for frame_num in [1,2,4,8])
for i in range(1,9):
    new_dir = f'Z:\\data\\raw_image\\exptime\\scene{i}\\'
    new_view_dir = f'Z:\\data\\raw_image\\exptime\\scene{i}-visual\\'
        
    files = glob.glob(os.path.join(new_dir, '*.CR2'))[:-7]
    Parallel(n_jobs=num_cores)(delayed(generate_raw)(file_) for file_ in tqdm(files))
# Parallel(n_jobs=num_cores)(delayed(visualize_raw)(file_) for file_ in tqdm(files))