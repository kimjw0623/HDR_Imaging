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
new_dir = 'Z:\\data\\raw_image\\iso\\raw\\'
new_view_dir = 'Z:\\data\\raw_image\\iso\\visualize\\'
    
files = glob.glob(os.path.join(new_dir, '*.CR2'))  

def visualize_raw(file_):
    filename = os.path.split(file_)[-1]
    demosaiced_img = np.array(rawpy.imread(os.path.join(new_dir,filename)).postprocess(
        demosaic_algorithm=None,
        no_auto_bright=True, 
        output_bps=16,
        output_color=rawpy.ColorSpace(0))) # gamma = (1,1)
    filename = os.path.splitext(os.path.split(file_)[-1])[0]
    
    demosaiced_img = np.uint8((demosaiced_img/65535)*255)
    imageio.imwrite(f'{new_view_dir}{filename}.png',demosaiced_img)
    
from joblib import Parallel, delayed
num_cores = 10
Parallel(n_jobs=num_cores)(delayed(visualize_raw)(file_) for file_ in tqdm(files))