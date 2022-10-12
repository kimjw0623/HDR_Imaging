from os import listdir, path
import os
import numpy as np
import imageio
import cv2
import skimage
import math
import time
import glob


def color_psnr(img1, img2, imax):
    mse = np.mean((img1 - img2) ** 2)
    psnr = 20 * np.log10(imax) - 10 * np.log10(mse)
    #print(mse)
    return psnr

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

method_dir = 'C:\\Users\\user\\comparison\\result_dcnn\\noise_0.25\\'
ours_dir = 'C:\\Users\\user\\comparison\\ours_0.25\\'
exr_dir = 'C:\\Users\\user\\comparison\\test_exr\\'

data_dir = 'C:\\Users\\user\\comparison\\data_serrano\\0.25_level0\\'


fairchild_gt_dir = 'C:\\Users\\user\\comparison\\fairchild\\gt\\'

deblur_gt_cropped_dir = 'Z:\\data\deblurring_dataset\\test_gt_hdr_crops\\'

test_npy = ['Z:\\data\\deblurring_dataset\\test_gt_hdr_crops\\cars_fullshot_000400-41.npy','Z:\\data\\deblurring_dataset\\test_gt_hdr_crops\\poker_fullshot_000630-14.npy']

test_name = ['cars_fullshot_000448-41', 'poker_fullshot_000630-36', 'cars_fullshot_000596-47', 'cars_fullshot_000724-55', 'poker_fullshot_000430-43', 'cars_fullshot_000736-49', 'cars_fullshot_000760-50', 'poker_fullshot_000790-35', 'carousel_fireworks_03_000392-59', 'cars_fullshot_000356-56', 'cars_fullshot_000488-40', 'poker_fullshot_000890-33', 'cars_fullshot_000532-3', 'cars_fullshot_000564-29', 'poker_fullshot_000570-21', 'cars_fullshot_000776-32', 'cars_fullshot_000692-55', 'cars_fullshot_000408-9', 'cars_fullshot_000620-1', 'carousel_fireworks_03_000476-43', 'poker_fullshot_000730-46', 'cars_fullshot_000716-57', 'carousel_fireworks_03_000492-15', 'cars_fullshot_000564-46', 'cars_fullshot_000768-55', 'cars_fullshot_000572-8', 'cars_fullshot_000364-13', 'cars_fullshot_000480-3', 'carousel_fireworks_03_000356-18', 'carousel_fireworks_03_000440-9', 'cars_fullshot_000472-54', 'poker_fullshot_000640-15', 'carousel_fireworks_03_000388-27', 'poker_fullshot_000710-21', 'cars_fullshot_000476-22', 'carousel_fireworks_03_000372-8', 'cars_fullshot_000612-57', 'cars_fullshot_000676-28', 'carousel_fireworks_03_000424-44', 'cars_fullshot_000756-13', 'cars_fullshot_000472-51', 'cars_fullshot_000768-41', 'cars_fullshot_000660-48', 'cars_fullshot_000628-5', 'cars_fullshot_000784-8', 'poker_fullshot_000610-32', 'cars_fullshot_000748-40', 'carousel_fireworks_03_000524-12', 'cars_fullshot_000356-10', 'poker_fullshot_000560-18', 'cars_fullshot_000484-10', 'cars_fullshot_000572-29', 'poker_fullshot_000780-52', 'poker_fullshot_000370-15', 'carousel_fireworks_03_000516-2', 'cars_fullshot_000552-5', 'cars_fullshot_000528-4', 'poker_fullshot_000610-47', 'carousel_fireworks_03_000460-29', 'poker_fullshot_000580-59', 'carousel_fireworks_03_000420-53', 'poker_fullshot_000670-53', 'carousel_fireworks_03_000404-50', 'poker_fullshot_000690-6', 'carousel_fireworks_03_000392-33', 'poker_fullshot_000370-25', 'poker_fullshot_000400-3', 'cars_fullshot_000536-34', 'poker_fullshot_000380-51', 'carousel_fireworks_03_000484-16', 'cars_fullshot_000484-29', 'poker_fullshot_000960-3', 'poker_fullshot_000900-18', 'poker_fullshot_000760-35', 'cars_fullshot_000752-11', 'cars_fullshot_000712-9', 'carousel_fireworks_03_000416-4', 'carousel_fireworks_03_000440-60', 'carousel_fireworks_03_000368-44', 'cars_fullshot_000784-55', 'poker_fullshot_000770-60', 'cars_fullshot_000612-11', 'cars_fullshot_000596-56', 'poker_fullshot_000500-13', 'poker_fullshot_000810-31', 'carousel_fireworks_03_000396-32', 'cars_fullshot_000728-8', 'carousel_fireworks_03_000520-38', 'cars_fullshot_000536-38', 'cars_fullshot_000612-5', 'carousel_fireworks_03_000460-6', 'cars_fullshot_000716-30', 'poker_fullshot_000890-58', 'carousel_fireworks_03_000360-31', 'cars_fullshot_000700-50', 'carousel_fireworks_03_000468-55', 'carousel_fireworks_03_000392-18', 'cars_fullshot_000780-35', 'carousel_fireworks_03_000460-14', 'carousel_fireworks_03_000460-5', 'cars_fullshot_000648-33', 'cars_fullshot_000388-49', 'carousel_fireworks_03_000432-16', 'cars_fullshot_000724-16', 'cars_fullshot_000400-45', 'cars_fullshot_000592-51', 'cars_fullshot_000752-57', 'poker_fullshot_000960-43', 'cars_fullshot_000668-21', 'cars_fullshot_000392-54', 'cars_fullshot_000652-38', 'cars_fullshot_000632-14', 'cars_fullshot_000744-40', 'cars_fullshot_000776-41', 'cars_fullshot_000784-32', 'cars_fullshot_000784-31', 'poker_fullshot_000890-5', 'carousel_fireworks_03_000484-58', 'poker_fullshot_000630-14', 'cars_fullshot_000556-21', 'carousel_fireworks_03_000412-39', 'cars_fullshot_000576-11', 'cars_fullshot_000584-33', 'poker_fullshot_000840-36', 'cars_fullshot_000724-43', 'cars_fullshot_000780-19', 'poker_fullshot_000610-5', 'cars_fullshot_000544-1', 'cars_fullshot_000780-12', 'cars_fullshot_000440-36', 'cars_fullshot_000380-25', 'carousel_fireworks_03_000504-10', 'poker_fullshot_000690-27', 'poker_fullshot_000620-16', 'cars_fullshot_000728-17', 'poker_fullshot_000840-39', 'poker_fullshot_000540-26', 'poker_fullshot_000470-58', 'carousel_fireworks_03_000432-1', 'carousel_fireworks_03_000416-6', 'cars_fullshot_000780-16', 'cars_fullshot_000420-5', 'carousel_fireworks_03_000508-59', 'poker_fullshot_000570-38', 'cars_fullshot_000504-24', 'cars_fullshot_000456-37', 'cars_fullshot_000752-35', 'poker_fullshot_000780-23', 'poker_fullshot_000370-46', 'cars_fullshot_000532-20', 'poker_fullshot_000770-29', 'poker_fullshot_000540-45', 'carousel_fireworks_03_000440-59', 'poker_fullshot_000560-1', 'cars_fullshot_000568-52', 'cars_fullshot_000360-21', 'cars_fullshot_000588-4', 'cars_fullshot_000464-13', 'cars_fullshot_000700-4', 'poker_fullshot_000750-23', 'cars_fullshot_000692-56', 'cars_fullshot_000360-15', 'carousel_fireworks_03_000508-36', 'carousel_fireworks_03_000380-48', 'carousel_fireworks_03_000488-57', 'cars_fullshot_000688-33', 'poker_fullshot_000500-53', 'poker_fullshot_000720-42', 'cars_fullshot_000432-43', 'poker_fullshot_000550-36', 'carousel_fireworks_03_000456-4', 'carousel_fireworks_03_000428-2', 'poker_fullshot_000630-4', 'poker_fullshot_000450-29', 'poker_fullshot_000940-39', 'carousel_fireworks_03_000428-49', 'cars_fullshot_000536-45', 'poker_fullshot_000690-21', 'cars_fullshot_000632-8', 'cars_fullshot_000772-39', 'carousel_fireworks_03_000416-37', 'cars_fullshot_000716-8', 'poker_fullshot_000380-37', 'cars_fullshot_000416-9', 'cars_fullshot_000384-15', 'cars_fullshot_000672-25', 'cars_fullshot_000784-20', 'poker_fullshot_000820-1', 'cars_fullshot_000380-39', 'poker_fullshot_000380-38', 'cars_fullshot_000644-17', 'cars_fullshot_000404-42', 'cars_fullshot_000420-55', 'carousel_fireworks_03_000524-18', 'cars_fullshot_000688-30', 'cars_fullshot_000364-55', 'cars_fullshot_000556-33', 'cars_fullshot_000396-29', 'carousel_fireworks_03_000424-29', 'carousel_fireworks_03_000500-5']

for img_name in test_name: #glob.glob('{}*.npy'.format(deblur_gt_cropped_dir)):
    print('start')
    #hdr = imageio.imread(img_name)
    
    #hdr_peak = np.amax(hdr)
    #hdr = hdr/hdr_peak
    
    hdr = np.load('Z:\\data\\deblurring_dataset\\test_gt_hdr_crops\\{}.npy'.format(img_name))
    hdr = np.clip(hdr,0,1)#[1:,:,:]
    
    #pow(23.5,2))/pow(4.1,2)
    
    noises = np.random.normal(0.0, scale=0.25, size=(hdr.shape))
    hdr = hdr + (np.sqrt((hdr*16383.0 + pow(23.5,2))/pow(4.1,2))/16383.0 * noises) 
    # pow(23.5,2))/pow(4.1,2))/16383.0 suda
    # pow(6.4,2))/pow(1.01,2))/16383.0 serrano
    
    
    hdr = np.clip(hdr,0,1)
    
    print(hdr.shape)
    
    # C:\Users\user\DeepSHDR_Data\DeepSHDR_Data\excutable_code\our_data
    # imageio.imsave('C:\\Users\\user\\deblurring_dataset\\comparison\\serrano_data\\{}.hdr'.format(img_name),np.float32(hdr))
    imageio.imsave('C:\\Users\\user\\DeepSHDR_Data\\DeepSHDR_Data\\excutable_code\\our_data\\{}.hdr'.format(img_name),np.float32(hdr))
