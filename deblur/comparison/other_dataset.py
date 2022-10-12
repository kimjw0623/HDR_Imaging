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

# from colour_demosaicing import (
#     demosaicing_CFA_Bayer_bilinear,
#     demosaicing_CFA_Bayer_Malvar2004,
#     demosaicing_CFA_Bayer_Menon2007,
#     mosaicing_CFA_Bayer)

# Generate Raw Bayer Images from HDR Images

gamma = 2.2
gain = 200



def hdr2ldr(srcArray, bin=2):
    # Exposure separation
    hdrs = []
    exposures = bin ** 2

    #srcArray = np.clip(srcArray,0,150)
    peak = np.amax(srcArray)
    srcArray = srcArray/peak

    # if True:
    #     filter_size = 11
    #     gt_copy = srcArray.copy()
    #     avg_filter = np.ones((filter_size,filter_size))
    #     gt_peak = 0
    #     for i in range(3): # per channel
    #         gt_avg = ndimage.convolve(gt_copy[:,:,i], avg_filter, mode = 'constant', cval=0.0)
    #         channel_gt_peak = np.amax(gt_avg)/(filter_size*filter_size)
    #         if channel_gt_peak > gt_peak:
    #             gt_peak = channel_gt_peak
    # srcArray = srcArray/gt_peak

    #print(gt_peak)
    
    srcArray = np.clip(srcArray,0,1)

    for level in range(exposures):
        hdrs.append(srcArray * pow(4.0, level) * (pow(2, 14)-1) )
    ldrs = np.clip(np.stack(hdrs, axis=0), 0, (pow(2, 14)-1))
    ldrs = np.around(ldrs-0.5)/(pow(2, 14)-1)
    ldrs = np.clip(ldrs,0,1)
    #ldrs_gamma = np.uint8(ldrs * 255)
    #for level in range(exposures):
    #   cv2.imwrite(path.join(OUT_DIR, "LDR_{}.png".format(level)), cv2.cvtColor(ldrs_gamma[level],cv2.COLOR_BGR2RGB))

    return ldrs

def BinnedBayer_2_Bayer(srcArray, bin=2):
    """
    bin = 2,    block = 4
    _______     _______
    |RR|GG|     |RG|RG|
    |RR|GG|     |GB|GB|
    -------  -> -------
    |GG|BB|     |RG|RG|
    |GG|BB|     |GB|GB|
    -------     -------
    """
    h, w, _ = srcArray.shape
    block = 2 * bin
    res_h = h // block * block
    res_w = w // block * block

    srcArray = srcArray[:res_h, :res_w, :]
    resArray = np.zeros((res_h, res_w, 1), dtype=np.float32)

    for i in range(bin):
        for j in range(bin):
            # Red
            resArray[i * bin::block, j * bin::block, 0] = srcArray[i::block, j::block, 0]
            # Green 1
            resArray[i * bin::block, j * bin + 1::block, 0] = srcArray[i::block, bin + j::block, 0]
            # Green 2
            resArray[i * bin + 1::block, j * bin::block, 0] = srcArray[bin + i::block, j::block, 0]
            # Blue
            resArray[i * bin + 1::block, j * bin + 1::block, 0] = srcArray[bin + i::block, bin + j::block, 0]

    return resArray

def RGB_2_BinnedBayer(srcArray, bin=2):
    """
    Default     Tetra Pattern
    bin = 1     bin = 2
    _____       _______
    |R|G|       |RR|GG|
    |G|B|       |RR|GG|
    -----       -------
                |GG|BB|
                |GG|BB|
                -------
    """
    h, w, _ = srcArray.shape
    block = 2 * bin
    res_h = h // block * block
    res_w = w // block * block
    srcArray = srcArray[:res_h, :res_w, :]
    resArray = np.zeros((res_h, res_w, 1), dtype=np.float32)
    # Red
    for i in range(bin):
        for j in range(bin):
            # Red
            resArray[i::block, j::block, 0] = srcArray[i::block, j::block, 0]
            # Green 1
            resArray[i::block, bin + j::block, 0] = srcArray[i::block, bin + j::block, 1]
            # Green 2
            resArray[bin + i::block, j::block, 0] = srcArray[bin + i::block, j::block, 1]
            # Blue
            resArray[bin + i::block, bin + j::block, 0] = srcArray[bin + i::block, bin + j::block, 2]

    return resArray

def LDRs_2_LDRBayers(ldrs, OUT_DIR=None, bin=2):
    exposures, h, w, c = ldrs.shape
    assert(exposures == bin ** 2)

    block = bin * 2
    ldr_bayers = np.zeros((exposures, h // block * block, w // block * block, 1))
    for level in range(exposures):
        binnedBayer = RGB_2_BinnedBayer(ldrs[level], bin=2)
        ldr_bayers[level, :, :, :] = BinnedBayer_2_Bayer(binnedBayer, bin=2)
    
    np.save(path.join(OUT_DIR, "ldr_bayers.npy"), ldr_bayers)
    return ldr_bayers


def LDRBayers_2_LDRBayer(ldrs, OUT_DIR=None, bin=2):
    """
    +-----+-----+
    |R0 G0|R1 G1|
    |G0 B0|G1 B1|
    +-----+-----+
    |R2 G2|R3 G3|
    |G2 B2|G3 B3|
    +-----+-----+
    """
    exposures, h, w, _ = ldrs.shape
    block = bin * 2
    resBayer = np.zeros((h, w))
    for level in range(exposures):
        i = level // bin
        j = level % bin
        resBayer[i * 2::block, j * 2::block] = ldrs[level, i * 2::block, j * 2::block, 0]
        resBayer[i * 2 + 1::block, j * 2::block] = ldrs[level, i * 2 + 1::block, j * 2::block, 0]
        resBayer[i * 2::block, j * 2 + 1::block] = ldrs[level, i * 2::block, j * 2 + 1::block, 0]
        resBayer[i * 2 + 1::block, j * 2 + 1::block] = ldrs[level, i * 2 + 1::block, j * 2 + 1::block, 0]

    np.save(path.join(OUT_DIR, "ldr_exposure_binned_bayer.npy"), resBayer)
    """
    if OUT_DIR is not None:
        cv2.imwrite(path.join(OUT_DIR, "ldr_bayer.png"), resBayer[:, :, 0])
    """

    return resBayer

def BinBayer_2_LDRLRs(ldr_bayer, OUT_DIR=None, bin=2):
    # Exposure separation
    h, w = ldr_bayer.shape
    exposures = bin ** 2
    block = bin * 2
    print(h,w)
    LDR_BAYERs = np.zeros([exposures, h // bin, w // bin, 1], dtype=np.uint8)
    LDR_RGBs = np.zeros([exposures, h // (2*bin), w // (2*bin), 3], dtype=np.uint8)
    for level in range(exposures):
        i = level // bin
        j = level % bin
        LDR_BAYERs[level, ::2, ::2, 0] = ldr_bayer[bin * i::block, bin * j::block]
        LDR_BAYERs[level, 1::2, ::2, 0] = ldr_bayer[bin * i + 1::block, bin * j::block]
        LDR_BAYERs[level, ::2, 1::2, 0] = ldr_bayer[bin * i::block, bin * j + 1::block]
        LDR_BAYERs[level, 1::2, 1::2, 0] = ldr_bayer[bin * i + 1::block, bin * j + 1::block]
    
    for level in range(exposures):
        cv2.imwrite("{}bilinear_{}.png".format(OUT_DIR,level),demosaicing_CFA_Bayer_bilinear(LDR_BAYERs[level].squeeze(2),'RGGB'))
        # cv2.imwrite("{}/Malvar_{}.png".format(Imgname,level),demosaicing_CFA_Bayer_Malvar2004(LDR_BAYERs[level].squeeze(2),'RGGB'))
        cv2.imwrite("{}Menon_{}.png".format(OUT_DIR,level),demosaicing_CFA_Bayer_Menon2007(LDR_BAYERs[level].squeeze(2),'RGGB'))
        #cv2.imwrite("Demosaicing/cv2_{}.png".format(level),cv2.demosaicing(LDR_BAYERs[level].squeeze(2), cv2.COLOR_BayerBG2GRAY_VNG))
    for level in range(exposures):
        np.save("{}bayers\\ldr_{}.npy".format(OUT_DIR,level), LDR_BAYERs[level])

    return LDR_BAYERs

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

def readHDR(fname):
    im = imageio.imread(fname, format='HDR-FI')
    return np.array(im) 

dir = 'Z:\\data\\froehlich_dataset\\test\\'
imageio.plugins.freeimage.download()

test_name = ['cars_fullshot_000448-41', 'poker_fullshot_000630-36', 'cars_fullshot_000596-47', 'cars_fullshot_000724-55', 'poker_fullshot_000430-43', 'cars_fullshot_000736-49', 'cars_fullshot_000760-50', 'poker_fullshot_000790-35', 'carousel_fireworks_03_000392-59', 'cars_fullshot_000356-56', 'cars_fullshot_000488-40', 'poker_fullshot_000890-33', 'cars_fullshot_000532-3', 'cars_fullshot_000564-29', 'poker_fullshot_000570-21', 'cars_fullshot_000776-32', 'cars_fullshot_000692-55', 'cars_fullshot_000408-9', 'cars_fullshot_000620-1', 'carousel_fireworks_03_000476-43', 'poker_fullshot_000730-46', 'cars_fullshot_000716-57', 'carousel_fireworks_03_000492-15', 'cars_fullshot_000564-46', 'cars_fullshot_000768-55', 'cars_fullshot_000572-8', 'cars_fullshot_000364-13', 'cars_fullshot_000480-3', 'carousel_fireworks_03_000356-18', 'carousel_fireworks_03_000440-9', 'cars_fullshot_000472-54', 'poker_fullshot_000640-15', 'carousel_fireworks_03_000388-27', 'poker_fullshot_000710-21', 'cars_fullshot_000476-22', 'carousel_fireworks_03_000372-8', 'cars_fullshot_000612-57', 'cars_fullshot_000676-28', 'carousel_fireworks_03_000424-44', 'cars_fullshot_000756-13', 'cars_fullshot_000472-51', 'cars_fullshot_000768-41', 'cars_fullshot_000660-48', 'cars_fullshot_000628-5', 'cars_fullshot_000784-8', 'poker_fullshot_000610-32', 'cars_fullshot_000748-40', 'carousel_fireworks_03_000524-12', 'cars_fullshot_000356-10', 'poker_fullshot_000560-18', 'cars_fullshot_000484-10', 'cars_fullshot_000572-29', 'poker_fullshot_000780-52', 'poker_fullshot_000370-15', 'carousel_fireworks_03_000516-2', 'cars_fullshot_000552-5', 'cars_fullshot_000528-4', 'poker_fullshot_000610-47', 'carousel_fireworks_03_000460-29', 'poker_fullshot_000580-59', 'carousel_fireworks_03_000420-53', 'poker_fullshot_000670-53', 'carousel_fireworks_03_000404-50', 'poker_fullshot_000690-6', 'carousel_fireworks_03_000392-33', 'poker_fullshot_000370-25', 'poker_fullshot_000400-3', 'cars_fullshot_000536-34', 'poker_fullshot_000380-51', 'carousel_fireworks_03_000484-16', 'cars_fullshot_000484-29', 'poker_fullshot_000960-3', 'poker_fullshot_000900-18', 'poker_fullshot_000760-35', 'cars_fullshot_000752-11', 'cars_fullshot_000712-9', 'carousel_fireworks_03_000416-4', 'carousel_fireworks_03_000440-60', 'carousel_fireworks_03_000368-44', 'cars_fullshot_000784-55', 'poker_fullshot_000770-60', 'cars_fullshot_000612-11', 'cars_fullshot_000596-56', 'poker_fullshot_000500-13', 'poker_fullshot_000810-31', 'carousel_fireworks_03_000396-32', 'cars_fullshot_000728-8', 'carousel_fireworks_03_000520-38', 'cars_fullshot_000536-38', 'cars_fullshot_000612-5', 'carousel_fireworks_03_000460-6', 'cars_fullshot_000716-30', 'poker_fullshot_000890-58', 'carousel_fireworks_03_000360-31', 'cars_fullshot_000700-50', 'carousel_fireworks_03_000468-55', 'carousel_fireworks_03_000392-18', 'cars_fullshot_000780-35', 'carousel_fireworks_03_000460-14', 'carousel_fireworks_03_000460-5', 'cars_fullshot_000648-33', 'cars_fullshot_000388-49', 'carousel_fireworks_03_000432-16', 'cars_fullshot_000724-16', 'cars_fullshot_000400-45', 'cars_fullshot_000592-51', 'cars_fullshot_000752-57', 'poker_fullshot_000960-43', 'cars_fullshot_000668-21', 'cars_fullshot_000392-54', 'cars_fullshot_000652-38', 'cars_fullshot_000632-14', 'cars_fullshot_000744-40', 'cars_fullshot_000776-41', 'cars_fullshot_000784-32', 'cars_fullshot_000784-31', 'poker_fullshot_000890-5', 'carousel_fireworks_03_000484-58', 'poker_fullshot_000630-14', 'cars_fullshot_000556-21', 'carousel_fireworks_03_000412-39', 'cars_fullshot_000576-11', 'cars_fullshot_000584-33', 'poker_fullshot_000840-36', 'cars_fullshot_000724-43', 'cars_fullshot_000780-19', 'poker_fullshot_000610-5', 'cars_fullshot_000544-1', 'cars_fullshot_000780-12', 'cars_fullshot_000440-36', 'cars_fullshot_000380-25', 'carousel_fireworks_03_000504-10', 'poker_fullshot_000690-27', 'poker_fullshot_000620-16', 'cars_fullshot_000728-17', 'poker_fullshot_000840-39', 'poker_fullshot_000540-26', 'poker_fullshot_000470-58', 'carousel_fireworks_03_000432-1', 'carousel_fireworks_03_000416-6', 'cars_fullshot_000780-16', 'cars_fullshot_000420-5', 'carousel_fireworks_03_000508-59', 'poker_fullshot_000570-38', 'cars_fullshot_000504-24', 'cars_fullshot_000456-37', 'cars_fullshot_000752-35', 'poker_fullshot_000780-23', 'poker_fullshot_000370-46', 'cars_fullshot_000532-20', 'poker_fullshot_000770-29', 'poker_fullshot_000540-45', 'carousel_fireworks_03_000440-59', 'poker_fullshot_000560-1', 'cars_fullshot_000568-52', 'cars_fullshot_000360-21', 'cars_fullshot_000588-4', 'cars_fullshot_000464-13', 'cars_fullshot_000700-4', 'poker_fullshot_000750-23', 'cars_fullshot_000692-56', 'cars_fullshot_000360-15', 'carousel_fireworks_03_000508-36', 'carousel_fireworks_03_000380-48', 'carousel_fireworks_03_000488-57', 'cars_fullshot_000688-33', 'poker_fullshot_000500-53', 'poker_fullshot_000720-42', 'cars_fullshot_000432-43', 'poker_fullshot_000550-36', 'carousel_fireworks_03_000456-4', 'carousel_fireworks_03_000428-2', 'poker_fullshot_000630-4', 'poker_fullshot_000450-29', 'poker_fullshot_000940-39', 'carousel_fireworks_03_000428-49', 'cars_fullshot_000536-45', 'poker_fullshot_000690-21', 'cars_fullshot_000632-8', 'cars_fullshot_000772-39', 'carousel_fireworks_03_000416-37', 'cars_fullshot_000716-8', 'poker_fullshot_000380-37', 'cars_fullshot_000416-9', 'cars_fullshot_000384-15', 'cars_fullshot_000672-25', 'cars_fullshot_000784-20', 'poker_fullshot_000820-1', 'cars_fullshot_000380-39', 'poker_fullshot_000380-38', 'cars_fullshot_000644-17', 'cars_fullshot_000404-42', 'cars_fullshot_000420-55', 'carousel_fireworks_03_000524-18', 'cars_fullshot_000688-30', 'cars_fullshot_000364-55', 'cars_fullshot_000556-33', 'cars_fullshot_000396-29', 'carousel_fireworks_03_000424-29', 'carousel_fireworks_03_000500-5']

is_deblurring = False
is_experiment = True

if is_experiment:
    ldrs = np.load('exp_data/ldr_group.npy')
    print(ldrs.shape)
    output = ldrs2dual(ldrs)
    output = np.clip(output,0,1)
    print(np.amax(output))
    
    # mdic = {'raw':output}
    # savemat('C:\\Users\\user\\deblurring_dataset\\comparison\\heide_data\\test0.mat',mdic)
    
    # imageio.imwrite('exp_data/test_dual.png',np.uint8(output*255))
    
    print(output.shape)
    

if is_deblurring:
    for img_name in test_name:
        hdr = np.load('Z:\\data\\deblurring_dataset\\test_gt_hdr_crops\\{}.npy'.format(img_name))
        hdr = np.clip(hdr,0,1)
        ldrs = hdr2ldr(hdr)
        #output = ldrs2raw(ldrs)
        output = ldrs2dual(ldrs)
        patch = np.clip(output,0,1)
        
        noise_h, noise_w = patch.shape
        noises = np.random.normal(0.0, scale=0.25, size=(noise_h, noise_w))

        patch[::2,::2] = patch[::2,::2] + (np.sqrt((patch[::2,::2]*16383.0 + pow(23.5,2))/pow(4.1,2))/16383.0 * noises[::2,::2])
        patch[::2,1::2] = patch[::2,1::2] + (np.sqrt((patch[::2,1::2]*16383.0 + pow(23.5,2))/pow(4.1,2))/16383.0 * noises[::2,1::2])
        patch[1::2,::2] = patch[1::2,::2] + (np.sqrt((patch[1::2,::2]*16383.0 + pow(2.5,2))/pow(0.25,2))/16383.0 * noises[1::2,::2])
        patch[1::2,1::2] = patch[1::2,1::2] + (np.sqrt((patch[1::2,1::2]*16383.0 + pow(2.5,2))/pow(0.25,2))/16383.0 * noises[1::2,1::2])
        
        # patch[::2,::2] = patch[::2,::2] + (np.sqrt((patch[::2,::2]*16383.0 + pow(23.5,2))/pow(4.1,2))/16383.0 * noises[::2,::2])
        # patch[::2,1::2] = patch[::2,1::2] + (np.sqrt((patch[::2,1::2]*16383.0 + pow(6.4,2))/pow(1.01,2))/16383.0 * noises[::2,1::2])
        # patch[1::2,::2] = patch[1::2,::2] + (np.sqrt((patch[1::2,::2]*16383.0 + pow(2.5,2))/pow(0.25,2))/16383.0 * noises[1::2,::2])
        # patch[1::2,1::2] = patch[1::2,1::2] + (np.sqrt((patch[1::2,1::2]*16383.0 + pow(2.5,2))/pow(0.063,2))/16383.0 * noises[1::2,1::2])  
        
        output = patch
        output = np.clip(output,0,1)
        
        mdic = {'raw':output}
        savemat('C:\\Users\\user\\deblurring_dataset\\comparison\\heide_data\\{}.mat'.format(img_name),mdic)
        
        print(output.shape)
    