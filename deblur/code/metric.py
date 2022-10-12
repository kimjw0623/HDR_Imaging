import os
import math
import time
import numpy as np
import torch
import cv2

def tm_mu_law(image, mu = 5000):
    assert np.amin(image) >= 0
    assert np.amax(image) <= 1
    tm_image = np.log2(image*mu+1.0)/np.log2(mu + 1.0)
    return tm_image

def calc_psnr(pred, gt, imax = 1.0):
    b, _, _, _ = pred.shape
    psnr = 0
    for i in range(b):
        mse = torch.mean((gt[i] - pred[i]) ** 2)# dtype=np.float64)
        mse = mse.cpu().detach().numpy()
        psnr += 20 * np.log10(imax) - 10 * np.log10(mse)
    psnr = psnr/b
    
    return psnr

def ldr_calc_psnr(pred, gt, imax = 1.0):
    b, _, _, _, _ = pred.shape
    psnr = 0
    for i in range(b):
        mse = torch.mean((gt[i] - pred[i]) ** 2)# dtype=np.float64)
        mse = mse.cpu().detach().numpy()
        psnr += 20 * np.log10(imax) - 10 * np.log10(mse)
    psnr = psnr/b
    
    return psnr

def color_psnr(img1, img2, imax=1.0):
    mse = np.mean((img1 - img2) ** 2)
    psnr = 20 * np.log10(imax) - 10 * np.log10(mse)

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