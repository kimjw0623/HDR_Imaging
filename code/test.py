import shutil
import glob
import imageio.v3 as imageio

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import dataset
import network
from option import args
from metric import *

def main():
    print('Demo')
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = network.make_model(args,0).cuda(dev)

    # Load pre-trained model
    checkpoint = torch.load(f'{args.result_dir}/ckpt/best_psnr_mu.pt', map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])

    input_bayer = '' # (b h w 1)
    gt_hdr = '' # (b h w c)

    pred_hdr = net(input_bayer).permute(0,2,3,1) # pred_hdr: (b h w c)

    cur_psnr = batch_psnr(pred_hdr, gt_hdr, 1.0)
    mu_pred_hdr = range_compressor_cuda(pred_hdr)
    mu_gt_hdr = range_compressor_cuda(gt_hdr)
    cur_psnr_mu = batch_psnr(mu_pred_hdr, mu_gt_hdr,1.0)

    pred_hdr = np.clip(np.squeeze(pred_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)
    gt_hdr = np.clip(np.squeeze(gt_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)
    mu_pred_hdr = np.clip(np.squeeze(mu_pred_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)
    mu_gt_hdr = np.clip(np.squeeze(mu_gt_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)

    # cur_ssim = calculate_ssim(pred_hdr, gt_hdr)
    cur_ssim_mu = calculate_ssim(mu_pred_hdr, mu_gt_hdr)

    print(f"PSNR: {cur_psnr}, PSNR-mu: {cur_psnr_mu}, SSIM-mu {cur_ssim_mu}")

if __name__ == '__main__':
    main()