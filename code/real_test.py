import torch
import numpy as np
import os
import model
import loss
import dataset
import glob
import time
import cv2
import imageio
import shutil
import metric
from tqdm import tqdm
from metric import calc_psnr
from metric import ssim
from einops import rearrange

# Get result HDR image of real test dataset
def get_real_result(args, model, test_bayers, test_dataset_dir, test_gt_dir, device, new_path):
    is_hdrvideo = 1
    total_time = 0
    bayer_number = 0
    model.eval()
    with torch.no_grad():
        for image_name in glob.glob(f'/workspace/hdrvideo/*.npy'):
            bayer_number += 1
            filename = os.path.splitext(os.path.split(image_name)[-1])[0]
            bayer = np.load(image_name)
            print(filename)
            print(image_name)
            print(bayer.shape)
            if is_hdrvideo:
                noise_h, noise_w = bayer.shape
                noises = np.random.normal(0.0, scale=args.noise_value, size=(noise_h, noise_w))
                bayer += (np.sqrt((bayer*16383.0 + pow(23.5,2))/pow(4.1,2))/16383.0 * noises)
                bayer = bayer[:1024,:]
            bayer = np.clip(np.expand_dims(bayer,0),0,1)

            if is_hdrvideo:
                if args.is_log_scale:
                    bayer = metric.tm_mu_law(bayer)

                bayer = torch.from_numpy(bayer)
                bayer = bayer.to(device)
                curr_time = time.time()
                res = model(bayer)
                total_time += time.time()-curr_time
                print(time.time()-curr_time)
                res = res.cpu()
                res = torch.squeeze(res).permute(1,2,0).detach().numpy()
                res = np.clip(res,0,1)
                
                if args.is_log_scale:
                    mu = 5000
                    true_res = (pow((mu + 1.0),res)-1)/mu
                else:
                    true_res = res
                final_image = true_res.copy()
                final_image[:,:,0] = true_res[:,:,2]
                final_image[:,:,2] = true_res[:,:,0]

            else:
                final_image = np.zeros((1792*2,2816*2,3),np.float32)
                # bayer = torch.from_numpy(bayer)

                for i in range(4):
                    offset_w = i%2
                    offset_h = i//2
                    temp_bayer = bayer[:,(offset_h*1792):((offset_h+1)*1792),(82+offset_w*2816):(82+(offset_w+1)*2816)]
                    if args.is_log_scale:
                        print('log')
                        print(bayer.shape)
                        temp_bayer = metric.tm_mu_law(temp_bayer)

                    temp_bayer = torch.from_numpy(temp_bayer)
                    temp_bayer = temp_bayer.to(device)
                    start_time = time.time()
                    res = model(temp_bayer)
                    print(time.time()-start_time)
                    res = res.cpu()
                    res = torch.squeeze(res).permute(1,2,0).detach().numpy()
                    res = np.clip(res,0,1)
                    
                    if args.is_log_scale:
                        mu = 5000
                        true_res = (pow((mu + 1.0),res)-1)/mu
                    else:
                        true_res = res

                    final_image[offset_h*1792:(offset_h+1)*1792,offset_w*2816:(offset_w+1)*2816,:] = true_res

                temp = final_image.copy()
                final_image[:,:,0] = temp[:,:,2]
                final_image[:,:,2] = temp[:,:,0]

                # final_image[:,:,1] = temp[:,:,2]
                # final_image[:,:,2] = temp[:,:,1]

                # final_image[:,:,1] = temp[:,:,0]
                # final_image[:,:,0] = temp[:,:,2]


            imageio.imwrite('{}/{}.exr'.format(new_path,filename), final_image)
            tm_final_image = metric.tm_mu_law(final_image)
            imageio.imwrite('{}/{}.png'.format(new_path,filename), np.uint8(tm_final_image*255))
    

    print(total_time/bayer_number)