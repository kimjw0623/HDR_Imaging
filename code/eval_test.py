import time
import imageio
import metric
import dataset
import numpy as np
from tqdm import tqdm
from metric import ssim
from einops import rearrange, reduce, repeat
import torch

'''Get result HDR image of test dataset'''
def eval_model(args, model, test_bayers, test_dataset_dir, test_gt_dir, device, new_path):
    
    # generate dataloader
    test_params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 1}
    test_dataset = dataset.Dataset(test_bayers, test_bayers, test_dataset_dir, test_gt_dir, args)
    testing_generator = torch.utils.data.DataLoader(test_dataset, **test_params)

    time_elapsed = 0
    sum_psnr = 0
    sum_mu_psnr = 0
    sum_ssim = 0
    sum_mu_ssim = 0
    mu = 5000

    model.eval()
    with torch.no_grad():
        for data_group in tqdm(testing_generator):
            # _, _, bayer_name = data_group
            # bayer_name = bayer_name[0]#.squeeze()
            # # print(bayer_name)
            # if bayer_name != 'cars_fullshot_000384-51':
            #     continue

            # layer_list = ["module.transformerBlock21.mlp","module.transformerBlock11.mlp"]# ["module.sr_conv","module.transformerBlock11","module.transformerBlock21"] # ,"module.final_conv"
            # custom_features = feature_util.FeatureExtractor(model, layers = layer_list)
            
            # data_cuda = [x.float().cuda(0, non_blocking=True) for x in data_group[:2]]
            # input_bayer, gt_hdr = data_cuda

            # features = custom_features(input_bayer)
            # for name, output in features.items():
            #     print(output.shape)
            #     if name != "module.sr_conv":
            #         output = torch.squeeze(output)
            #     # E, C, H, W = output.shape
            #     #if name == "module.transformerBlock11.mlp":
            #     output = rearrange(output, '(nH nW) (wH wW) (Hd nE C) -> nE nH nW C Hd wH wW',nH=16,nW=16,wH=8,wW=8,nE=4,C=8)
            #         # x = x.view(128//8, 128//8, 4, 8, 8, 4, 8
            #         # self.window_size[0], self.window_size[1], 4, self.exp_channels//self.num_heads) # nW -> nH * nW (width)
            #         # x = x.permute(0, 6, 7, 3, 1, 4, 2, 5).contiguous().view(B, 4, self.exp_channels, Hp, Wp) 
            #     print(output)
            #     output = output.cpu().detach().numpy()
            #     output = ((output + 1.0)/2.0)*255.0
                
            #     nE, nH, nW, C, Head, wH, wW = output.shape
            #     for height in range(nH):
            #         for width in range(nW):
            #             feature_2d = np.zeros((wH*8+2*7+2*3,wW*16+2*15))
            #             for channel in range(C):
            #                 for head in range(Head):
            #                     offset_h = (head*C + channel)//16
            #                     offset_w = (head*C + channel)%16
            #                     x = offset_w*wW + offset_w*2
            #                     y = offset_h*wH + offset_h*2 + head*2
            #                     feature_2d[y:y+wW,x:x+wH] = output[0,height,width,channel,head,:,:]
            #                     print(feature_2d)
            #             imageio.imwrite('/workspace/feature/feature_{}_exp0_{}_{}_{}.png'.format(name,height,width,bayer_name),np.uint8(feature_2d))

            #     block = False
            #     if block:
            #         for num_exp in range(E):
            #             feature_2d = np.zeros((H*4+2*3,W*8+2*7))
            #             for channel in range(C):
            #                 offset_h = channel // 8
            #                 offset_w = channel % 8
            #                 x = offset_w*W + offset_w*2
            #                 y = offset_h*H + offset_h*2
            #                 feature_2d[y:y+H,x:x+W] = output[num_exp,channel,:,:]
            #                 # print(output[num_exp,channel,:,:])
            #             print(feature_2d)
            #             imageio.imwrite('/workspace/feature/feature_{}_{}_{}.png'.format(name,num_exp,bayer_name),np.uint8(feature_2d))
            # # print({name: output.shape for name, output in features.items()})
        
            _, _, bayer_name = data_group
            bayer_name = bayer_name[0]#.squeeze()
            print(bayer_name)
            data_cuda = [x.float().cuda(0, non_blocking=True) for x in data_group[:2]]
            input_bayer, gt_hdr = data_cuda
            start = time.time()
            pred_hdr = model(input_bayer).permute(0,2,3,1) # (b h w c)
            time_elapsed += time.time()-start
            
            sum_mu_psnr += metric.calc_psnr(pred_hdr, gt_hdr)
    
            pred_hdr = torch.squeeze(pred_hdr.cpu()).detach().numpy()
            input_bayer = torch.squeeze(input_bayer.cpu()).detach().numpy()
            gt_hdr = torch.squeeze(gt_hdr.cpu()).detach().numpy()

            sum_mu_ssim += ssim(np.uint8(pred_hdr*255),np.uint8(gt_hdr*255))
            
            imageio.imwrite('{}/pred_ldr_{}.png'.format(new_path,bayer_name),np.uint8(pred_hdr*255))
            # imageio.imwrite('{}/bayer_ldr_{}.png'.format(new_path,bayer_name),np.uint8(bayer*255))
            # imageio.imwrite('{}/gt_ldr_{}.png'.format(new_path,bayer_name),np.uint8(gt_hdr*255))

            if args.is_log_scale:
                true_res = (pow(2,np.log2(mu + 1.0)*pred_hdr)-1)/mu
                gt_hdr = (pow(2,np.log2(mu + 1.0)*gt_hdr)-1)/mu
            else:
                true_res = pred_hdr
            imageio.imwrite('{}/pred_{}.exr'.format(new_path,bayer_name), true_res)
            
            sum_psnr += metric.color_psnr(true_res, gt_hdr)

            print('avg time elapsed: {}'.format(time_elapsed/len(test_dataset)))
            print('avg psnr: {}'.format(sum_psnr/len(test_dataset)))
            print('avg mu_psnr: {}'.format(sum_mu_psnr/len(test_dataset)))
            print('avg mu_ssim: {}'.format(sum_mu_ssim/len(test_dataset)))

            with open("{}/result.txt".format(new_path), 'w') as f:
                f.write('avg time elapsed: {}\n'.format(time_elapsed/len(test_dataset)))
                f.write('avg psnr: {}\n'.format(sum_psnr/len(test_dataset)))
                f.write('avg mu_psnr: {}\n'.format(sum_mu_psnr/len(test_dataset)))
                f.write('avg mu_ssim: {}'.format(sum_mu_ssim/len(test_dataset)))
            f.close()

    return
