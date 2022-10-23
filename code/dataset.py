import random
import torch
import numpy as np
from util import ForkedPdb

class Dataset(torch.utils.data.Dataset):
    def __init__(self, bayer_ids, gt_ids, bayer_path, gt_path, args, is_test=False):
        self.bayer_ids = bayer_ids
        self.gt_ids = gt_ids
        self.bayer_path = bayer_path
        self.gt_path = gt_path
        self.input_size = args.input_size
        self.is_noise = args.is_noise
        self.noise_value = args.noise_value
        self.is_log_scale = args.is_log_scale
        self.is_origin_size = args.is_origin_size
        self.is_hdr = args.is_hdr
        self.is_train = args.is_train
        self.comparison = args.comparison
        self.is_test = is_test
        
        expo = np.zeros((args.input_size,args.input_size))
        expo[::2,::2] = 1.0
        expo[::2,1::2] = 4.0
        expo[1::2,::2] = 4.0
        expo[1::2,1::2] = 16.0
        self.exp_level_arr = expo

    def __len__(self):
        return len(self.bayer_ids)

    def __getitem__(self, index):
        bayer_id = self.bayer_ids[index]
        input_bayer, gt_hdr = self._load_file(bayer_id)

        # if self.is_origin_size:
        #     gt_hdr = self._get_origin_size(gt_hdr)

        if not self.is_test:
            input_bayer, gt_hdr = self._augmentation(input_bayer, gt_hdr)

        if self.is_noise:
            if self.is_train:
                input_bayer = self._add_train_noise(input_bayer)
            else:
                input_bayer = self._add_noise(input_bayer)
                
        input_bayer = np.clip(input_bayer,0,1)
        gt_hdr = np.clip(gt_hdr,0,1)

        # if self.is_log_scale:
        #     input_bayer = np.log2(input_bayer*mu+1.0)/np.log2(mu + 1.0)
        #     gt_hdr = np.log2(gt_hdr*mu+1.0)/np.log2(mu + 1.0)
        
        if self.comparison:
            h,w = input_bayer.shape
            if self.is_test:
                expo = np.zeros((h,w))
                expo[::2,::2] = 1.0
                expo[::2,1::2] = 4.0
                expo[1::2,::2] = 4.0
                expo[1::2,1::2] = 16.0
                self.exp_level_arr = expo

            input_bayer_gamma = np.expand_dims((input_bayer) / (self.exp_level_arr + 1e-8), axis=-1)
            input_bayer = np.expand_dims(input_bayer,axis=-1)
            input_bayer = np.concatenate((input_bayer,input_bayer_gamma),-1)
            assert(input_bayer.shape==(h,w,2))

        # print(gt_hdr.shape)

        return input_bayer, gt_hdr, bayer_id
    
    def _load_file(self, bayer_id):
        bayer_name = '{}/{}.npy'.format(self.bayer_path,bayer_id)
        gt_name = '{}/{}.npy'.format(self.gt_path,bayer_id)
        bayer = np.load(bayer_name,allow_pickle=True)
        gt = np.load(gt_name,allow_pickle=True)
        return bayer, gt

    '''
    Get p*p size patch (GT: 4*p*p*3) All patch should start with R (exposure level 0) pixel / p has to be 4*k shape
    '''
    def _get_patch(self, bayer, gt):
        p = self.input_size 
        h, w = bayer.shape[:2]

        y = random.randrange(0, (h - p)//4 + 1)
        x = random.randrange(0, (w - p)//4 + 1)

        bayer_patch = bayer[4*y:4*y + p, 4*x:4*x + p] 
        gt_patch = gt[4*y:4*y + p, 4*x:4*x + p, :]

        if self.is_noise:
            bayer_patch = self._add_noise(bayer_patch)

        return bayer_patch, gt_patch
    
    '''Add random augmentation: Flip, Rotation.''' 
    def _augmentation(self, bayer, gt_hdr, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5 

        if hflip: bayer = bayer[:, ::-1].copy()
        if vflip: bayer = bayer[::-1, :].copy()
        if rot90: bayer = np.transpose(bayer).copy()

        if hflip: gt_hdr = gt_hdr[:, ::-1, :].copy()
        if vflip: gt_hdr = gt_hdr[::-1, :, :].copy()
        if rot90: gt_hdr = np.transpose(gt_hdr, (1, 0, 2)).copy()
        
        return bayer, gt_hdr

    '''Add noise to raw bayer iamge for training.'''
    def _add_train_noise(self, patch):
        patch_h, patch_w = patch.shape
        noises = np.random.normal(0.0, scale=random.uniform(self.noise_value*0.8,self.noise_value*1.2), size=(patch_h, patch_w))
        patch += np.sqrt((patch*16383.0 + pow(34.9,2))/pow(5.04,2))/16383.0 * noises
        return patch

    '''Add noise to raw bayer iamge for testing.'''
    def _add_noise(self, patch):
        noise_h, noise_w = patch.shape
        noises = np.random.normal(0.0, scale=self.noise_value, size=(noise_h, noise_w))
        patch += np.sqrt((patch*16383.0 + pow(34.9,2))/pow(5.04,2))/16383.0 * noises
        return patch
    
    def _get_origin_size(self, gt):
        N,H,W,C = gt.shape
        sampled_gt = np.zeros((N,H//2,W//2,C))
        sampled_gt[0,:,:,:] = gt[0,::2,::2,:]
        sampled_gt[1,:,:,:] = gt[1,::2,1::2,:]
        sampled_gt[2,:,:,:] = gt[2,1::2,::2,:]
        sampled_gt[3,:,:,:] = gt[3,1::2,1::2,:]
        return sampled_gt
