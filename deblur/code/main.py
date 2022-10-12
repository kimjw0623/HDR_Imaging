from operator import index
import torch
import numpy as np
import os
import model
import loss
import dataset
import trainer
import glob
import cv2
import imageio
import shutil
from metric import color_psnr
from eval_test import eval_model
from option import args
from torch.utils.tensorboard import SummaryWriter
import torchsummary
#from torchvision import datasets, transforms

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print("Cuda device: {}".format(device))

imageio.plugins.freeimage.download()

# Parameters
train_params = {'batch_size': args.batch_size,
                'shuffle': True,
                'num_workers': 0,
                }

test_params = {'batch_size': args.batch_size,
                'shuffle': True,
                'num_workers': 0,
                }


train_gt_dir = 'C:\\Users\\user\\deblurring_dataset\\ours\\dataset_1248\\gt_crops\\'# "/workspace/froehlich_dataset_4.0_1.0/train_exr/"
test_gt_dir = 'C:\\Users\\user\\deblurring_dataset\\ours\\dataset_1248\\test_gt_crops\\'#"/workspace/froehlich_dataset_4.0_1.0/test_exr/"
train_dataset_dir = 'C:\\Users\\user\\deblurring_dataset\\ours\\dataset_1248\\bayer_crops\\'
test_dataset_dir = 'C:\\Users\\user\\deblurring_dataset\\ours\\dataset_1248\\test_bayer_crops\\'

# Datasets
# Input Bayer: (H,W,1) 14bits Bayer image [0,1] .npy files
# 4 LDR images: (H,W,3,4) GT HDR image*(1,2,4,8) [0,1]
# Get list of bayer image names
# For linux:
# train_bayers = [x.split('/')[-1].split('.')[0] for x in glob.glob('{}*.npy'.format(train_dataset_dir))]
# For window:
train_bayers = [x.split('\\')[-1].split('.')[0] for x in glob.glob('{}*.npy'.format(train_dataset_dir))]
train_gts = train_bayers.copy() # List of gt HDR image names
print(train_gts[:10])

test_bayers = [x.split('\\')[-1].split('.')[0] for x in glob.glob('{}*.npy'.format(test_dataset_dir))]
test_bayers = [test_bayers[i] for i in range(len(test_bayers)) if i%409 == 0]
test_gts = test_bayers.copy() 
print(test_gts[:10])

if __name__ == '__main__':
    # Generators: dataset loader
    train_dataset = dataset.Dataset(train_bayers, train_gts, train_dataset_dir, train_gt_dir, 
                                    args.patch_size, args.noise_value, args.noise, args.log_scale, False)
    test_dataset = dataset.Dataset(test_bayers, test_gts, test_dataset_dir, test_gt_dir,
                                    args.patch_size, args.noise_value, args.noise, args.log_scale, True)

    training_generator = torch.utils.data.DataLoader(train_dataset, **train_params)
    testing_generator = torch.utils.data.DataLoader(test_dataset, **test_params)
    
    print(len(training_generator))
    print(len(testing_generator))

    model = model.Model(args, device).cuda()
    torchsummary.summary(model, (256, 256))
    model_name = 'froelich_lr{}_epochs{}_patchsize{}_batchsize{}_noise{}_noisevalue{}_log_scale{}_{}'.format(
            args.lr,args.epochs,args.patch_size,args.batch_size,args.noise,args.noise_value,args.log_scale,args.name)

    writer = SummaryWriter(comment=model_name)

    loss = loss.Loss()
    t = trainer.Trainer(args, training_generator, testing_generator, model, loss, writer)

    new_path = 'trained_models/{}'.format(model_name)
    new_dir = os.path.normpath(new_path)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    shutil.copyfile('./network.py','{}/network.py'.format(new_dir))

    while not t.terminate():
        t.train()
        t.test()
    
    writer.close()

    model.load_state_dict(torch.load('trained_models/best_psnr_{}.pt'.format(model_name)))
    model.eval()

    # Get PSNR test dataset
    eval_model(args, model, test_bayers, test_dataset_dir, test_gt_dir, device, new_dir)



