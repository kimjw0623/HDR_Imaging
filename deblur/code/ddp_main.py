import os
import argparse
import datetime
import json
import math

import imageio
import shutil
import glob
import importlib.util 
import natsort
from option import args
import imp

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary as summary_

import sys
import warnings
warnings.filterwarnings("error")

import network
import dataset
import metric
import feature_util
from eval_test import eval_model
from real_test import get_real_result

sys.path.append("..")
imageio.plugins.freeimage.download()


train_gt_dir = '/workspace/gt_deblur/gt_hdr_crops/'
test_gt_dir = '/workspace/gt_deblur/test_gt_hdr_crops/'
train_dataset_dir = '/workspace/deblur/bayer_hdr_crops/'
test_dataset_dir = '/workspace/deblur/test_bayer_hdr_crops/'

train_bayers = [x.split('/')[-1].split('.')[0] for x in glob.glob('{}*.npy'.format(train_dataset_dir))]
train_gts = train_bayers.copy()

test_bayers = [x.split('/')[-1].split('.')[0] for x in glob.glob('{}*.npy'.format(test_dataset_dir))]
test_bayers = [test_bayers[i] for i in range(len(test_bayers)) if i%23 == 0] # Pick random test image
# test_bayers = ['cars_fullshot_000384-51']
test_gts = test_bayers.copy() 

def make_dir(dir_path):
    new_dir = os.path.normpath(dir_path)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir

def main(args):
    args.result_dir = make_dir(f'./runs/{args.result_dir}')
    with open(f'{args.result_dir}/run_config.txt', 'w') as f:
        f.write(__file__)
        f.write('\n')
        json.dump(args.__dict__, f, indent=4)

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

def main_worker(rank, world_size, args):
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(777)

    torch.distributed.init_process_group(
        backend='nccl', # Recommended backend for DDP on GPU
        init_method=f'tcp://127.0.0.1:7777', # Port number should be empty port
        world_size=world_size,
        rank=rank)
    print(f'{rank+1}/{world_size} process initialized.')
    num_worker = 0 # Should be 0
    batch_size = args.batch_size # per 1 process, refer total_iteration below

    train_dataset = dataset.Dataset(train_bayers, train_gts, train_dataset_dir, train_gt_dir, args) 
    test_dataset = dataset.Dataset(test_bayers, test_gts, test_dataset_dir, test_gt_dir, args)
    TrainSampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)

    # For DDP, shuffle should false. shuffle is done in Sampler
    training_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                                     num_workers=num_worker, pin_memory=True, sampler=TrainSampler)
    testing_generator = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False,
                                                     num_workers=num_worker, pin_memory=True)
    
    torch.cuda.empty_cache()
    torch.cuda.set_device(rank)

    net = network.make_model(args).cuda(rank)
    net = DDP(net, device_ids=[rank])

    if args.is_resume:
        print('resume training')
        # Get best ckpt then load
        order_list = os.listdir("runs/" + args.result_dir + "/ckpt")
        best_ckpt = natsort.natsorted(order_list)[-1]
        checkpoint = torch.load(f'{args.result_dir}/ckpt/{best_ckpt}', map_location='cpu') # 'cpu': prevent memory leakage
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        iteration = int(epoch * train_dataset.__len__() / (world_size * batch_size))
        best_loss = loss
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print('start new training')
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        best_loss = np.inf
        epoch = 0
        iteration = 0
        if rank==0:
            make_dir(f'{args.result_dir}')
            shutil.copyfile(f'./network.py',f'{args.result_dir}/network.py')
            make_dir(f'{args.result_dir}/tb')
            make_dir(f'{args.result_dir}/ckpt')
    
    writer = SummaryWriter(log_dir=f'{args.result_dir}/tb')
    train_state = True
    total_iteration = int(args.num_epoch * train_dataset.__len__() / (world_size * batch_size))

    print(f'Train Start on GPU {rank}')
    while train_state:
        net.train()
        TrainSampler.set_epoch(epoch) # For random sampling differ by epoch
        for data_group in training_generator:
            data_cuda = [x.float().cuda(rank, non_blocking=True) for x in data_group[:2]]
            input_bayer, gt_hdr = data_cuda
            optimizer.zero_grad()
            if args.is_hdr:
                pred_hdr = net(input_bayer).permute(0,2,3,1)
                loss = F.l1_loss(pred_hdr, gt_hdr)
                cur_psnr = metric.calc_psnr(pred_hdr, gt_hdr)
            else:
                pred_hdr = net(input_bayer).permute(0,1,3,4,2)
                loss = F.l1_loss(pred_hdr, gt_hdr)
                cur_psnr = metric.ldr_calc_psnr(pred_hdr, gt_hdr)
            loss.backward()
            optimizer.step()
            iteration += 1
            loss_sum = loss.detach().clone()
            dist.all_reduce(loss_sum)
            if rank==0:
                print(f'[Train] Iter: {iteration:06d} / {total_iteration:06d} <{datetime.datetime.now()}>')
                writer.add_scalar('loss/train', loss_sum.item(), iteration)
                writer.add_scalar('PSNR/train', cur_psnr, iteration)
        
        with torch.no_grad():
            net.eval()
            pred_list = []
            valid_loss_sum = 0
            psnr_sum = 0
            for data_group in testing_generator:
                data_cuda = [x.float().cuda(rank, non_blocking=True) for x in data_group[:2]]
                input_bayer, gt_hdr = data_cuda # input_bayer: (b c h w)
                if args.is_hdr:
                    pred_hdr = net(input_bayer).permute(0,2,3,1) # (b h w c)
                    pred_list.append(pred_hdr)
                    valid_loss = F.l1_loss(pred_hdr, gt_hdr)
                    cur_psnr = metric.calc_psnr(pred_hdr, gt_hdr)
                else:
                    pred_hdr = net(input_bayer).permute(0,1,3,4,2) # (n h w c)
                    pred_list.append(pred_hdr)
                    valid_loss = F.l1_loss(pred_hdr, gt_hdr)
                    cur_psnr = metric.ldr_calc_psnr(pred_hdr, gt_hdr)
                psnr_sum += cur_psnr
                valid_loss_sum += valid_loss.detach().clone()

            valid_loss_sum = valid_loss_sum/len(testing_generator)
            dist.all_reduce(valid_loss_sum)
            psnr_sum = psnr_sum/len(testing_generator)

        # Show training status and tensorboard settings
        if rank==0:
            print(f'[Valid] Iter: {iteration:06d} <{datetime.datetime.now()}>')
            writer.add_scalar('loss/valid', valid_loss_sum.item(), iteration)
            writer.add_scalar('PSNR/valid', psnr_sum, iteration)
            if args.is_hdr:
                mu = 5000
                show_list = [6,11,-5,-16]
                pred_img_list = []
                for i in show_list:
                    pred_img = pred_list[i].permute(0,3,1,2).detach().cpu().squeeze()
                    pred_img = np.clip(pred_img,0,1)
                    pred_img_list.append(pred_img)
                pred_imgs = np.stack((pred_img_list), axis = 0)
                writer.add_images('pred0', pred_imgs, global_step=iteration, dataformats='NCHW')

            if valid_loss_sum.item() < best_loss:
                best_loss = valid_loss_sum.item()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                },f'{args.result_dir}/ckpt/iter_{iteration}.pt')
        
        epoch += 1
        if epoch >= args.num_epoch: train_state=False
        dist.barrier()
    
    # Last epoch
    if rank==0:
        writer.close()
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
        },f'{args.result_dir}/ckpt/last.pt')
        
    dist.destroy_process_group()

def evaluate(args):
    # Import network of the saved model (when the path contains dot: "noise_0.25" is not a valid module name)
    spec = importlib.util.spec_from_file_location(
        name="network", 
        location= "runs/" + args.result_dir + "/network.py",
    )
    network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(network)
    sys.modules['network'] = network

    # Get best ckpt
    order_list = os.listdir("runs/" + args.result_dir + "/ckpt")
    order_list = natsort.natsorted(order_list)
    if order_list[-1] == 'last.pt'
        best_ckpt = order_list[-2]
    else:
        best_ckpt = order_list[-1]

    rank = 0
    torch.distributed.init_process_group(
        backend='nccl', # Recommended backend for DDP on GPU
        init_method=f'tcp://127.0.0.1:7777', # Port number should be empty port
        world_size=1,
        rank=0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(0)

    net = network.make_model(args) # from network module
    checkpoint = torch.load(f'./runs/{args.result_dir}/ckpt/{best_ckpt}', map_location='cpu')
    print(f"ckpt {best_ckpt} selected!")
    net = net.cuda(rank)
    net = DDP(net, device_ids=[rank])
    net.load_state_dict(checkpoint['model_state_dict'])

    print_modules = False
    if print_modules:
        for idx, m in enumerate(net.named_modules()):
            print(idx, '->', m)

    # custom_features = feature_util.FeatureExtractor(net, layers["module.sr_conv"])
    # features = custom_features()

    # target_layer = dict([*net.named_modules()])['module.sr_conv']
    # print(target_layer)

    eval_dir = make_dir(f'./runs/{args.result_dir}/result-hdrvideo')
    # eval_model(args, net, test_bayers, test_dataset_dir, test_gt_dir, device, eval_dir)
    get_real_result(args, net, test_bayers, test_dataset_dir, test_gt_dir, device, eval_dir)

    dist.destroy_process_group()
    
if __name__=='__main__':
    if not args.is_train:
        print('eval start')
        evaluate(args)
    else:
        print('train start')
        main(args)