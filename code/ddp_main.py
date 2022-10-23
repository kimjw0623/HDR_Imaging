import os
import argparse
import datetime
import json

import shutil
import glob
import importlib.util 
import natsort
from option import args

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

import sys
import warnings
warnings.filterwarnings("error")

import dataset
import network_trans as network
from metric import *
from eval_test import eval_model
from real_test import get_real_result
import warnings 
from util import ForkedPdb

sys.path.append("..")

train_gt_dir = '/workspace/dataset/gt_hdr_train_cropped/'
test_gt_dir = '/workspace/dataset/gt_hdr_test/'
train_dataset_dir = '/workspace/dataset/input_train_cropped/'
test_dataset_dir = '/workspace/dataset/input_test/'

train_bayers = [x.split('/')[-1].split('.')[0] for x in sorted(glob.glob('{}*.npy'.format(train_dataset_dir)))]
train_gts = train_bayers.copy()
print(f"Size of train dataset: {len(train_bayers)}")

test_bayers = [x.split('/')[-1].split('.')[0] for x in sorted(glob.glob('{}*.npy'.format(test_dataset_dir)))][:40]
test_gts = test_bayers.copy() 
print(f"Size of test dataset: {len(test_bayers)}")

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
    test_dataset = dataset.Dataset(test_bayers, test_gts, test_dataset_dir, test_gt_dir, args, is_test=True)
    TrainSampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    TestSampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)

    # For DDP, shuffle should false. shuffle is done in Sampler
    training_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                                     num_workers=num_worker, pin_memory=True, sampler=TrainSampler)
    testing_generator = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False,
                                                     num_workers=num_worker, pin_memory=True, sampler=TestSampler)
    
    torch.cuda.empty_cache()
    torch.cuda.set_device(rank)

    net = network.make_model(args, rank).cuda(rank)
    net = DDP(net, device_ids=[rank])

    if args.is_resume:
        print('resume training')
        # Get best ckpt then load
        order_list = os.listdir(args.result_dir + "/ckpt")
        best_ckpt = natsort.natsorted(order_list)[-1]
        checkpoint = torch.load(f'{args.result_dir}/ckpt/{best_ckpt}', map_location='cpu') # 'cpu': prevent memory leakage
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        best_psnr_mu = 36#checkpoint['psnr_mu']
        epoch = checkpoint['epoch']
        iteration = int(epoch * train_dataset.__len__() / (world_size * batch_size))
        best_loss = loss
    
    else:
        print('start new training')
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        best_loss = np.inf
        best_psnr_mu = 0
        epoch = 0
        iteration = 0
        if rank==0:
            make_dir(f'{args.result_dir}')
            shutil.copyfile(f'./network_trans.py',f'{args.result_dir}/network_trans.py')
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
            pred_hdr = net(input_bayer).permute(0,2,3,1)
            loss = F.l1_loss(range_compressor_cuda(pred_hdr), range_compressor_cuda(gt_hdr))

            cur_psnr = batch_psnr(pred_hdr, gt_hdr, 1.0)
            cur_psnr_mu = batch_psnr_mu(pred_hdr, gt_hdr, 1.0)

            loss.backward()
            optimizer.step()
            iteration += 1
            loss_sum = loss.detach().clone()
            dist.all_reduce(loss_sum)
            if rank==0:
                print(f'[Train] Iter: {iteration:06d} / {total_iteration:06d} Loss: {loss_sum.item():06f} <{datetime.datetime.now()}>')
                writer.add_scalar('loss/train', loss_sum.item(), iteration)
                writer.add_scalar('PSNR/train', cur_psnr, iteration)
                writer.add_scalar('PSNR-mu/train', cur_psnr_mu, iteration)

        with torch.no_grad():
            net.eval()
            pred_list = []
            show_list = ['cars_fullshot_000480','cars_fullshot_000690','poker_fullshot_000390','showgirl_02_000360'] # list of image name
            metric_sum_list = torch.zeros(4).cuda(rank)
            valid_loss_sum = 0

            for data_group in testing_generator:
                data_cuda = [x.float().cuda(rank, non_blocking=True) for x in data_group[:2]]
                input_bayer, gt_hdr = data_cuda # input_bayer: (b h w 2)
                pred_hdr = net(input_bayer).permute(0,2,3,1) # pred_hdr: (b h w c)
                input_name = data_group[2][0]

                # Add tensorboard image
                if input_name in show_list:
                    pred_list.append((pred_hdr,input_name))

                # loss with compressed range
                valid_loss = F.l1_loss(range_compressor_cuda(pred_hdr), range_compressor_cuda(gt_hdr))

                cur_psnr = batch_psnr(pred_hdr, gt_hdr, 1.0)
                mu_pred_hdr = range_compressor_cuda(pred_hdr)
                mu_gt_hdr = range_compressor_cuda(gt_hdr)
                cur_psnr_mu = batch_psnr(mu_pred_hdr, mu_gt_hdr,1.0)
                
                pred_hdr = np.clip(np.squeeze(pred_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)
                gt_hdr = np.clip(np.squeeze(gt_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)
                mu_pred_hdr = np.clip(np.squeeze(mu_pred_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)
                mu_gt_hdr = np.clip(np.squeeze(mu_gt_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)

                cur_ssim = calculate_ssim(pred_hdr, gt_hdr)
                cur_ssim_mu = calculate_ssim(mu_pred_hdr, mu_gt_hdr)

                metric_list = torch.Tensor([cur_psnr, cur_ssim, cur_psnr_mu, cur_ssim_mu]).cuda(rank)
                metric_sum_list = torch.add(metric_sum_list, metric_list)

                valid_loss_sum += valid_loss.detach().clone()

            for item in pred_list:
                visual_index = show_list.index(item[1])
                tm_pred_image = tm_mu_law(np.clip(item[0][0,:,:,:].detach().cpu().squeeze(0).numpy(),0,1))
                writer.add_image(f'pred{visual_index}', tm_pred_image, global_step=epoch, dataformats='HWC')

            dist.reduce(valid_loss_sum, 0)
            dist.reduce(metric_sum_list, 0)

        # Show training status and tensorboard settings
        if rank==0:
            valid_loss_sum = valid_loss_sum/len(test_bayers)
            metric_sum_list = torch.div(metric_sum_list, len(test_bayers))

            print(f'[Valid] Iter: {iteration:06d} Loss: {valid_loss_sum.item():06f} <{datetime.datetime.now()}>')
            writer.add_scalar('loss/valid', valid_loss_sum.item(), iteration)
            writer.add_scalar('PSNR/valid', metric_sum_list[0], iteration)
            writer.add_scalar('PSNR-mu/valid', metric_sum_list[2], iteration)
            writer.add_scalar('SSIM/valid', metric_sum_list[1], iteration)
            writer.add_scalar('SSIM-mu/valid', metric_sum_list[3], iteration)

            if metric_sum_list[2] > best_psnr_mu:
                best_psnr_mu = metric_sum_list[2]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'psnr_mu' : best_psnr_mu,
                },f'{args.result_dir}/ckpt/best_psnr_mu.pt')
        
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
    if order_list[-1] == 'last.pt':
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