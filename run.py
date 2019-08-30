
# coding: utf-8


from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from networks.skip import skip
from networks.fcn import fcn
import cv2
import torch
import torch.optim
import glob
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.deconv_utils import *
from TVLoss import TVLoss

parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[512, 512], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[21, 21], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str, default="datasets/levin/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/levin_iter5k/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='lfrequency to save results')
opt = parser.parse_args()
#print(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
# dtype = torch.FloatTensor

warnings.filterwarnings("ignore")

files_source = glob.glob(os.path.join(opt.data_path, '*.tif'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

# start #image
for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    tv_weight = 0e-6 # usually large tv_weight for high noise level. And for Levin dataset (sigma -> 0), TVLoss is not necessary
    num_iter = opt.num_iter
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    if imgname.find('kernel1') != -1:
        opt.kernel_size = [17, 17]
    if imgname.find('kernel2') != -1:
        opt.kernel_size = [15, 15]
    if imgname.find('kernel3') != -1:
        opt.kernel_size = [13, 13]
    if imgname.find('kernel4') != -1:
        opt.kernel_size = [27, 27]
    if imgname.find('kernel5') != -1:
        opt.kernel_size = [11, 11]
    if imgname.find('kernel6') != -1:
        opt.kernel_size = [19, 19]
    if imgname.find('kernel7') != -1:
        opt.kernel_size = [21, 21]
    if imgname.find('kernel8') != -1:
        opt.kernel_size = [21, 21]

    # _, imgs = get_image(path_to_image, -1) # load image and convert to np.
    import iio
    imgs = iio.read(path_to_image)
    imgs = imgs.transpose((2, 0, 1))
    y = np_to_torch(imgs).type(dtype)

    img_size = imgs.shape
    print(imgname, img_size, y.size())
    # ######################################################################
    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw

    '''
    x_net:
    '''
    input_depth = 8

    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype).detach()

    net = skip(input_depth, 3,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16,  16,  16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    # if os.path.exists(os.path.join(opt.save_path, "%s_xnet.pth" % imgname)):
        # net = torch.load(os.path.join(opt.save_path, "%s_xnet.pth" % imgname))

    net = net.type(dtype)

    '''
    k_net:
    '''
    n_k = 200
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype).detach()
    net_input_kernel.squeeze_()

    net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])

    if os.path.exists(os.path.join(opt.save_path, "%s_knet.pth" % imgname)):
        net_kernel = torch.load(os.path.join(opt.save_path, "%s_knet.pth" % imgname))

    net_kernel = net_kernel.type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    L1 = torch.nn.L1Loss(reduction='sum').type(dtype)
    tv_loss = TVLoss(tv_loss_weight=tv_weight)

    # optimizer
    optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

    ### start SelfDeblur
    for step in tqdm(range(num_iter)):

        # input regularization
        net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()
        # net_input_kernel = net_input_kernel_saved + reg_noise_std*torch.zeros(net_input_kernel_saved.shape).type_as(net_input_kernel_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input)
        out_k = net_kernel(net_input_kernel)

        out_k_m = out_k.view(-1,1,opt.kernel_size[0],opt.kernel_size[1])
        out_y = nn.functional.conv2d(out_x, out_k_m.expand((3,-1,-1,-1)), padding=0, bias=None, groups=3)

        total_loss = mse(out_y, y) + tv_loss(out_x) #+ tv_loss2(out_k_m)
        total_loss.backward()
        optimizer.step()

        if step % opt.save_frequency == 0:
            #print('Iteration %05d' %(step+1))
            save_path = os.path.join(opt.save_path, '%s_%d_x.tif'%(imgname,step))
            out_x_np = torch_to_np(out_x)
            out_x_np = out_x_np.squeeze()
            out_x_np = out_x_np[:, padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
            out_x_np = out_x_np.transpose((1, 2, 0))
            iio.write(save_path, out_x_np)

            save_path = os.path.join(opt.save_path, '%s_%d_k.tif'%(imgname,step))
            out_k_np = torch_to_np(out_k_m)
            out_k_np = out_k_np.squeeze()
            iio.write(save_path, out_k_np)

            # torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))
            # torch.save(net_kernel, os.path.join(opt.save_path, "%s_knet.pth" % imgname))

