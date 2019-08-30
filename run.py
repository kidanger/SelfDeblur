# coding: utf-8

from networks.skip import skip
from networks.fcn import fcn
import torch
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.deconv_utils import *
from TVLoss import TVLoss

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
# dtype = torch.FloatTensor

warnings.filterwarnings("ignore")

def deblur(input, kernel_size, output, outputk=None, tv=0, lr=0.01,
           reg_noise_std=0.001, num_iter=5000, normalization=1):
    INPUT = 'noise'
    pad = 'reflection'

    kernel_size = [kernel_size, kernel_size]

    import iio
    imgs = iio.read(input)/normalization
    imgs = imgs.transpose((2, 0, 1))
    y = np_to_torch(imgs).type(dtype)

    img_size = imgs.shape

    padh, padw = kernel_size[0]-1, kernel_size[1]-1

    '''
    x_net:
    '''
    input_depth = 8

    net_input = get_noise(input_depth, INPUT, (img_size[1]+padh, img_size[2]+padw)).type(dtype).detach()

    net = skip(input_depth, 3,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16,  16,  16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
    net = net.type(dtype)

    '''
    k_net:
    '''
    n_k = 200
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype).detach()
    net_input_kernel = net_input_kernel.squeeze()

    net_kernel = fcn(n_k, kernel_size[0]*kernel_size[1])
    net_kernel = net_kernel.type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    L1 = torch.nn.L1Loss(reduction='sum').type(dtype)
    tv_loss = TVLoss(tv_loss_weight=tv).type(dtype)

    # optimizer
    optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

    ### start SelfDeblur
    for step in tqdm(range(num_iter)):

        # input regularization
        net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input)
        out_k = net_kernel(net_input_kernel)

        out_k_m = out_k.view(-1,1,kernel_size[0],kernel_size[1])
        out_y = nn.functional.conv2d(out_x, out_k_m.expand((3,-1,-1,-1)), padding=0, bias=None, groups=3)

        total_loss = mse(out_y, y) + tv_loss(out_x)
        total_loss.backward()
        optimizer.step()

    out_x_np = torch_to_np(out_x)
    out_x_np = out_x_np.squeeze()
    out_x_np = out_x_np[:, padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
    out_x_np = out_x_np.transpose((1, 2, 0))
    iio.write(output, out_x_np*normalization)

    if outputk:
        out_k_np = torch_to_np(out_k_m)
        out_k_np = out_k_np.squeeze()
        iio.write(outputk, out_k_np)


if __name__ == '__main__':
    import fire
    fire.Fire(deblur)

