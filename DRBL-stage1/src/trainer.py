import os
import math
from decimal import Decimal

import utility
import IPython
import torch.nn.functional as F
import torch.nn as nn

import torchvision
import torch
from torch.autograd import Variable
from tqdm import tqdm
import pytorch_ssim
import torchvision
from PIL import Image
import numpy as np

def tensor_save_rgbimage(tensor, filename, cuda=False):
	if cuda:
		img = tensor.clone().cpu().clamp(0, 255).numpy()
	else:
		img = tensor.clone().clamp(0, 255).numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	img = Image.fromarray(img)
	img.save(filename)

class GaussianBlur5(nn.Module):
    def __init__(self):
        super(GaussianBlur5, self).__init__()
        kernel = [[0.0007,    0.0063,    0.0129,    0.0063,    0.0007],
                  [0.0063,    0.0543,    0.1116,    0.0543,    0.0063],
                  [0.0129,    0.1116,    0.2292,    0.1116,    0.0129],
                  [0.0063,    0.0543,    0.1116,    0.0543,    0.0063],
                  [0.0007,    0.0063,    0.0129,    0.0063,    0.0007]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        [b, c, h, w] = x.shape
        x1 = x.view(b*c, 1, h, w)
        x1 = F.conv2d(x1, self.weight, padding=2)
        x1 = x1.view(b, c, h, w)
        return x1

class GaussianBlur3(nn.Module):
    def __init__(self):
        super(GaussianBlur3, self).__init__()
        kernel = [[0.0117,    0.0862,    0.0117],
                  [0.0862,    0.6366,    0.0862],
                  [0.0117,    0.0862,    0.0117]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        [b, c, h, w] = x.shape
        x1 = x.view(b*c, 1, h, w)
        x1 = F.conv2d(x1, self.weight, padding=1)
        x1 = x1.view(b, c, h, w)
        return x1

class GaussianBlur7(nn.Module):
    def __init__(self):
        super(GaussianBlur7, self).__init__()
        kernel = [[0.0002,    0.0010,    0.0030,    0.0043,    0.0030,    0.0010,    0.0002],
                  [0.0010,    0.0062,    0.0186,    0.0269,    0.0186,    0.0062,    0.0010],
                  [0.0030,    0.0186,    0.0561,    0.0810,    0.0561,    0.0186,    0.0030],
                  [0.0043,    0.0269,    0.0810,    0.1169,    0.0810,    0.0269,    0.0043],
                  [0.0030,    0.0186,    0.0561,    0.0810,    0.0561,    0.0186,    0.0030],
                  [0.0010,    0.0062,    0.0186,    0.0269,    0.0186,    0.0062,    0.0010],
                  [0.0002,    0.0010,    0.0030,    0.0043,    0.0030,    0.0010,    0.0002]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        [b, c, h, w] = x.shape
        x1 = x.view(b*c, 1, h, w)
        x1 = F.conv2d(x1, self.weight, padding=3)
        x1 = x1.view(b, c, h, w)
        return x1

class vgg_v2(nn.Module):
    def __init__(self, vgg_model):
        super(vgg_v2, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

def normal_grad_loss(img, gt):
    [b, c, h, w] = img.shape

    img_log = torch.log10(img.detach()+0.5)
    gt_log = torch.log10(gt.detach()+0.5)
    factor = img_log/(gt_log)

    factor = torch.mean(torch.mean(torch.mean(factor, 1), 1), 1)
    
    gt_log_scale = gt_log.clone()
    for i in range(b):
        gt_log_scale[i, :, :, :] = gt_log_scale[i, :, :, :] * factor[i]

    gt_scale = torch.pow(10, gt_log_scale)-0.5
    gt_scale[gt_scale>1] = 1
    gt_scale[gt_scale<0] = 0

    return grad_loss(img, gt_scale.detach()) + ch_grad_loss(img, gt_scale.detach())

def grad_loss(img, gt):
    mse = nn.MSELoss(size_average=True)
    img_x = img[:,:,:,2:] - img[:,:,:,:-2]
    img_y = img[:,:,2:,:] - img[:,:,:-2,:]
    gt_x = gt[:,:,:,2:] - gt[:,:,:,:-2]
    gt_y = gt[:,:,2:,:] - gt[:,:,:-2,:]

    return mse(img_x, gt_x) + mse(img_y, gt_y)

def grad_th_loss(img, gt, th):
    mse = nn.MSELoss(size_average=True)
    img_x = img[:,:,:,2:] - img[:,:,:,:-2]
    img_y = img[:,:,2:,:] - img[:,:,:-2,:]
    gt_x = gt[:,:,:,2:] - gt[:,:,:,:-2]
    gt_y = gt[:,:,2:,:] - gt[:,:,:-2,:]

    gt_y[gt_y.abs()<th] = 0
    gt_x[gt_x.abs()<th] = 0

    return mse(img_x, gt_x) + mse(img_y, gt_y)

def ch_grad_loss(img, gt):
    mse = nn.MSELoss(size_average=True)
    img_ch1 = img[:,0:1,:,:] - img[:,1:2,:,:]
    gt_ch1 = gt[:,0:1,:,:] - gt[:,1:2,:,:]
    img_ch2 = img[:,1:2,:,:] - img[:,2:,:,:]
    gt_ch2 = gt[:,1:2,:,:] - gt[:,2:,:,:]
    img_ch3 = img[:,0:1,:,:] - img[:,2:,:,:]
    gt_ch3 = gt[:,0:1,:,:] - gt[:,2:,:,:]

    return mse(img_ch1, gt_ch1) + mse(img_ch2, gt_ch2) + mse(img_ch3, gt_ch3)

def gaussian_grad_loss(lr, img, gt):
    mse = nn.MSELoss(size_average=True)
    gaussian_op = GaussianBlur()

    [b, c, h, w] = img.shape
    img_max = torch.max(img, dim=1)
    img_max = img_max[0].view(b, 1, h, w)
    img_max = gaussian_op(img_max).view(b, 1, h, w)
    img_max = torch.cat((img_max, img_max, img_max), 1)
 
    img_ref = img/(img_max.detach()+0.001)

    lr_max = torch.max(lr, dim=1)
    lr_max = lr_max[0].view(b, 1, h, w)
    lr_max = gaussian_op(lr_max).view(b, 1, h, w)
    lr_max = torch.cat((lr_max, lr_max, lr_max), 1)

    lr_ref = lr/(lr_max.detach()+0.001)

    gt_max = torch.max(gt, dim=1)
    gt_max = gt_max[0].view(b, 1, h, w)
    gt_max = gaussian_op(gt_max).view(b, 1, h, w)
    gt_max = torch.cat((gt_max, gt_max, gt_max), 1)

    gt_ref = gt/(gt_max.detach()+0.001)

    return grad_loss(img_ref, gt_ref.detach()) + mse(img_ref, lr_ref.detach())

def vgg_loss(vgg, img, gt):
    mse = nn.MSELoss(size_average=True)
    img_vgg = vgg(img)
    gt_vgg = vgg(gt)

    #return 0.4*mse(img_vgg[2], gt_vgg[2]) + 0.2*mse(img_vgg[3], gt_vgg[3])
    return mse(img_vgg[0], gt_vgg[0]) + 0.6*mse(img_vgg[1], gt_vgg[1]) + 0.4*mse(img_vgg[2], gt_vgg[2]) + 0.2*mse(img_vgg[3], gt_vgg[3])

def vgg_init(vgg_loc):
    vgg_model = torchvision.models.vgg16(pretrained = False).cuda()
    vgg_model.load_state_dict(torch.load(vgg_loc))
    trainable(vgg_model, False)

    return vgg_model

def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()

        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        criterion_ssim = pytorch_ssim.SSIM(window_size = 11)
        criterion_mse = nn.MSELoss(size_average=True)

    #    vgg_model = vgg_init('./pretrained/vgg16-397923af.pth')
    #    vgg = vgg_v2(vgg_model)
    #    vgg.eval()

        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            lr = lr/255.0
            hr = hr/255.0
 
            [b, c, h, w] = hr.shape
 
            phr1, phr2, phr4 = self.model(lr, 3)

            hr4 = hr[:, :, 0::4, 0::4]
            hr2 = hr[:, :, 0::2, 0::2]
            hr1 = hr 

            rect_loss = criterion_ssim(phr1, hr1) +  criterion_ssim(phr2, hr2) + criterion_ssim(phr4, hr4)

            full_loss = rect_loss
 
            if full_loss.item() < self.args.skip_threshold * self.error_last:
                full_loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, rect_loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}=\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    full_loss.item(),
                    rect_loss.item(),
                    #percept_loss.item(),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        print(rect_loss.item())

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)

                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    lr = lr/255.0
                    hr = hr/255.0

                    [b, c, h, w] = hr.shape
                    n_map = torch.zeros(b, c, h, w).cuda()

                    phr1, phr2, phr4 = self.model(lr, 3)

                    phr = utility.quantize(phr1*255, self.args.rgb_range)
                    lr = utility.quantize(lr*255, self.args.rgb_range)
                    hr = utility.quantize(hr*255, self.args.rgb_range)

                    save_list = [hr, lr, phr, lr]

                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            phr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale, epoch)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

