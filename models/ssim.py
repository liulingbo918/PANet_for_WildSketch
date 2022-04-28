import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import logging
from math import exp
from .networks import compute_same_padding2d


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # _2D_window = 1.0 * torch.ones((window_size, window_size))
    # _2D_window = _2D_window / _2D_window.sum()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window #/ window.sum()


def t_ssim(img1, img2, img11, img22, img12, window, channel, dilation=1, size_average=True, full=False):
    window_size = window.size()[2]
    input_shape = list(img1.size())

    padding, pad_input = compute_same_padding2d(input_shape, \
                                                kernel_size=(window_size, window_size), \
                                                strides=(1,1), \
                                                dilation=(dilation, dilation))
    if img11 is None:
        img11 = img1 * img1
    if img22 is None:
        img22 = img2 * img2
    if img12 is None:
        img12 = img1 * img2

    if pad_input[0] == 1 or pad_input[1] == 1:
        img1 = F.pad(img1, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img2 = F.pad(img2, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img11 = F.pad(img11, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img22 = F.pad(img22, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img12 = F.pad(img12, [0, int(pad_input[0]), 0, int(pad_input[1])])

    padd = (padding[0] // 2, padding[1] // 2)

    mu1 = F.conv2d(img1, window , padding=padd, dilation=dilation, groups=channel)
    mu2 = F.conv2d(img2, window , padding=padd, dilation=dilation, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    si11 = F.conv2d(img11, window, padding=padd, dilation=dilation, groups=channel)
    si22 = F.conv2d(img22, window, padding=padd, dilation=dilation, groups=channel)
    si12 = F.conv2d(img12, window, padding=padd, dilation=dilation, groups=channel)

    sigma1_sq = si11 - mu1_sq
    sigma2_sq = si22 - mu2_sq
    sigma12 = si12 - mu1_mu2

    C1 = (0.01*255)**2
    C2 = (0.03*255)**2
    # C1 = (0.001*255)**2
    # C2 = (0.003*255)**2
    # C1 = (0.1*255)**2
    # C2 = (0.3*255)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))


    # print("mu1_sq: {} mu2_sq: {} mu1_mu2: {} sigma1_sq: {} sigma2_sq: {} sigma12: {}\n C1: {} C2: {} SSIM: {}"\
    #     .format(float(mu1_sq.mean().data[0]),float(mu2_sq.mean().data[0]),float(mu1_mu2.mean().data[0]),
    #         float(sigma1_sq.mean().data[0]),float(sigma2_sq.mean().data[0]),float(sigma12.mean().data[0]),
    #         # float(C1.data[0]), float(C2.data[0]), float(ssim_map.mean().data[0])))
    #         float(C1), float(C2), float(ssim_map.mean().data[0])))

    # raw_input("test")
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    # cs = torch.mean(v1 / v2 + 1e-6)
    cs = torch.mean(v1 / v2)


    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs, mu1, mu2, si11, si22, si12

    # print("{}-{}-{}".format(float(ret.data[0]), float(l1.data[0]), float(result.data[0])))

    return ret


class NORMSSIM(torch.nn.Module):

    def __init__(self, sigma=1.5, size_average=True, channel=1):
        super(NORMSSIM, self).__init__()
        self.sigma = sigma
        self.window_size = 11
        # print(self.window_sizes)
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', create_window(self.window_size, self.channel, self.sigma))
        # self.mseLoss = nn.L1Loss()

    def forward(self, img1, img2):
        # loss = self.mseLoss(img1, img2)
        img1 = (img1 + 1.0) / 2.0 * 255.0
        img2 = (img2 + 1.0) / 2.0 * 255.0
        ssim_loss = (1 - self.ssim(img1, img2))
        # print "Sssss"
        # print float(loss), float(ssim_loss)
        return ssim_loss# + loss
        # return 0.1 * ssim_loss
        # return 0.1 * ssim_loss + loss
        # return (1 - self.ssim(img1, img2))

    def ssim(self, img1, img2):
        img1, img2, img11, img22, img12 = img1, img2, None, None, None
        return t_ssim(img1, img2, img11, img22, img12, \
                            Variable(getattr(self, "window"), requires_grad=False),\
                            self.channel, size_average=self.size_average, dilation=1, full=False)



