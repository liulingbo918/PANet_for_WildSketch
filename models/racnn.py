import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from itertools import product
from .networks import compute_same_padding2d, _pair, _triple, ResnetGenerator
from .acnn import AdapCNN_Block
from .dfcnn import DeformCNN_Block
import math

class Region_AdapCNN_Block(nn.Module):

    def __init__(self, input_nc, adc_kers=3, n_splits=[4], short_cut=True, \
                                reduce_fc=32, reduce_adc=32, fmn_nc=[2048, 2048],group_fc=2,bias=True, norm_layer=nn.BatchNorm2d):
        super(Region_AdapCNN_Block, self).__init__()

        self.n_splits = n_splits
        self.short_cut = short_cut

        self.acnn = AdapCNN_Block(adc_in_nc=input_nc, adc_out_nc=reduce_adc, adc_kers=3, fc_size=reduce_fc*reduce_fc, fmn_nc=fmn_nc,group_fc=group_fc,bias=bias)
        # self.activation = nn.LeakyReLU(0.2, False)
        self.activation = nn.ReLU(True)

        self.adc_in_nc = input_nc
        self.adc_out_nc = reduce_adc
        self.adc_kers = adc_kers
        self.weight_shuffle = False

        self.reducer = nn.Sequential(
                nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1),
                ###nn.Conv2d(input_nc, math.ceil(input_nc/2), kernel_size=3, stride=1, padding=1),
                # norm_layer(input_nc),
                nn.LeakyReLU(0.2, True),
                # nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1),
                # norm_layer(input_nc),
                # nn.LeakyReLU(0.2, True),
                nn.Conv2d(input_nc, 1, kernel_size=3, stride=1, padding=1),
                ###nn.Conv2d(math.ceil(input_nc/2), 1, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )
              
        self.context = nn.AdaptiveAvgPool2d((reduce_fc,reduce_fc))

    def forward(self, feature):

        batch_size, C, H, W = feature.shape

        ## Reduce the dim of feature from input_nc to 1,which is used for producing the weights of the net
        adaptive = self.reducer(feature)
        n_splits = self.n_splits

        if self.short_cut:
            split_ys = [feature]
        else:
            split_ys = []

        if not self.training:
            collector = []


        grams = []
        slices = []
        ### Divide the input of the input of the adaptive network to n_split*n_split grids
        for n_split in n_splits:
            adap = self.grid_slice(adaptive, n_split=n_split)
            ab, at, ac, ah, aw = adap.shape

            gram = self.context(adap.view(ab * at, ac, ah, aw))

            grams += [gram.view(ab, at, -1)]
            # grams += [gram]
            slices += [n_split ** 2]

        # for n_split, gram in zip(n_splits, grams_new if self.weight_shuffle else grams):
        for n_split, gram in zip(n_splits, grams):

            adap = self.grid_slice(adaptive, n_split=n_split)
            ab, at, ac, ah, aw = adap.shape
	    ### Divide the feature to n_split*n_split grids 
            fea = self.grid_slice(feature, n_split=n_split)
            fb, ft, fc, fh, fw = fea.shape

            if self.training:
                fea = self.acnn(fea.view(-1, fc, fh, fw), gram.view(ab*at, -1), n_split**2)
            else:
                fea, weight = self.acnn(fea.view(-1, fc, fh, fw), gram.view(ab*at, -1), n_split**2, return_weight=True)
                # collector += [weight.view(ft, self.adc_in_nc, self.adc_out_nc, self.adc_kers, self.adc_kers)]

            # fea = self.acnn_act(fea)
            # fea = getattr(self, 'norm_layer_%d' % n_split)(fea.view(fb, -1, fh, fw))
            fea = self.grid_splice(fea.view(fb, ft, -1, fh - 2, fw - 2))
            fea = fea.view(batch_size, -1, H, W)
            # fea = getattr(self, 'norm_layer_%d' % n_split)(fea)
            # split_ys += [self.activation(fea)]
            split_ys += [fea]

        if self.training:
            y = torch.cat(split_ys, dim=1)
            return y
        else:
            y = torch.cat(split_ys, dim=1)
            # weight = torch.cat(collector, dim=0)
            return y, [self.activation(y)]
        # y = self.estimater(y)


    def grid_slice(self, x, n_split, padding=1):

        xb, xc, xh, xw = x.shape

        xh_slice = int(xh / n_split) + int(xh % n_split != 0)
        self.xh_padding = xh_slice * n_split - xh
        xw_slice = int(xw / n_split) + int(xw % n_split != 0)
        self.xw_padding = xw_slice * n_split - xw
        x = nn.functional.pad(x, (int(self.xw_padding / 2), int(self.xw_padding) - int(self.xw_padding / 2),
                                  int(self.xh_padding / 2), int(self.xh_padding) - int(self.xh_padding / 2)), 'constant', 0)
        
        x = nn.functional.pad(x, (padding, padding,
                                  padding, padding), 'constant', 0)

        slices = []
        for hh, ww in product(range(n_split), repeat=2):
            patch = x[:, :,hh * xh_slice - 1 + 1:(hh + 1) * xh_slice + 1 + 1,
                      ww * xw_slice - 1 + 1:(ww + 1) * xw_slice + 1 + 1].contiguous()
            slices += [patch.view(-1, 1, xc, xh_slice + 2, xw_slice + 2)]
        slices = torch.cat(slices, dim=1)
        return slices

    def grid_splice(self, x):
        xb, n_split, xc, xh_slice, xw_slice = x.shape
        n_split = int(n_split ** 0.5)
        # xp = Variable(x.data.new(xb, xc, xh_slice*n_split, xw_slice*n_split))
        xp = x.new(xb, xc, xh_slice * n_split, xw_slice * n_split)

        for i, (hh, ww) in enumerate(product(range(n_split), repeat=2)):
            xp[:, :, hh * xh_slice:(hh + 1) * xh_slice, ww *
               xw_slice:(ww + 1) * xw_slice] = x[:, i, ...]

        x = xp[..., int(self.xh_padding / 2):xh_slice * n_split - int(self.xh_padding) + int(self.xh_padding / 2),
               int(self.xw_padding / 2):xw_slice * n_split - int(self.xw_padding) + int(self.xw_padding / 2)]

        return x.contiguous()
