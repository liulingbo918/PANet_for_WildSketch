import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from .networks import compute_same_padding2d, _pair, _triple

class AdapCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding=1):
        super(AdapCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.bias = bias

    def forward(self, input, weight, bias, dilation=None):
        # assert weight.shape == (self.out_channels, self.in_channels/self.groups, self.kernel_size, self.kernel_size)
        # assert bias is None or bias.shape == (self.out_channels)
        input_shape = list(input.size())
        # print(input_shape)
        dilation_rate = self.dilation if dilation is None else dilation
        # print dilation_rate
        # padding, pad_input = compute_same_padding2d(input_shape, kernel_size=_pair(self.kernel_size), strides=_pair(self.stride), dilation=_pair(dilation_rate))
        # if pad_input[0] == 1 or pad_input[1] == 1:
        #     input = F.pad(input, [0, int(pad_input[0]), 0, int(pad_input[1])], mode='replicate')
        padding = (0,0)
        batch_size, C, H, W = input_shape
        # weight.shape == (self.out_channels, self.in_channels/self.groups, 1, self.kernel_size, self.kernel_size)
        # weight.shape == (adc_out * n_splits, adc_in * n_splits / n_splits, 1, self.kernel_size, self.kernel_size)
        return F.conv3d(input.view(1, batch_size * C, 1, H, W), weight, bias, _triple(self.stride),
                       (0, padding[0] // 2, padding[1] // 2), _triple(dilation_rate), groups=batch_size)
        #https://github.com/pytorch/pytorch/issues/3867

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != self.padding:
            s += ', padding={padding}'
        if self.dilation != self.dilation:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class AdapCNN_Block(nn.Module):
    def __init__(self, adc_in_nc, adc_out_nc, adc_kers, fc_size, fmn_nc, group_fc, bias=True):
        super(AdapCNN_Block, self).__init__()
        self.adc_in_nc = adc_in_nc
        self.adc_out_nc = adc_out_nc
        self.adc_kers = adc_kers
        self.fc_size = fc_size
        self.fmn_nc = fmn_nc
        self.group_fc = group_fc
        self.bias = bias

        if bias:
            self.cnn_para_size = self.adc_in_nc *  self.adc_out_nc *  self.adc_kers * self.adc_kers + self.adc_out_nc
        else:
            self.cnn_para_size = self.adc_in_nc *  self.adc_out_nc *  self.adc_kers * self.adc_kers

        print("group fc", group_fc)
	###Architecture of the group fully connected layers
        self.FMN  = nn.Sequential(
                #nn.Linear(self.fc_size, self.fmn_nc[0], bias=True), 
                nn.Conv1d(self.fc_size,self.fmn_nc[0],kernel_size=1,bias=True),
                nn.ReLU(True),
                #nn.Conv1d(self.fmn_nc[0],self.fmn_nc[1],kernel_size=1,groups=self.group_fc,bias=True),
                nn.Conv1d(self.fmn_nc[0],self.fmn_nc[1],kernel_size=1,bias=True),
                nn.ReLU(True),
                nn.Conv1d(self.fmn_nc[1],self.cnn_para_size,kernel_size=1,groups=self.group_fc, bias=True),
                nn.ReLU(True),
        )
        self.acnn1 = AdapCNN(self.adc_in_nc, self.adc_out_nc, self.adc_kers, bias=bias)

    def forward(self, x, fc_in, splits, return_weight=False):
        # fc_in: [n_splits, sh * sw]
        fc_in = fc_in.view(fc_in.shape[0],fc_in.shape[1],1)
        weight_bias = self.FMN(fc_in)
        #print(weight_bias.shape)
		# weight_bias: [n_splits, weightsize]
        if self.bias:
            weight = weight_bias[:, :- self.adc_out_nc].contiguous().view(splits, -1, self.adc_in_nc,1, self.adc_kers, self.adc_kers)
            bias = weight_bias[:,- self.adc_out_nc:].contiguous().view(-1)
        else:
            weight = weight_bias.contiguous().view(splits, -1, self.adc_in_nc,1, self.adc_kers, self.adc_kers)
            bias = None

        #print(weight.shape)
        #print(bias.shape)

        weight = weight.view(-1, self.adc_in_nc, 1, self.adc_kers, self.adc_kers)

        y = self.acnn1(x, weight, bias)
        if not return_weight:
            return y
        else:
            return y, weight
