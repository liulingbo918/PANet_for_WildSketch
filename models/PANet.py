import torch
from torch import nn
import functools
from .acnn import AdapCNN_Block
from .dfcnn import DeformCNN_Block
from .racnn import Region_AdapCNN_Block

class PANet(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, adc=32, FMN=None,group_fc=2,ngf=64, adcnn=True, dfcnn=True, splits=[3,4,5], shortcut=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            dfcnn           -- whether to use deformable convolutional layers in the Face-adaptive perceiving decoder
            adcnn           -- whether to use adaptive convolution layers in the Component-adaptive perceiving module
            splits          -- the multi-scale setting of Component-adaptive perceiving module
            adc             -- the channle number of adaptive convolution layers
            FMN             -- the dimension of fully connected layers in adaptive convolution layer
            group_fc        -- the group number of fully connected layers in adaptive convolution layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(PANet, self).__init__()
        # construct unet structure
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 3, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 3, input_nc=None, submodule=unet_block, norm_layer=norm_layer, dfcnn=dfcnn, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 1, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, dfcnn=dfcnn, use_dropout=use_dropout)

        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, adcnn=adcnn, dfcnn=dfcnn, adc=adc, FMN=FMN,group_fc=group_fc,n_splits=splits, shortcut=shortcut)  # add the outermost layer

        print(self.model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, **kargs):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Sequential(
                            nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                                    stride=1, padding=1, bias=use_bias),
                            norm_layer(inner_nc),
                            nn.LeakyReLU(0.2, True),
                            nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
                                    stride=1, padding=1, bias=use_bias),
                            )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        downpool = nn.MaxPool2d(2, stride=2)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)


        collect_types = [DeformCNN_Block, SequentialCollector, UnetSkipConnectionBlock, Region_AdapCNN_Block]


        if outermost:
            if kargs['adcnn'] or kargs['dfcnn']:
                upconv = nn.Sequential(
                                    nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, padding=1,bias=use_bias)
                                    )
            else:
                upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=1, padding=0)
            refine = []
            final_nc = -1
            if kargs['dfcnn']:
                final_nc = inner_nc
                refinement = SequentialCollector(
                        collect_types,
                        norm_layer(inner_nc),
                        nn.ReLU(True),
                        DeformCNN_Block(in_nc=inner_nc, out_nc=inner_nc, kers=3, offset_kers=3, padding=1, bias=use_bias),
                )
                refine += [refinement]
            if kargs['adcnn']:
                final_nc = kargs['adc'] * len(kargs['n_splits']) + (inner_nc if kargs['shortcut'] else 0)
                fmn_nc = [2048, 2048] if kargs['FMN'] is None or len(kargs['FMN']) != 2 else kargs['FMN']
				#group = kargs['group_fc'] # The groups of the fully connected layers   
                region_adpcnn = SequentialCollector(
                                    collect_types,
                                    norm_layer(inner_nc),
                                    nn.ReLU(True),
                                    Region_AdapCNN_Block(input_nc=inner_nc, adc_kers=3, n_splits=kargs['n_splits'], short_cut=kargs['shortcut'], \
                                    reduce_fc=kargs['adc'], reduce_adc=kargs['adc'], fmn_nc=fmn_nc, group_fc=kargs['group_fc'],bias=use_bias),
                                )
                refine += [region_adpcnn]
            else:
                refine += [nn.Sequential(
                            norm_layer(inner_nc),
                            nn.ReLU(True),
                            nn.Conv2d(inner_nc, 160, kernel_size=3, padding=1,bias=use_bias)
                    )]
                final_nc = 160

            if final_nc != -1:
                refine += [norm_layer(final_nc), nn.ReLU(True), nn.Conv2d(final_nc, outer_nc, kernel_size=1, padding=0)]
           
            up = [uprelu, upconv] + refine + [nn.Tanh()]
            down = [downconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downpool, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            if kargs['dfcnn']:
                refinement = SequentialCollector(
                    collect_types,
                    DeformCNN_Block(in_nc=inner_nc * 2, out_nc=inner_nc * 2, kers=3, offset_kers=3, padding=1, bias=use_bias),
                    norm_layer(inner_nc),
                    nn.ReLU(True),
                )
            else:
                refinement = nn.Sequential(
                    nn.Conv2d(inner_nc * 2, inner_nc * 2, kernel_size=3,
                            stride=1, padding=1, bias=use_bias),
                    norm_layer(inner_nc * 2),
                    nn.ReLU(True),
                )

            upconv = SequentialCollector(
                    collect_types,
                    refinement,
                    nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            )

            down = [downrelu, downpool, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = SequentialCollector(
                    collect_types,*model)

    def forward(self, x):
        if self.training:
            if self.outermost:
                return self.model(x)
            else:   # add skip connections
                x = torch.cat([x, self.model(x)], 1)
                return x
        else:
            if self.outermost:
                result, collections = self.model(x)
                #return result, collections
                return result
            else:   # add skip connections
                result, collections = self.model(x)
                x = torch.cat([x, result], 1)
                return x, collections

class SequentialCollector(nn.Module):

    def __init__(self, collect_types, *layers):
        super(SequentialCollector, self).__init__()
        self.collect_types = collect_types
        for idx, module in enumerate(layers):
            self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, tensor_in):
        collector = []
        for key, module in self._modules.items():
            # print(type(module))
            if (not self.training) and \
                type(module) in self.collect_types:
                tensor_in, collection = module(tensor_in)
                collector += [*collection]
            else:
                tensor_in = module(tensor_in)
                
        if self.training:
            return tensor_in
        else:        
            return tensor_in, collector
