import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
import itertools
from .base_model import BaseModel
from . import networks
from . import ssim

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class SketchModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_SSIM', type=float, default=100.0, help='weight for SSIM loss')
        parser.add_argument('--gan', type=str2bool, default=True, help='using gan or not')
        parser.add_argument('--adcnn', type=str2bool, default=True, help='using adaptive cnn or not')
        parser.add_argument('--splits', type=int, action='append', help='adaptive cnn region splits, \
                                    --splits 3 --splits 5 given a conbination of 9 and 25 regions', required=True)
        parser.add_argument('--FMN', type=int, action='append', help='adaptive cnn\'s FC kernels')
        parser.add_argument('--adc', type=int, default=32, help='number of down on unet')
        parser.add_argument('--shortcut_adaptive', type=str2bool, default=True, help='using shortcut_adaptive or not')
        parser.add_argument('--dfcnn', type=str2bool, default=True, help='use deformable cnn or not')
        parser.add_argument('--num_downs', type=int, default=7, help='number of down on unet')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = (['G_GAN', 'G_L1', 'D_real', 'D_fake'] if opt.gan else ['G_L1'] ) + (['G_ssim'] if opt.isTrain and opt.lambda_SSIM != 0 else [])
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.visual_names = ['fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D'] if opt.gan else ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = self.define_sktnet(opt, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain and opt.gan:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSSIM = ssim.NORMSSIM().to(self.device)
            # networks.init_net(self.criterionL1, gpu_ids=self.gpu_ids)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if opt.gan:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)



    def define_sktnet(self, opt, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
        norm_layer = networks.get_norm_layer(norm_type=norm)
        net = networks.find_model_using_name(opt.netG)(
                    input_nc=opt.input_nc,
                    output_nc=opt.output_nc,
                    num_downs=opt.num_downs,
                    adcnn=opt.adcnn,
                    dfcnn=opt.dfcnn,
                    splits=opt.splits,
                    adc=opt.adc,
                    FMN=opt.FMN,
                    shortcut=opt.shortcut_adaptive,
                    ngf=opt.ngf,
                    group_fc=opt.group_fc,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout)
        return networks.init_net(net, init_type, init_gain, gpu_ids)



    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_L1
        if self.opt.gan:
            """Calculate GAN and L1 loss for the generator"""
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G += self.loss_G_GAN
        if self.opt.lambda_SSIM != 0.0:
            self.loss_G_ssim = self.criterionSSIM(self.fake_B, self.real_B) * self.opt.lambda_SSIM 
            self.loss_G += self.loss_G_ssim
        # combine loss and calculate gradients
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.opt.gan:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

