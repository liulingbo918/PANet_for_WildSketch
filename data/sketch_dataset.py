import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
from functools import cmp_to_key

import re
datasets = {

    'WildSketch': {
        "train_image_path": 'dataset/WildSketch/train/images',
        "train_label_path": 'dataset/WildSketch/train/sketches',
        "test_image_path": 'dataset/WildSketch/test/images',
        "test_label_path": 'dataset/WildSketch/test/sketches',
        "train_val_split": (lambda x:x, lambda x:x[:29]),
        'image_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        'label_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        'test_image_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        'test_label_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        # "mean_std": [157.5362469066273, 63.201925767999185],
    },
    'CUHK': {
        "train_image_path": 'dataset/CUHK/train/images',
        "train_label_path": 'dataset/CUHK/train/ground_truth',
        "test_image_path": 'dataset/CUHK/test_rename/images',
        "test_label_path": 'dataset/CUHK/test_rename/ground_truth',
        "train_val_split": (lambda x:x, lambda x:x[:29]),
        'image_id': lambda x: x[0].upper() + '_'.join(re.findall(r'\d+',x)[0:2]),
        'label_id': lambda x: x[0].upper() + '_'.join(re.findall(r'\d+',x)[1:3]),
        'test_image_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        'test_label_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        # "mean_std": [157.5362469066273, 63.201925767999185],
    },
    'AR': {
        "train_image_path": 'dataset/AR/train/images',
        "train_label_path": 'dataset/AR/train/sketches',
        "test_image_path": 'dataset/AR/test_rename/images',
        "test_label_path": 'dataset/AR/test_rename/sketches',
        "train_val_split": (lambda x:x, lambda x:x[:29]),
        'image_id': lambda x: x[0].upper() + '_'.join(re.findall(r'\d+',x)[0:1]),
        'label_id': lambda x: x[0].upper() + '_'.join(re.findall(r'\d+',x)[0:1]),
        'test_image_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        'test_label_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        # "mean_std": [157.5362469066273, 63.201925767999185],
    },
    'XM2VTS': {
        "train_image_path": 'dataset/XM2VTS/train/images',
        "train_label_path": 'dataset/XM2VTS/train/sketches',
        "test_image_path": 'dataset/XM2VTS/test_rename/images',
        "test_label_path": 'dataset/XM2VTS/test_rename/sketches',
        "train_val_split": (lambda x:x, lambda x:x[:29]),
        'image_id': lambda x: '_'.join(re.findall(r'\d+',x)[0:1]),
        'label_id': lambda x: '_'.join(re.findall(r'\d+',x)[0:1]),
        'test_image_id': lambda x: '_'.join(re.findall(r'\d+',x)[0:1]),
        'test_label_id': lambda x: '_'.join(re.findall(r'\d+',x)[0:1]),
        # "mean_std": [157.5362469066273, 63.201925767999185],
    },
    'CUFSF': {
        "train_image_path": 'dataset/CUFSF/train/images',
        "train_label_path": 'dataset/CUFSF/train/sketches',
        "test_image_path": 'dataset/CUFSF/test_rename/images',
        "test_label_path": 'dataset/CUFSF/test_rename/sketches',
        "train_val_split": (lambda x:x, lambda x:x[:29]),
        'image_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        'label_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        'test_image_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        'test_label_id': lambda x: '_'.join(re.findall(r'\d+',x)),
        # "mean_std": [157.5362469066273, 63.201925767999185],
    },
}

def cmp(a, b):
    return (a > b) - (a < b)

class SketchDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--dataset',  type=str, default='CUHK', help='dataset name')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        if opt.phase == 'train':
            self.image_path, self.label_path = datasets[opt.dataset]['train_image_path'], datasets[opt.dataset]['train_label_path']
            image_id_func, label_id_func = datasets[opt.dataset]['image_id'] ,datasets[opt.dataset]['label_id']
        else:
            self.image_path, self.label_path = datasets[opt.dataset]['test_image_path'], datasets[opt.dataset]['test_label_path']
            image_id_func, label_id_func = datasets[opt.dataset]['test_image_id'] ,datasets[opt.dataset]['test_label_id']

        self.image_files = [filename for filename in os.listdir(self.image_path) \
                           if os.path.isfile(os.path.join(self.image_path,filename))]
        self.label_files = [filename for filename in os.listdir(self.label_path) \
                           if os.path.isfile(os.path.join(self.label_path,filename))]

        self.image_files.sort(key=cmp_to_key(lambda x, y: cmp(image_id_func(x), image_id_func(y))))
        self.label_files.sort(key=cmp_to_key(lambda x, y: cmp(label_id_func(x), label_id_func(y))))

        for img, lab in zip(self.image_files, self.label_files):
            assert image_id_func(img) == label_id_func(lab), \
                    image_id_func(img) + '~' + label_id_func(lab)


        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = os.path.join(self.image_path, self.image_files[index])
        B_path = os.path.join(self.label_path, self.label_files[index])
        # print(self.image_files[index], self.label_files[index])

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_files)
