import os
import numpy as np
import argparse


class Options:
    def __init__(self, isTrain):
        self.isTrain = isTrain  # train or test mode
        self.dataset = 'MO'     # LC: Lung Cancer, CC: Colon Cancer, MO: Multi-Organ
        self.ratio = 0.10        # ratio of retained points annotation in each image
        if self.dataset == 'LC':
            self.r1 = 8
            self.r2 = 16
            self.gaussian_sigma = 2
        elif self.dataset == 'MO':
            self.r1 = 11
            self.r2 = 22
            self.gaussian_sigma = 11.0/4
        else:
            raise ValueError('Wrong dataset')  # you can define your own dataset and change this code

        # --- model hyper-parameters --- #
        self.model = dict()
        self.model['name'] = 'ResUNet34'
        self.model['pretrained'] = True
        self.model['out_c'] = 1

        # --- training params --- #
        self.train = dict()
        self.train['random_seed'] = -1
        self.train['data_dir'] = '../data_for_train/{:s}'.format(self.dataset)  # path to data
        self.train['root_save_dir'] = '../experiments/detection/{:s}/{:.2f}'.format(self.dataset, self.ratio)  # path to save results
        self.train['input_size'] = 224      # input size of the image
        self.train['train_epochs'] = 80     # number of training iterations
        self.train['batch_size'] = 16       # batch size
        self.train['checkpoint_freq'] = 100  # epoch to save checkpoints
        self.train['lr'] = 0.0001           # initial learning rate
        self.train['weight_decay'] = 1e-4   # weight decay
        self.train['log_interval'] = 30     # iterations to print training results
        self.train['workers'] = 1           # number of workers to load images
        self.train['gpus'] = [0, ]           # select gpu devices

        # --- data transform --- #
        self.transform = dict()

        # --- test parameters --- #
        self.test = dict()
        self.test['epoch'] = 'best'
        self.test['gpus'] = [0, ]
        self.test['threshold'] = 0.35
        self.test['img_dir'] = '../data/{:s}/images'.format(self.dataset)
        self.test['label_dir'] = '../data/{:s}/labels_point'.format(self.dataset)
        self.test['save_flag'] = True
        self.test['patch_size'] = 224
        self.test['overlap'] = 80
        self.test['save_dir'] = '../experiments/detection/{:s}/test_results'.format(self.dataset)
        self.test['model_path'] = '../experiments/detection/{:s}/checkpoints/checkpoint_{:s}.pth.tar' \
            .format(self.dataset, self.test['epoch'])

        # --- post processing --- #
        self.post = dict()
        if self.dataset == 'LC':
            self.post['max_area'] = 150
        else:
            self.post['max_area'] = 200
        self.post['min_area'] = 12

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--batch-size', type=int, default=self.train['batch_size'], help='input batch size for training')
            parser.add_argument('--random-seed', type=int, default=self.train['random_seed'], help='random seed for training')
            parser.add_argument('--epochs', type=int, default=self.train['train_epochs'], help='number of epochs to train')
            parser.add_argument('--lr', type=float, default=self.train['lr'], help='learning rate')
            parser.add_argument('--log-interval', type=int, default=self.train['log_interval'], help='how many batches to wait before logging training status')
            parser.add_argument('--gpus', type=int, nargs='+', default=self.train['gpus'], help='GPUs for training')
            parser.add_argument('--data-dir', type=str, default=self.train['data_dir'], help='directory of training data')
            parser.add_argument('--root-save-dir', type=str, default=self.train['root_save_dir'], help='directory to save training results')
            args = parser.parse_args()

            self.train['batch_size'] = args.batch_size
            self.train['train_epochs'] = args.epochs
            self.train['lr'] = args.lr
            self.train['log_interval'] = args.log_interval
            self.train['gpus'] = args.gpus
            self.train['data_dir'] = args.data_dir
            self.train['img_dir'] = '{:s}/images'.format(self.train['data_dir'])
            self.train['label_dir'] = '{:s}/labels_detect'.format(self.train['data_dir'])
            self.train['bg_dir'] = '{:s}/labels_bg'.format(self.train['data_dir'])

            self.train['root_save_dir'] = args.root_save_dir
            if not os.path.exists(self.train['root_save_dir']):
                os.makedirs(self.train['root_save_dir'], exist_ok=True)
        else:
            parser.add_argument('--gpus', type=int, nargs='+', default=self.test['gpus'], help='GPUs for test')
            parser.add_argument('--img-dir', type=str, default=self.test['img_dir'], help='directory of test images')
            parser.add_argument('--label-dir', type=str, default=self.test['label_dir'], help='directory of labels')
            parser.add_argument('--save-dir', type=str, default=self.test['save_dir'], help='directory to save test results')
            parser.add_argument('--model-path', type=str, default=self.test['model_path'], help='train model to be evaluated')
            parser.add_argument('--threshold', type=float, default=self.test['threshold'], help='threshold to obtain the prediction from probability map')
            args = parser.parse_args()
            self.test['gpus'] = args.gpus
            self.test['img_dir'] = args.img_dir
            self.test['label_dir'] = args.label_dir
            self.test['save_dir'] = args.save_dir
            self.test['model_path'] = args.model_path
            self.test['threshold'] = args.threshold

            if not os.path.exists(self.test['save_dir']):
                os.makedirs(self.test['save_dir'], exist_ok=True)

    def define_transforms(self):
        # define data transforms for training
        self.transform['train_stage1'] = {  # for the training of stage 1
            'random_resize': [0.8, 1.25],
            'horizontal_flip': True,
            'vertical_flip': True,
            'random_affine': 0.3,
            'random_rotation': 90,
            'random_crop': self.train['input_size'],
            'label_gaussian': (-1, self.gaussian_sigma),
            'to_tensor': 2,
            'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
        }
        self.transform['train_stage2'] = {  # for the training of stage 2
            'random_resize': [0.8, 1.25],
            'horizontal_flip': True,
            'vertical_flip': True,
            'random_affine': 0.3,
            'random_rotation': 90,
            'random_crop': self.train['input_size'],
            'label_gaussian': (-2, self.gaussian_sigma),
            'label_binarization': (-1,),
            'to_tensor': 3,
            'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
        }
        self.transform['val'] = {
            'to_tensor': 2,
            'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
        }
        self.transform['test'] = {
            'to_tensor': 1,
            'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
        }

    def print_options(self, logger=None):
        message = '\n'
        message += self._generate_message_from_options()
        if not logger:
            print(message)
        else:
            logger.info(message)

    def save_options(self):
        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        message = self._generate_message_from_options()
        file = open(filename, 'w')
        file.write(message)
        file.close()

    def _generate_message_from_options(self):
        message = ''
        message += '# {str:s} Options {str:s} #\n'.format(str='-' * 25)
        train_groups = ['model', 'train', 'transform']
        test_groups = ['model', 'test', 'post', 'transform']
        cur_group = train_groups if self.isTrain else test_groups

        for group, options in self.__dict__.items():
            if group not in train_groups + test_groups:
                message += '{:>20}: {:<35}\n'.format(group, str(options))
            elif group in cur_group:
                message += '\n{:s} {:s} {:s}\n'.format('*' * 15, group, '*' * 15)
                if group == 'transform':
                    for name, val in options.items():
                        if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                            message += '{:s}:\n'.format(name)
                            for t_name, t_val in val.items():
                                t_val = str(t_val).replace('\n', ',\n{:22}'.format(''))
                                message += '{:>20}: {:<35}\n'.format(t_name, str(t_val))
                else:
                    for name, val in options.items():
                        message += '{:>20}: {:<35}\n'.format(name, str(val))
        message += '# {str:s} End {str:s} #\n'.format(str='-' * 26)
        return message