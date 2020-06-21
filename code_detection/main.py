import os, json
import torch
import numpy as np
import random
from skimage import morphology, measure, io, util

from options import Options
from prepare_data import main as prepare_data
from train import main as train
from test import main as test


def main():
    opt = Options(isTrain=True)
    opt.parse()

    if opt.train['random_seed'] >= 0:
        print('=> Using random seed {:d}'.format(opt.train['random_seed']))
        torch.manual_seed(opt.train['random_seed'])
        torch.cuda.manual_seed(opt.train['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.train['random_seed'])
        random.seed(opt.train['random_seed'])
    else:
        torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    with open('../data/{:s}/train_val_test.json'.format(opt.dataset), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    num_repeat = 3
    opt.test['epoch'] = 'best'
    opt.test['threshold'] = 0.35
    opt.test['label_dir'] = '../data/{:s}/labels_point'.format(opt.dataset)

    # initial training
    opt.print_options()
    print('=============== initial training ===============')
    opt.round = 0
    print('=> Preparing training samples')
    prepare_data(opt)
    opt.train['save_dir'] = '{:s}/{:d}'.format(opt.train['root_save_dir'], opt.round)
    print('=> Start training')
    train(opt)

    # test model
    print('=> Testing ...')
    opt.test['img_dir'] = '../data_for_train/{:s}/images/test'.format(opt.dataset)
    opt.test['save_dir'] = '{:s}/0/{:s}'.format(opt.train['root_save_dir'], opt.test['epoch'])
    opt.test['model_path'] = '{:s}/0/checkpoints/checkpoint_{:s}.pth.tar' \
        .format(opt.train['root_save_dir'],  opt.test['epoch'])
    test(opt)

    print('=> Inference ...')
    opt.test['img_dir'] = '../data/{:s}/images'.format(opt.dataset)
    test(opt)

    for i in range(1, num_repeat+1):
        print('=============== self training round {:d} ==============='.format(i))
        opt.round = i

        # ----- prepare training data ----- #
        probmap_dir = '{:s}/{:d}/{:s}/images_prob_maps'.format(opt.train['root_save_dir'], i-1, opt.test['epoch'])
        label_bg_dir = '../data/{:s}/labels_bg_{:.2f}_round{:d}'.format(opt.dataset, opt.ratio, i)
        get_bg_from_prob_maps(probmap_dir, label_bg_dir, train_list, opt)
        print('=> Preparing training samples')
        prepare_data(opt)

        # ----- train model ----- #
        print('=> Start training')
        opt.train['save_dir'] = '{:s}/{:d}'.format(opt.train['root_save_dir'], i)
        train(opt)

        # test model
        print('=> Testing ...')
        opt.test['img_dir'] = '../data_for_train/{:s}/images/test'.format(opt.dataset)
        opt.test['save_dir'] = '{:s}/{:d}/{:s}'.format(opt.train['root_save_dir'], i, opt.test['epoch'])
        opt.test['model_path'] = '{:s}/{:d}/checkpoints/checkpoint_{:s}.pth.tar' \
            .format(opt.train['root_save_dir'], i, opt.test['epoch'])
        test(opt)

        print('=> Inference ...')
        opt.test['img_dir'] = '../data/{:s}/images'.format(opt.dataset)
        test(opt)


def get_bg_from_prob_maps(probmap_dir, bg_dir, img_list, opt):
    if not os.path.exists(bg_dir):
        os.mkdir(bg_dir)

    for img_name in sorted(img_list):
        name = img_name.split('.')[0]

        prob_map = io.imread('{:s}/{:s}_prob.png'.format(probmap_dir, name))
        prob_map = util.img_as_float(prob_map)

        # remove large white bg areas
        bg_area = prob_map > 0.7
        bg_area_labeled = measure.label(bg_area)
        bg_area = morphology.remove_small_objects(bg_area_labeled, opt.post['max_area']) > 0
        prob_map2 = prob_map * (bg_area == 0)
        bg = prob_map2 < 0.1

        label_bg = np.zeros((prob_map.shape[0], prob_map.shape[1]), dtype=np.uint8)
        label_bg[bg > 0] = 255

        io.imsave('{:s}/{:s}_label_bg.png'.format(bg_dir, name), label_bg)


if __name__ == '__main__':
    main()
