
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
from PIL import Image
import numpy as np
import logging
from tensorboardX import SummaryWriter
from skimage import measure, io
import skimage.morphology as ski_morph

from model import create_model
import utils
from dataset import DataFolder
from my_transforms import get_transforms


def main(opt):
    global best_score, num_iter, tb_writer, logger, logger_results
    best_score = 0
    opt.isTrain = True

    if not os.path.exists(opt.train['save_dir']):
        os.makedirs(opt.train['save_dir'])
    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    opt.define_transforms()
    opt.save_options()

    # set up logger
    logger, logger_results = setup_logging(opt)

    # ----- create model ----- #
    model_name = opt.model['name']
    model = create_model(model_name, opt.model['out_c'], opt.model['pretrained'])
    # if not opt.train['checkpoint']:
    #     logger.info(model)
    model = nn.DataParallel(model)
    model = model.cuda()

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    # ----- define criterion ----- #
    criterion = torch.nn.MSELoss(reduction='none').cuda()

    # ----- load data ----- #
    img_dir = '{:s}/train'.format(opt.train['img_dir'])
    target_dir = '{:s}/train'.format(opt.train['label_dir'])
    if opt.round == 0:
        dir_list = [img_dir, target_dir]
        post_fix = ['label_detect.png']
        num_channels = [3, 1]
        train_transform = get_transforms(opt.transform['train_stage1'])
    else:
        bg_dir = '{:s}/train'.format(opt.train['bg_dir'])
        dir_list = [img_dir, target_dir, bg_dir]
        post_fix = ['label_detect.png', 'label_bg.png']
        num_channels = [3, 1, 1]
        train_transform = get_transforms(opt.transform['train_stage2'])
    train_set = DataFolder(dir_list, post_fix, num_channels, train_transform)
    train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'])
    val_transform = get_transforms(opt.transform['val'])

    # ----- training and validation ----- #
    num_epoch = opt.train['train_epochs']
    num_iter = num_epoch * len(train_loader)
    # print training parameters
    logger.info("=> Initial learning rate: {:g}".format(opt.train['lr']))
    logger.info("=> Batch size: {:d}".format(opt.train['batch_size']))
    logger.info("=> Number of training iterations: {:d}".format(num_iter))
    logger.info("=> Training epochs: {:d}".format(opt.train['train_epochs']))

    for epoch in range(num_epoch):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, num_epoch))
        train_loss = train(opt, train_loader, model, optimizer, criterion)

        # evaluate on val set
        with torch.no_grad():
            val_recall, val_prec, val_F1 = validate(opt, model, val_transform)

        # check if it is the best accuracy
        is_best = val_F1 > best_score
        best_score = max(val_F1, best_score)

        cp_flag = True if (epoch + 1) % opt.train['checkpoint_freq'] == 0 else False
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, is_best, opt.train['save_dir'], cp_flag)

        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f} || {:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch + 1, train_loss, val_recall, val_prec, val_F1))
        # tensorboard logs
        tb_writer.add_scalars('epoch_loss', {'train_loss': train_loss}, epoch)
        tb_writer.add_scalars('epoch_acc', {'val_recall': val_recall, 'val_prec': val_prec, 'val_F1': val_F1}, epoch)

    tb_writer.close()
    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(opt, train_loader, model, optimizer, criterion):
    # list to store the average loss for this epoch
    results = utils.AverageMeter(1)

    # switch to train mode
    model.train()

    for i, sample in enumerate(train_loader):
        if opt.round == 0:
            input, target = sample
            target = target.squeeze(1)
            input, target = input.cuda(), target.cuda()
        else:
            input, target, bg = sample
            target = target.squeeze(1)
            bg = bg.squeeze(1)
            input, target, bg = input.cuda(), target.cuda(), bg.cuda()

        # compute output
        output = model(input).squeeze(1)
        probmaps = torch.sigmoid(output)

        mask = torch.zeros_like(target).float()
        for k in range(target.size(0)):
            mask_k = ski_morph.dilation(target[k].cpu().numpy()==1, selem=ski_morph.disk(opt.r2))
            mask[k] = torch.Tensor(mask_k.astype(np.float))

        # update background
        if opt.round > 0:
            mask = (mask + bg) > 0
            mask = mask.float()
        weight_map = mask.float().clone()
        weight_map[target > 0] = 10

        # compute loss
        loss_map = criterion(probmaps, target)
        loss = torch.sum(loss_map * weight_map) / mask.sum()

        result = [loss.item(),]
        results.update(result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del input, output, loss

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]\tLoss {r[0]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'.format(r=results.avg))

    return results.avg[0]


def validate(opt, model, data_transform):
    total_TP = 0.0
    total_FP = 0.0
    total_FN = 0.0

    # switch to evaluate mode
    model.eval()

    img_dir = '{:s}/images/val'.format(opt.train['data_dir'])
    label_dir = opt.test['label_dir']

    img_names = os.listdir(img_dir)
    for img_name in img_names:
        # load test image
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)
        name = os.path.splitext(img_name)[0]

        label_path = '{:s}/{:s}_label_point.png'.format(label_dir, name)
        gt = io.imread(label_path)

        input, label = data_transform((img, Image.fromarray(gt)))
        input = input.unsqueeze(0)

        prob_map = get_probmaps(input, model, opt)
        prob_map = prob_map.cpu().numpy()
        pred = prob_map > opt.test['threshold']  # prediction
        pred_labeled, N = measure.label(pred, return_num=True)
        if N > 1:
            bg_area = ski_morph.remove_small_objects(pred_labeled, opt.post['max_area']) > 0
            large_area = ski_morph.remove_small_objects(pred_labeled, opt.post['min_area']) > 0
            pred = pred * (bg_area==0) * (large_area>0)

        TP, FP, FN = utils.compute_accuracy(pred, gt, radius=opt.r1)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    recall = float(total_TP) / (total_TP + total_FN + 1e-8)
    precision = float(total_TP) / (total_TP + total_FP + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    logger.info('\t=> Val Avg:\tRecall {:.4f}\tPrec {:.4f}\tF1 {:.4f}'.format(recall, precision, F1))

    return recall, precision, F1


def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if size == 0:
        with torch.no_grad():
            output = model(input.cuda())
    else:
        output = utils.split_forward(model, input, size, overlap)
    output = output.squeeze(0)
    prob_maps = torch.sigmoid(output[0,:,:])

    return prob_maps


def save_checkpoint(state, epoch, is_best, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))


def setup_logging(opt):
    mode = 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\tval_recall\tval_prec\tval_F1')

    return logger, logger_results


if __name__ == '__main__':
    main()
