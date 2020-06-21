"""
This script is used to prepare the dataset for training and testing.

"""

import os
import shutil
import numpy as np
from skimage import measure, io
from scipy.ndimage import gaussian_filter
import glob
import json


def main(opt):
    dataset = opt.dataset
    data_dir = '../data/{:s}'.format(dataset)
    img_dir = '../data/{:s}/images'.format(dataset)
    label_point_dir = '../data/{:s}/labels_point_{:.2f}'.format(dataset, opt.ratio)
    label_detect_dir = '../data/{:s}/labels_detect_{:.2f}'.format(dataset, opt.ratio)
    label_bg_dir = '../data/{:s}/labels_bg_{:.2f}_round{:d}'.format(dataset, opt.ratio, opt.round)
    train_data_dir = '../data_for_train/{:s}'.format(dataset)

    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    if opt.round == 0:
        # ----- sample points
        old_label_point_dir = '../data/{:s}/labels_point'.format(dataset)
        create_folder(label_point_dir)
        sample_points(old_label_point_dir, label_point_dir, opt.ratio, train_list)

        # ----- create detection label from points
        create_detect_label_from_points(label_point_dir, label_detect_dir, train_list, radius=opt.r1)

    # ------ split large images into 250x250 patches
    patch_folder = '{:s}/patches'.format(data_dir)
    create_folder(patch_folder)
    if opt.round == 0:
        split_patches(img_dir, '{:s}/images'.format(patch_folder))
        split_patches(label_detect_dir, '{:s}/labels_detect'.format(patch_folder), 'label_detect')
    else:
        split_patches(label_bg_dir, '{:s}/labels_bg'.format(patch_folder), 'label_bg')

    # ------ divide dataset into train, val and test sets
    organize_data_for_training(data_dir, train_data_dir, opt)

    # ------ compute mean and std
    if opt.round == 0:
        compute_mean_std(data_dir, train_data_dir, train_list)


def sample_points(label_point_dir, new_label_point_dir, ratio, train_list):
    image_list = os.listdir(label_point_dir)
    for imgname in sorted(image_list):
        if len(imgname) < 15 or imgname[-15:] != 'label_point.png':
            continue
        if '{:s}.bmp'.format(imgname[:-16]) not in train_list and '{:s}.png'.format(imgname[:-16]) not in train_list:
            continue

        points = io.imread('{:s}/{:s}'.format(label_point_dir, imgname))
        points_labeled, N = measure.label(points, return_num=True)
        indices = np.random.choice(range(1, N + 1), int(N * ratio))
        label_partial_point = np.isin(points_labeled, indices)

        io.imsave('{:s}/{:s}'.format(new_label_point_dir, imgname), (label_partial_point > 0).astype(np.uint8) * 255)


def create_detect_label_from_points(label_point_dir, label_detect_dir, img_list, radius):
    create_folder(label_detect_dir)

    for image_name in sorted(img_list):
        name = image_name.split('.')[0]
        points = io.imread('{:s}/{:s}_label_point.png'.format(label_point_dir, name))

        if np.sum(points > 0):
            label_detect = gaussian_filter(points.astype(np.float), sigma=radius/3)
            val = np.min(label_detect[points > 0])
            label_detect = label_detect / val
            label_detect[label_detect < 0.05] = 0
            label_detect[label_detect > 1] = 1
        else:
            label_detect = np.zeros(points.shape)

        # import utils
        # utils.show_figures((points, label_detect))
        io.imsave('{:s}/{:s}_label_detect.png'.format(label_detect_dir, name), label_detect.astype(np.float))


def split_patches(data_dir, save_dir, post_fix=None):
    import math
    """ split large image into small patches """
    create_folder(save_dir)

    image_list = os.listdir(data_dir)
    for image_name in image_list:
        name = image_name.split('.')[0]
        if post_fix and name[-len(post_fix):] != post_fix:
            continue
        image_path = os.path.join(data_dir, image_name)
        image = io.imread(image_path)
        seg_imgs = []

        # split into 16 patches of size 250x250
        h, w = image.shape[0], image.shape[1]
        patch_size = 250
        h_overlap = math.ceil((4 * patch_size - h) / 3)
        w_overlap = math.ceil((4 * patch_size - w) / 3)
        for i in range(0, h-patch_size+1, patch_size-h_overlap):
            for j in range(0, w-patch_size+1, patch_size-w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)

        for k in range(len(seg_imgs)):
            if post_fix:
                io.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(post_fix)-1], k, post_fix), seg_imgs[k])
            else:
                io.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])


def organize_data_for_training(data_dir, train_data_dir, opt):
    # --- Step 1: create folders --- #
    create_folder(train_data_dir)
    create_folder('{:s}/images/train'.format(train_data_dir))
    create_folder('{:s}/images/val'.format(train_data_dir))
    create_folder('{:s}/images/test'.format(train_data_dir))
    create_folder('{:s}/labels_detect/train'.format(train_data_dir))

    if opt.round > 0:
        create_folder('{:s}/labels_bg'.format(train_data_dir))
        create_folder('{:s}/labels_bg/train'.format(train_data_dir))

    # --- Step 2: move images and labels to each folder --- #
    # print('Organizing data for training...')
    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    if opt.round == 0:
        # train
        for img_name in train_list:
            name = img_name.split('.')[0]
            # images
            for file in glob.glob('{:s}/patches/images/{:s}_*'.format(data_dir, name)):
                file_name = file.split('/')[-1]
                dst = '{:s}/images/train/{:s}'.format(train_data_dir, file_name)
                shutil.copyfile(file, dst)
            # label detect
            for file in glob.glob('{:s}/patches/labels_detect/{:s}_*'.format(data_dir, name)):
                file_name = file.split('/')[-1]
                dst = '{:s}/labels_detect/train/{:s}'.format(train_data_dir, file_name)
                shutil.copyfile(file, dst)
        # val
        for img_name in val_list:
            # images
            for file in glob.glob('{:s}/images/{:s}'.format(data_dir, img_name)):
                file_name = file.split('/')[-1]
                dst = '{:s}/images/val/{:s}'.format(train_data_dir, file_name)
                shutil.copyfile(file, dst)
        # test
        for img_name in test_list:
            # images
            for file in glob.glob('{:s}/images/{:s}'.format(data_dir, img_name)):
                file_name = file.split('/')[-1]
                dst = '{:s}/images/test/{:s}'.format(train_data_dir, file_name)
                shutil.copyfile(file, dst)
    else:
        for img_name in train_list:
            name = img_name.split('.')[0]
            # background
            for file in glob.glob('{:s}/patches/labels_bg/{:s}_*'.format(data_dir, name)):
                file_name = file.split('/')[-1]
                dst = '{:s}/labels_bg/train/{:s}'.format(train_data_dir, file_name)
                shutil.copyfile(file, dst)


def compute_mean_std(data_dir, train_data_dir, train_list):
    """ compute mean and standarad deviation of training images """
    total_sum = np.zeros(3)  # total sum of all pixel values in each channel
    total_square_sum = np.zeros(3)
    num_pixel = 0  # total num of all pixels
    print('Computing the mean and standard deviation of training data...')

    for file_name in train_list:
        img_name = '{:s}/images/{:s}'.format(data_dir, file_name)
        img = io.imread(img_name)
        if len(img.shape) != 3 or img.shape[2] < 3:
            continue
        img = img[:, :, :3].astype(int)
        total_sum += img.sum(axis=(0, 1))
        total_square_sum += (img ** 2).sum(axis=(0, 1))
        num_pixel += img.shape[0] * img.shape[1]

    # compute the mean values of each channel
    mean_values = total_sum / num_pixel

    # compute the standard deviation
    std_values = np.sqrt(total_square_sum / num_pixel - mean_values ** 2)

    # normalization
    mean_values = mean_values / 255
    std_values = std_values / 255

    np.save('{:s}/mean_std.npy'.format(train_data_dir), np.array([mean_values, std_values]))
    np.savetxt('{:s}/mean_std.txt'.format(train_data_dir), np.array([mean_values, std_values]), '%.4f', '\t')


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


if __name__ == '__main__':
    from options import Options
    opt = Options(isTrain=True)
    main(opt)

