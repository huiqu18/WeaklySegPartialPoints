import os
import cv2 as cv
import numpy as np
from skimage import io
import stainNorm_Reinhard
import matplotlib.pyplot as plt



def compute_mean_std(data_dir, save_dir, postfix):
    """ compute mean and standarad deviation of training images """
    total_sum = np.zeros(3)  # total sum of all pixel values in each channel
    total_square_sum = np.zeros(3)
    num_pixel = 0  # total num of all pixels
    print('Computing the mean and standard deviation of training data...')

    img_list = os.listdir(data_dir)
    for file_name in img_list:
        img_name = '{:s}/{:s}'.format(data_dir, file_name)
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
    print(mean_values)
    print(std_values)

    np.save('{:s}/mean_std_{:s}.npy'.format(save_dir, postfix), np.array([mean_values, std_values]))


data_dir = '../data/MO/original_images'
save_dir = '../data_for_train/MO/images_normalized_by_LC/test'
os.makedirs(save_dir, exist_ok=True)

img_list = os.listdir('../data_for_train/MO/images/test')

ref_img_path = '../data/LC/images/193-a2-1.png'
ref_img = cv.cvtColor(cv.imread(ref_img_path), cv.COLOR_BGR2RGB)
normalizer = stainNorm_Reinhard.normalizer()
normalizer.fit(ref_img)

for i in range(0, len(img_list)):
    img_name = img_list[i]
    img_path = '{:s}/{:s}.png'.format(data_dir, img_name[:-4])
    img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

    # perform reinhard color normalization
    img_normalized = normalizer.transform(img)
    img_normalized2 = cv.cvtColor(img_normalized, cv.COLOR_RGB2BGR)

    cv.imwrite('{:s}/{:s}.png'.format(save_dir, img_name[:-4]), img_normalized2)

compute_mean_std(save_dir, '../data_for_train/MO', '193-a2-1')
