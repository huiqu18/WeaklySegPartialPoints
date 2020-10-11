import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology
import json


def main():
    data_dir = '../experiments/segmentation/LC/final_results'

    # read results
    all_results = {}
    for ratio in [0.05, 0.10, 0.25, 0.50, 1.00]:
        results = parse_results('{:s}/test_results_{:.2f}.txt'.format(data_dir, ratio))
        all_results[str(ratio)] = results

    img_names = list(all_results['0.05'].keys())
    N = len(img_names)
    k = 3
    results_acc = np.zeros((5, N))
    for i in range(N):
        img_name = img_names[i]
        results_acc[0, i] = all_results[str(0.05)][img_name][k]
        results_acc[1, i] = all_results[str(0.10)][img_name][k]
        results_acc[2, i] = all_results[str(0.25)][img_name][k]
        results_acc[3, i] = all_results[str(0.50)][img_name][k]
        results_acc[4, i] = all_results[str(1.00)][img_name][k]
    print(results_acc)


def show_all_diff():
    data_dir = '../experiments/segmentation/MO/final_results'

    # read results
    all_results = {}
    for ratio in [0.05, 0.10, 0.25, 0.50, 1.00]:
        results = parse_results('{:s}/test_results_{:.2f}.txt'.format(data_dir, ratio))
        all_results[str(ratio)] = results

    # analysis
    img_names = list(all_results['0.05'].keys())
    N = len(img_names)
    plt.figure()
    x = [i for i in range(4)]
    for i in range(N):
        img_name = img_names[i]
        results = []
        for ratio in [0.05, 0.10, 0.25, 0.50]:
            results.append(np.array(all_results[str(ratio)][img_name]) - np.array(all_results['1.0'][img_name]))
        results = np.array(results)

        print(img_name)
        print(results)

        # plt.subplot(N, 4, i*4+1)
        # plt.bar(x, results[:, 0])
        # # plt.ylabel(img_name)
        # plt.ylabel(img_name.split('_')[0])
        # # plt.ylim([-0.05, 0])
        # plt.title('Acc')
        # plt.subplot(N, 4, i*4+2)
        # plt.bar(x, results[:, 1])
        # # plt.ylim([-0.15, 0])
        # plt.title('F1')
        # plt.subplot(N, 4, i*4+3)
        # plt.bar(x, results[:, 2],)
        # # plt.ylim([-0.15, 0])
        # plt.title('Dice')
        # plt.subplot(N, 4, i*4+4)
        # plt.bar(x, results[:, 3])
        # # plt.ylim([-0.20, 0])
        # plt.title('AJI')

        plt.subplot(N, 4, i * 4 + 1)
        plt.bar(x, results[:, 0])
        plt.ylabel(img_name.split('_')[0])
        plt.ylim([-0.05, 0.01])
        plt.title('Acc')
        plt.subplot(N, 4, i * 4 + 2)
        plt.bar(x, results[:, 1])
        plt.ylim([-0.06, 0.01])
        plt.title('F1')
        plt.subplot(N, 4, i * 4 + 3)
        plt.bar(x, results[:, 2], )
        plt.ylim([-0.1, 0.025])
        plt.title('Dice')
        plt.subplot(N, 4, i * 4 + 4)
        plt.bar(x, results[:, 3])
        plt.ylim([-0.1, 0.1])
        plt.title('AJI')
    plt.show()


def parse_results(filepath):
    results = {}
    with open(filepath, 'r') as file:
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        res1 = file.readline().rstrip()
        res2 = file.readline().rstrip()
        while res1:
            fname = res1[:-1]
            metrics = res2[1:].split('\t')
            results[fname] = [float(metrics[0]), float(metrics[1]), float(metrics[4]), float(metrics[5])]
            res1 = file.readline().rstrip()
            res2 = file.readline().rstrip()

    return results


def compute_nuclei_variance(dataset, img_dir, labels_instance_dir):
    with open('../data/{:s}/train_val_test.json'.format(dataset), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    # file_list = os.listdir(labels_instance_dir)
    results = []
    for filename in [*test_list]:
        name = filename.split('.')[0]
        # if name[-5:] != 'label':
        #     continue

        # if '{:s}.png'.format(name) not in train_list:
        #     continue

        img = io.imread('{:s}/{:s}.png'.format(img_dir, name))
        label_instance = io.imread('{:s}/{:s}_label.png'.format(labels_instance_dir, name))
        unique_vals = np.unique(label_instance)
        unique_vals = unique_vals[unique_vals != 0]

        num_pixel = np.sum(label_instance > 0)
        num_nuclei = len(unique_vals)
        avg_std = 0
        for id in unique_vals:
            mask_i = label_instance == id
            # show_figures((img, mask_i))
            nuclei_i = img[label_instance == id, :]
            stds = np.sqrt(np.var(nuclei_i, axis=0))
            std_i = np.mean(stds)
            weight_i = np.sum(mask_i) / num_pixel
            # weight_i = 1.0 / num_nuclei
            avg_std += weight_i * std_i
            results.append(std_i)

        print('{:s}: {:.2f}'.format(name, avg_std))

    print('Average: {:.2f}'.format(np.mean(np.array(results))))
    np.save('{:s}_nuclei_std_all.npy'.format(dataset), results)

    # plt.figure()
    # plt.hist(results)
    # plt.show()


def compute_nuclei_bg_diff(dataset, img_dir, labels_instance_dir):
    with open('../data/{:s}/train_val_test.json'.format(dataset), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    results = []
    for filename in [*test_list]:
        name = filename.split('.')[0]

        img = io.imread('{:s}/{:s}.png'.format(img_dir, name))
        label_instance = io.imread('{:s}/{:s}_label.png'.format(labels_instance_dir, name))
        unique_vals = np.unique(label_instance)
        unique_vals = unique_vals[unique_vals != 0]

        num_pixel = np.sum(label_instance > 0)
        num_nuclei = len(unique_vals)
        avg_diff = 0
        for id in unique_vals:
            mask_i = label_instance == id
            mask_bg_i = morphology.binary_dilation(mask_i, morphology.disk(5)) * (~mask_i)

            # if name == 'Stomach_TCGA-KB-A93J-01A-01-TS1':
            #     plt.figure()
            #     plt.imshow(mask_i)
            #     plt.figure()
            #     plt.imshow(mask_bg_i)
            #     plt.figure()
            #     plt.imshow(img)
            #     plt.show()

            nuclei_i = img[mask_i, :]
            bg_i = img[mask_bg_i, :]

            mean_nuclei = np.mean(nuclei_i, axis=0)
            mean_bg = np.mean(bg_i, axis=0)

            diff = np.mean(np.abs(mean_nuclei - mean_bg))
            weight_i = np.sum(mask_i) / num_pixel
            avg_diff += weight_i * diff
        results.append(avg_diff)

        print('{:s}: {:.2f}'.format(name, avg_diff))
    print('Average: {:.2f}'.format(np.mean(np.array(results))))


def plot_histogram():
    import seaborn as sns
    MO_data = np.load('MO_nuclei_std_all.npy')
    LC_data = np.load('LC_nuclei_std_all.npy')
    # plt.figure()
    plt.hist(MO_data, bins=24, range=[0, 60], normed=False, color='r')
    plt.hist(LC_data, bins=24, range=[0, 60], normed=False, color='b')
    # plt.show()

    # sns.distplot(MO_data, hist=False, kde=True, kde_kws={'linewidth': 3}, label='MO')
    # sns.distplot(LC_data, hist=False, kde=True, kde_kws={'linewidth': 3}, label='LC')
    # plt.legend(prop={'size': 16})
    plt.xlabel('Nuclei color standard deviation')
    plt.ylabel('Density')
    plt.show()


if __name__ == '__main__':
    main()
    # plot_histogram()

    # dataset = 'MO'
    # img_dir = '../data/{:s}/images'.format(dataset)
    # labels_instance_dir = '../data/{:s}/labels_instance'.format(dataset)
    # # compute_nuclei_variance(dataset, img_dir, labels_instance_dir)
    # print('average diff between nuclei and background')
    # compute_nuclei_bg_diff(dataset, img_dir, labels_instance_dir)