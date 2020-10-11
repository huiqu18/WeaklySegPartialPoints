import os
import numpy as np
import random
import torch
from scipy.spatial import Voronoi
from skimage import draw
import torch.nn.functional as F


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


# borrowed from https://gist.github.com/pv/8036995
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def split_forward(model, input, size, overlap, outchannel=2):
    '''
    split the input image for forward passes
    '''

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), outchannel, h, w))
    for i in range(0, h-overlap, size-overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w-overlap, size-overlap):
            c_end = j+size if j+size < w else w

            input_patch = input[:,:,i:r_end,j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                output_patch = model(input_var)

            ind2_s = j+overlap//2 if j>0 else 0
            ind2_e = j+size-overlap//2 if j+size<w else w
            output[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]

    output = output[:,:,:h0,:w0].cuda()

    return output


def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if size == 0:
        with torch.no_grad():
            output = model(input.cuda())
    else:
        output = split_forward(model, input, size, overlap)
    output = output.squeeze(0)
    prob_maps = F.softmax(output, dim=0).cpu().numpy()

    return prob_maps


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b


def show_figures(imgs, new_flag=False):
    import matplotlib.pyplot as plt
    if new_flag:
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
    else:
        for i in range(len(imgs)):
            plt.figure(i+1)
            plt.imshow(imgs[i])

    plt.show()


# revised on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_txt(results, filename, mode='w'):
    """ Save the result of losses and F1 scores for each epoch/iteration
        results: a list of numbers
    """
    with open(filename, mode) as file:
        num = len(results)
        for i in range(num-1):
            file.write('{:.4f}\t'.format(results[i]))
        file.write('{:.4f}\n'.format(results[num-1]))


def save_results(header, all_result, test_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average results:\n')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(all_result[i]))
        file.write('{:.4f}\n'.format(all_result[N - 1]))
        file.write('\n')

        # results for each image
        for key, vals in sorted(test_results.items()):
            file.write('{:s}:\n'.format(key))
            for value in vals:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def compute_error_rate(dataset, labels_vor_dir, labels_cluster_dir, labels_instance_dir, ratio):
    import os
    from skimage import io
    import json

    with open('../data/{:s}/train_val_test.json'.format(dataset), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    results_vor = []
    results_cluster = []
    for filename in train_list:
        name = filename.split('.')[0]

        label_vor = io.imread('{:s}/{:s}_label_vor.png'.format(labels_vor_dir, name))
        label_cluster = io.imread('{:s}/{:s}_label_cluster.png'.format(labels_cluster_dir, name))
        label_instance = io.imread('{:s}/{:s}_label.png'.format(labels_instance_dir, name))

        # voronoi
        Vor_edges = label_vor[:,:,0]
        binary_label = label_instance > 0

        error_pixels = Vor_edges * binary_label
        error_num = np.sum(error_pixels>0)
        all_num = np.sum(Vor_edges>0)
        results_vor.append([error_num, all_num])

        # cluster
        nuclei_error_pixels = label_cluster[:, :, 1] * (~binary_label)
        bg_error_pixels = label_cluster[:, :, 0] * binary_label
        error_num_nuclei = np.sum(nuclei_error_pixels > 0)
        error_num_bg = np.sum(bg_error_pixels > 0)
        all_num_nuclei = np.sum(label_cluster[:, :, 1] > 0)
        all_num_bg = np.sum(label_cluster[:, :, 0] > 0)
        # results_cluster.append([error_num_nuclei, all_num_nuclei, error_num_bg, all_num_bg])
        results_cluster.append([error_num_nuclei+error_num_bg, all_num_nuclei+all_num_bg])

    results_vor = np.sum(np.array(results_vor), axis=0)
    error_rate_vor = results_vor[0] / results_vor[1]

    results_cluster = np.sum(np.array(results_cluster), axis=0)
    # error_rate_cluster_nuclei = results_cluster[0] / results_cluster[1]
    # error_rate_cluster_bg = results_cluster[2] / results_cluster[3]
    error_rate_cluster = results_cluster[0] / results_cluster[1]
    print('{:s}: {:.2f}: Vor {:.2f} %\tCluster {:.2f} %'.format(dataset, ratio, error_rate_vor*100,
                                                                error_rate_cluster*100))


def compute_avg_size(labels_instance_dir):
    import os
    from skimage import io
    import json

    with open('../data/LC/train_val_test.json', 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    # file_list = os.listdir(labels_instance_dir)
    results = []
    for filename in [*train_list, *val_list, *test_list]:
        name = filename.split('.')[0]
        # if name[-5:] != 'label':
        #     continue

        # if '{:s}.png'.format(name) not in train_list:
        #     continue

        label_instance = io.imread('{:s}/{:s}_label.png'.format(labels_instance_dir, name))

        num_nuclei = len(np.unique(label_instance)) - 1
        num_pixel = np.sum(label_instance>0)
        results.append([num_nuclei, num_pixel])

    result = np.sum(np.array(results), axis=0)
    avg_size = result[1] / result[0]
    radius = np.sqrt(avg_size/np.pi)
    print('Avg area: {:.2f}, radius: {:.2f}'.format(avg_size, radius))
    print('# of nuclei: {:d}'.format(result[0]))

# labels_instance_dir = '../data/LC/labels_instance'
# compute_avg_size(labels_instance_dir)


if __name__ == '__main__':
    # compute the error rate of two labels
    dataset = 'MO'
    for ratio in [0.05, 0.10, 0.25, 0.50, 1.00]:
        labels_vor_dir = '../data/{:s}/labels_voronoi_{:.2f}'.format(dataset, ratio)
        labels_cluster_dir = '../data/{:s}/labels_cluster_{:.2f}'.format(dataset, ratio)
        labels_instance_dir = '../data/{:s}/labels_instance'.format(dataset)
        compute_error_rate(dataset, labels_vor_dir, labels_cluster_dir, labels_instance_dir, ratio)


