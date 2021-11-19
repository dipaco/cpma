# -*- coding: utf-8 -*-
import numpy as np
import sys
import pickle
import os
import sknw
from time import sleep
from skimage.measure import find_contours
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def show_progress_bar(progress, message='Progress'):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write((message + ": [%-20s] %d%%") % ('='*progress, 5*progress))
    sys.stdout.flush()


def progress(count, total, status=''):
    """
    Show a progress bar. Taken from: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    :param count:
    :param total:
    :param status:
    :return:
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
    sleep(0.1)


def buffer_2d_image(image, buffer=(100, 100)):

    if len(buffer) == 2:
        buffer_cols = buffer[1]
        buffer_rows = buffer[0]
    else:
        buffer_cols = buffer
        buffer_rows = buffer

    rows, cols = buffer_rows + image.shape[0], buffer_cols + image.shape[1]
    n_image = np.zeros((rows, cols), dtype=int)
    row_offset, col_offset = int((n_image.shape[0] - image.shape[0]) / 2), int((n_image.shape[1] - image.shape[1]) / 2),
    n_image[row_offset:image.shape[0] + row_offset, col_offset:image.shape[1] + col_offset] = image
    return n_image


def buffer_3d_image(image, buffer=(100, 100, 100)):

    if len(buffer) == 3:
        buffer_cols = buffer[1]
        buffer_rows = buffer[0]
        buffer_depth = buffer[2]
    else:
        buffer_cols = buffer
        buffer_rows = buffer
        buffer_depth = buffer

    rows, cols, depth = buffer_rows + image.shape[0], buffer_cols + image.shape[1], buffer_depth + image.shape[2]
    n_image = np.zeros((rows, cols, depth), dtype=int)
    row_offset, col_offset, depth_offset = int((n_image.shape[0] - image.shape[0]) / 2), \
                                           int((n_image.shape[1] - image.shape[1]) / 2), \
                                           int((n_image.shape[2] - image.shape[2]) / 2)
    n_image[row_offset:image.shape[0] + row_offset, col_offset:image.shape[1] + col_offset, depth_offset:image.shape[2] + depth_offset] = image
    return n_image


def restore_buffered_image(expanded_image, original_shape):
    """
    Crop the buffer

    :param expanded_image: Expanded image that is desired to be in the original shape.
    :param original_shape: Original shape of the image.
    :return: The input image cropped to the original shape. THe object is placed in the center.
    """
    row_offset, col_offset = int((expanded_image.shape[0] - original_shape[0]) / 2), int(
        (expanded_image.shape[1] - original_shape[1]) / 2)
    expanded_image = expanded_image[row_offset:original_shape[0] + row_offset, col_offset:original_shape[1] + col_offset]
    return expanded_image


def square_image(mask, return_original_indexes=False):
    """
    Make de image square in centres the image in the new onw
    :param mask: image to make square sized
    :param return_original_indexes: True to return the indexes of the bounding box where the original images is located
    using the format -> (row_min, row_max, col_min, col_max). Default value is False.
    :return: squared size image
    """
    max_dim = np.max(mask.shape)
    border_mask = np.zeros((max_dim, max_dim), dtype=mask.dtype)
    row_min = (max_dim - mask.shape[0]) // 2
    row_max = row_min + mask.shape[0]
    col_min = (max_dim - mask.shape[1]) // 2
    col_max = col_min + mask.shape[1]
    border_mask[row_min:row_max, col_min:col_max] = mask

    if return_original_indexes:
        return border_mask, (row_min, row_max, col_min, col_max)
    else:
        return border_mask


def raster2xyz(input_raster, out_xyz, n_band=1, flt_val=1):
    """
    Transform a image to xyz
    :param input_raster:
    :param out_xyz:
    :param n_band:
    :param flt_val:
    :return:
    """


    """src_raster = gdal.Open(input_raster)
    raster_bnd = src_raster.GetRasterBand(n_band)
    raster_values = raster_bnd.ReadAsArray()

    gtr = src_raster.GetGeoTransform()

    y, x = np.where(raster_values == flt_val)

    gtr_x = gtr[0] + (x + 0.5) * gtr[1] + (y + 0.5) * gtr[2]
    gtr_y = gtr[3] + (x + 0.5) * gtr[4] + (y + 0.5) * gtr[5]

    data_vals = np.extract(raster_values == flt_val, raster_values)

    data_dict = {
        "x": gtr_x,
        "y": gtr_y,
        "z": data_vals
    }

    df = pd.DataFrame(data_dict)

    df.to_csv(out_xyz, index=False)

    src_raster = None"""

    print("New XYZ (csv file) created...")


def make_folders(base_path, dataset):
    '''
    Create all the paths to store results
    :param base_path: Base path
    :param dataset: The dataset being processed.
    :return: dictionary with all the paths for all results categories.
    '''
    results_paths = {
        'log': os.path.join(base_path, dataset.dataset_name(), 'logs'),
        'images': os.path.join(base_path, dataset.dataset_name(), 'images'),
        'noisy_images': os.path.join(base_path, dataset.dataset_name(), 'noisy_images'),
        'variables': os.path.join(base_path, dataset.dataset_name(), 'variables'),
        'tables': os.path.join(base_path, dataset.dataset_name(), 'tables'),
        'figures': os.path.join(base_path, dataset.dataset_name(), 'figures'),
        'data_augmentation': os.path.join(base_path, dataset.dataset_name(), 'data_augmentation'),
    }

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Creates the path to store all results
    for k in results_paths.keys():
        if not os.path.exists(results_paths[k]):
            os.makedirs(results_paths[k])
    return results_paths


def get_method_result_folders(folders_dict, method):
    '''
    Create all the paths to store results on an specific dataset
    :param folders_dict: Base path folders
    :return: dictionary with all the paths for all results categories on an specific dataset.
    '''

    folders_dict = folders_dict.copy()

    for k in folders_dict.keys():
        if k not in ['figures', 'tables', 'data_augmentation']:
            folders_dict[k] = os.path.join(folders_dict[k], method)

    # Creates the path to store all results
    for k in folders_dict.keys():
        if not os.path.exists(folders_dict[k]):
            os.makedirs(folders_dict[k])

    return folders_dict


def load_variables(filename):
    """
    Create or read from disk a file with all the variables for an image or a dataset.

    :param filename: Variables filename for an image in the dataset.
    :return: A dictionary containing the variables.
    """

    if not os.path.exists(filename):
        return {}

    with open(filename, 'rb') as f:
        variables = pickle.load(f)
    return variables


def save_variables(variables_filename, variables):
    """
    Save a dictionary of variables into disk. Replace existing variables.

    :param variables_filename: Variables filename for an image in the dataset.
    :param variables: Dictionary with all the images to store.
    :return:
    """

    stored_variables = load_variables(variables_filename)

    for key in variables:
        stored_variables[key] = variables[key]

    with open(variables_filename, 'wb') as f:
        pickle.dump(stored_variables, f)


def get_mask_contours(mask):
    all_contours = find_contours(mask, 0.8)
    contours = all_contours[0]
    for c in all_contours:
        if c.shape[0] > contours.shape[0]:
            contours = c
    return contours


def calc_normals_of_contours(contours):
    n_points = contours.shape[0]
    A = np.zeros((n_points, n_points))
    a = np.zeros((n_points))
    # a[-1:2] = np.array([-1, 0, 1])
    a[0:2] = np.array([0, 1])
    a[-1] = np.array([-1])
    for i in range(n_points):
        A[i, :] = a
        a = np.roll(a, 1)
    R = np.array([[0, -1], [1, 0]])
    N = np.dot(np.dot(A, contours), R)
    aux = np.linalg.norm(N, axis=1)
    N /= np.vstack((aux, aux)).T
    return N


def plot_skel_3d(mask_3d, plot_type='graph', show=True, markersize=2.0, linewidth=1.0):
    if plot_type == 'graph':
        return plot_skel_3d_graph(mask_3d, show=show, markersize=markersize, linewidth=linewidth)
    elif plot_type == 'voxel':
        return plot_skel_3d_voxel(mask_3d, show=show, markersize=markersize, linewidth=linewidth)
    else:
        raise ValueError('3D plot type: {} not supported. Valid values are "graph" or "voxel"'.format(plot_type))


def plot_skel_3d_voxel(mask_3d, show=True, markersize=2.0, linewidth=0.5):
    # For visualization
    #mask_3d = np.flip(mask_3d, axis=2).transpose(0, 2, 1)
    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.voxels(voxels, facecolors=colors, edgecolor='k')
    ax.voxels(mask_3d)
    #plt.savefig('skel_thining.png')

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    # ax.voxels(voxels, facecolors=colors, edgecolor='k')
    #ax.voxels(mask)
    #plt.savefig('voxel.png')

    if show:
        plt.show()


def plot_skel_3d_graph(mask_3d, show=True, markersize=2.0, linewidth=0.5):

    fig = plt.figure()
    ax = fig.gca(projection='3d')


    # build graph from skeleton
    graph = sknw.build_sknw(mask_3d)

    # draw edges by pts
    for (s, e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:, 0], ps[:, 1], ps[:, 2], 'green', linewidth=linewidth)

    # draw node by o
    node, nodes = graph.node, graph.nodes()
    ps = np.array([node[i]['o'] for i in nodes])
    plt.plot(ps[:, 0], ps[:, 1], ps[:, 2], 'r.', markersize=markersize)

    #axes limits
    plt.xlim(0, mask_3d.shape[0])
    plt.ylim(0, mask_3d.shape[1])
    ax.set_zlim(0, mask_3d.shape[2])

    # title and show
    #plt.title('Build Graph')
    if show:
        plt.show()
