# -*- coding: utf-8 -*-
import multiprocessing as mp
from functools import partial
import scipy
import sknw
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import dct, idct
from skimage.filters.rank import sum as sum_filter
from skimage.morphology import medial_axis, skeletonize_3d
from .utils import plot_skel_3d
#from sato2000 import sato_2000_3d

from skimage.morphology import medial_axis
import warnings
warnings.filterwarnings("ignore")

#import dijkstra3d


def f(i, ct):
    ct_hat = np.zeros_like(ct).astype(float)
    ct_hat[:i + 1, 0:i + 1] = ct[:i + 1, :i + 1]

    image_hat = idct(idct(ct_hat.T, norm='ortho').T, norm='ortho')
    mask_hat = image_hat > 0.3

    m_hat = medial_axis(mask_hat).astype(float)

    return m_hat


def f_3d(i, ct):
    ct_hat = np.zeros_like(ct).astype(float)
    ct_hat[:i + 1, :i + 1, :i + 1] = ct[:i + 1, :i + 1, :i + 1]

    image_hat = idct(idct(idct(ct_hat, norm='ortho', axis=0), norm='ortho', axis=1), norm='ortho', axis=2)

    mask_hat = (image_hat > 0.3).astype(int)

    #ma_hat = skeletonize_3d(mask_hat).astype(float) / 255.0 # skeletonize_3d return values [0, 255]
    ma_hat = sato_2000_3d(mask_hat, return_distance_function=False).astype(float)

    return ma_hat


def cpma(mask, verbose=False, **kwargs):
    """

    :return:
    """
    if 'enforce_connectivity' not in kwargs:
        kwargs['enforce_connectivity'] = False

    if 'return_scores' not in kwargs:
        kwargs['return_scores'] = False

    if 'pruning' not in kwargs:
        kwargs['pruning'] = 'threshold'

    return_scores = kwargs['return_scores']
    enforce_connectivity = kwargs['enforce_connectivity']
    pruning = kwargs['pruning']

    if 'num_cpu' not in kwargs:
        kwargs['num_cpu'] = None

    num_cpu = kwargs['num_cpu']

    tau = 0.5
    gt_medial_axis, d = medial_axis(mask, return_distance=True)

    ct = dct(dct(mask.T, norm='ortho').T, norm='ortho')

    max_f = max(mask.shape[0], mask.shape[1])

    # Compute in parallel the reconstruction of the image using the cosine transform
    if num_cpu is None:
        pool = mp.Pool()
    else:
        pool = mp.Pool(num_cpu)

    ans = np.array(pool.map(partial(f, ct=ct), range(int(max_f/2))))
    pool.close()

    #ans = np.array([f(i, ct=ct) for i in range(int(max_f / 2))])

    # The score function is the mean of all reconstructions
    scores = ans.mean(axis=0)

    # We mask the score function to allow only values inside the object
    scores *= mask.astype(int)

    if pruning == 'incremental':
        # build the connected skeleton incremental from the highest values in the score function
        cpma = np.zeros_like(gt_medial_axis).astype(bool)
        scheduled = np.zeros_like(gt_medial_axis).astype(bool)
        rows = []
        cols = []
        ii, jj = np.where(np.logical_and(scores == scores.max(), scores >= tau))

        scheduled[ii, jj] = True

        rows.extend(ii.tolist())
        cols.extend(jj.tolist())
        while len(rows) > 0:
            r = rows.pop(0)
            c = cols.pop(0)
            cpma[r, c] = True

            ii, jj = np.where(
                np.logical_and(
                    np.logical_not(scheduled[r - 1:r + 2, c - 1:c + 2]),
                    scores[r - 1:r + 2, c - 1:c + 2] > tau
                )
            )

            scheduled[ii + r - 1, jj + c - 1] = True

            rows.extend((ii + r - 1).tolist())
            cols.extend((jj + c - 1).tolist())

    elif pruning == 'in_medial':
        # Computes the cpma as a subset of the medial axis
        cpma = gt_medial_axis.copy()
        # We remove the end points that has a value lower than tau
        selem = np.ones((3, 3))
        fil_1 = sum_filter(cpma.astype(int), selem)
        # get a mask with values where the end points are
        end_points_mask = np.logical_and(gt_medial_axis, fil_1 == 2)
        ii, jj = np.where(np.logical_and(end_points_mask, scores < 0.5))
        while ii.size > 0:
            cpma[ii, jj] = False
            fil_1 = sum_filter(cpma.astype(int), selem)
            # get a mask with values where the end points are
            end_points_mask = np.logical_and(gt_medial_axis, fil_1 == 2)
            ii, jj = np.where(np.logical_and(end_points_mask, scores < tau))
    elif pruning == 'threshold':
        cpma = scores > tau

    # Enforce connectivity
    if enforce_connectivity:
        connected_cpma = _compute_connected_medial_axis_2d(cpma, mask, scores)

    if verbose:
        if enforce_connectivity:
            plt.imshow(cpma, interpolation=None)
        else:
            plt.imshow(connected_cpma, interpolation=None)

        ''''# to zoom the images
        offset = int(0.5 * (mask.shape[0] - mask.shape[0] / 1.7))

        # plot score function
        plt.title('Score function')
        plt.imshow(scores[offset:-offset, offset:-offset], interpolation=None)
        plt.colorbar()

        # plot results for the CPMA
        plt.figure()
        plt.title('Cosine-Pruned Medial Axis (CPMA)')
        res_cpma = mask.astype(int) + cpma.astype(int)
        plt.imshow(res_cpma[offset:-offset, offset:-offset], interpolation=None)

        cpma_graph = sknw.build_sknw(cpma[offset:-offset, offset:-offset])
        node, nodes = cpma_graph.node, cpma_graph.nodes()
        ps = np.array([node[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')

        # plot the medial axis for comparison
        plt.figure()
        plt.title('Integer Medial Axis (ground truth)')
        res_gt_medial_axis = mask.astype(int) + gt_medial_axis.astype(int)
        plt.imshow(res_gt_medial_axis[offset:-offset, offset:-offset], interpolation=None)

        gt_medial_axis_graph = sknw.build_sknw(gt_medial_axis[offset:-offset, offset:-offset])
        node, nodes = gt_medial_axis_graph.node, gt_medial_axis_graph.nodes()
        ps = np.array([node[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')

        if enforce_connectivity:
            plt.figure()
            plt.title('Cosine-Pruned Medial Axis (CPMA) + connect.')
            res_connect_cpma = mask.astype(int) + connected_cpma.astype(int)
            plt.imshow(res_connect_cpma[offset:-offset, offset:-offset], interpolation=None)

            connected_cpma_graph = sknw.build_sknw(connected_cpma[offset:-offset, offset:-offset])
            node, nodes = connected_cpma_graph.node, connected_cpma_graph.nodes()
            ps = np.array([node[i]['o'] for i in nodes])
            plt.plot(ps[:, 1], ps[:, 0], 'r.')'''

        plt.show()

    if enforce_connectivity:
        cpma = connected_cpma

    if return_scores:
        return cpma, d, scores
    else:
        return cpma, d


def _compute_connected_medial_axis_2d(cpma, mask, scores):

    connected_cpma = cpma.copy()

    # build a graph with all the points in the mask of the
    field = (1.0 - scores) + np.prod(mask.shape) * np.logical_not(mask).astype(int)
    mask_graph = _build_2d_mask_graph(mask, field)

    # build graph from skeleton
    sk_graph = sknw.build_sknw(connected_cpma)
    sk_graph_components = [sk_graph.subgraph(c) for c in nx.connected_components(sk_graph)]

    # Iterates through all the subgraphs and tries to merge them
    max_iter = 10
    it = 0
    while it < max_iter and len(sk_graph_components) > 1:

        graph_i = sk_graph_components[0]
        graph_f = sk_graph_components[1]

        node, nodes = graph_i.nodes, graph_i.nodes()
        # Get the ids of all nodes of degree < 2, end points
        ids_graph_i = [int(node[i]['o'][0])*(mask.shape[1]) + int(node[i]['o'][1]) for i in nodes]

        node, nodes = graph_f.nodes, graph_f.nodes()
        # Get the ids of all nodes of degree < 2, end points
        ids_graph_f = [int(node[i]['o'][0]) * (mask.shape[1]) + int(node[i]['o'][1]) for i in nodes]

        # look for the shortest path from the subgraph i to each element of subgraph f
        min_length = np.finfo(float).max
        min_path = []
        for n in ids_graph_f:
            try:
                length, path = nx.algorithms.shortest_paths.weighted.multi_source_dijkstra(mask_graph,
                                                                                             sources=ids_graph_i,
                                                                                             target=n,
                                                                                             weight='energy')
                if length < min_length:
                    min_path = path
                    min_length = length
            except:
                pass

        # mark the minimum path
        for i in min_path:
            row, col = mask_graph.nodes[i]['cords']
            connected_cpma[row, col] = True

        # Creates the skeleton graph again with the new version of the CPMA
        sk_graph = sknw.build_sknw(connected_cpma)
        sk_graph_components = [sk_graph.subgraph(c) for c in nx.connected_components(sk_graph)]

        it += 1

    return connected_cpma


def _compute_connected_medial_axis_3d(cpma, mask, scores):

    connected_cpma = cpma.copy()

    # build a graph with all the points in the mask of the object
    field = (1.0 - scores) + np.prod(mask.shape) * np.logical_not(mask).astype(int)

    # build graph from skeleton
    sk_graph = sknw.build_sknw(connected_cpma)
    sk_graph_components = [sk_graph.subgraph(c) for c in nx.connected_components(sk_graph)]

    # Iterates through all the subgraphs and tries to merge them
    max_iter = 10
    it = 0
    n_components = len(sk_graph_components)
    while it < max_iter and n_components > 1:

        # Compute in parallel the reconstruction of the image using the cosine transform
        m = n_components if n_components % 2 == 0 else n_components - 1
        for j in range(0, m, 2):
            graph_i = sk_graph_components[j]
            graph_f = sk_graph_components[j + 1]

            # Get the ids of all nodes of degree < 2, end points
            node, nodes = graph_i.nodes, graph_i.nodes()
            ids_graph_i = []
            for i in nodes:
                #n_id = int(node[i]['o'][0]) * (mask.shape[1]) * (mask.shape[2]) + int(node[i]['o'][1]) * (
                #mask.shape[2]) + int(node[i]['o'][2])
                ids_graph_i.append((int(node[i]['o'][0]), int(node[i]['o'][1]), int(node[i]['o'][2])))

            # Get the ids of all nodes of degree < 2, end points
            node, nodes = graph_f.node, graph_f.nodes()
            ids_graph_f = []
            for i in nodes:
                #n_id = int(node[i]['o'][0]) * (mask.shape[1]) * (mask.shape[2]) + int(node[i]['o'][1]) * (
                #mask.shape[2]) + int(node[i]['o'][2])
                ids_graph_f.append((int(node[i]['o'][0]), int(node[i]['o'][1]), int(node[i]['o'][2])))

            # look for the shortest path from the subgraph i to each element of subgraph f
            min_length = np.finfo(float).max
            min_path = None

            for ns in ids_graph_i:
                for nf in ids_graph_f:
                    path = dijkstra3d.dijkstra(field, ns, nf)  # terminates early

                    length = field[tuple(path.T)].sum()

                    if length < min_length:
                        min_path = path
                        min_length = length

            if min_path is not None:
                connected_cpma[tuple(min_path.T)] = True

        # Creates the skeleton graph again with the new version of the CPMA
        sk_graph = sknw.build_sknw(connected_cpma)
        sk_graph_components = [sk_graph.subgraph(c) for c in nx.connected_components(sk_graph)]
        n_components = len(sk_graph_components)

        it += 1

    return connected_cpma


def _connect_two_paths(connected_cpma, mask, mask_graph, graph_i, graph_f):

    connected_cpma = connected_cpma.copy()
    # Get the ids of all nodes of degree < 2, end points
    node, nodes = graph_i.nodes, graph_i.nodes()
    ids_graph_i = []
    for i in nodes:
        n_id = int(node[i]['o'][0]) * (mask.shape[1]) * (mask.shape[2]) + int(node[i]['o'][1]) * (mask.shape[2]) + int(
            node[i]['o'][2])
        ids_graph_i.append(n_id)

    # Get the ids of all nodes of degree < 2, end points
    node, nodes = graph_f.node, graph_f.nodes()
    ids_graph_f = []
    for i in nodes:
        n_id = int(node[i]['o'][0]) * (mask.shape[1]) * (mask.shape[2]) + int(node[i]['o'][1]) * (mask.shape[2]) + int(
            node[i]['o'][2])
        ids_graph_f.append(n_id)

    # look for the shortest path from the subgraph i to each element of subgraph f
    min_length = np.finfo(float).max
    min_path = []
    for n in ids_graph_f:
        try:
            length, path = nx.algorithms.shortest_paths.weighted.multi_source_dijkstra(mask_graph,
                                                                                       sources=ids_graph_i,
                                                                                       target=n,
                                                                                       weight='energy')
            if length < min_length:
                min_path = path
                min_length = length
        except:
            pass
    # mark the minimum path
    for i in min_path:
        row, col, dep = mask_graph.node[i]['cords']
        connected_cpma[row, col, dep] = True

    return connected_cpma


def _build_2d_mask_graph(mask, weight_map):

    mask_graph = nx.Graph()
    # The +1 in each coordinates account for the buffer that sknw applies
    all_nodes = [(row * mask.shape[1] + col, row, col) for row, col in zip(*np.where(mask))]

    for node in all_nodes:
        node_id, row, col = node

        # add the node
        mask_graph.add_node(node_id, cords=(row, col))

        # Compute all neighbors of row and col
        ngb = np.array([np.arange(9) // 3 - 1, np.arange(9) % 3 - 1]) + np.array([[row], [col]])

        # deletes the column corresponding to row and col
        ngb = np.delete(ngb, 4, axis=1)
        # deletes all the neighbors that are not inside the object
        val = np.where(np.logical_not(mask[ngb[0, :], ngb[1, :]]))[0]
        ngb = np.delete(ngb, val, axis=1)

        # ids of all the neighbors
        ng_ids = ngb[0, :] * mask.shape[1] + ngb[1, :]

        # creates the nodes and edges for each pair of neighbors
        for i in range(ngb.shape[1]):
            # add the neighbor node
            mask_graph.add_node(ng_ids[i], cords=(ngb[0, i], ngb[1, i]))

            weight = (weight_map[row, col] + weight_map[ngb[0, i], ngb[1, i]]) / 2.0
            mask_graph.add_edge(node_id, ng_ids[i], energy=weight)

    return mask_graph


def _build_3d_mask_graph(mask, weight_map):

    mask_graph = nx.Graph()
    # The +1 in each coordinates account for the buffer that sknw applies
    for row, col, dep in zip(*np.where(mask)):
        node_id = row * mask.shape[1] * mask.shape[2] + col * mask.shape[2] + dep

        # add the node
        mask_graph.add_node(node_id, cords=(row, col, dep))

        # Compute all neighbors of row and col
        ngb = np.array(np.where(np.ones((3, 3, 3), dtype=bool))) - 1 + np.array([[row], [col], [dep]])

        # deletes the column corresponding to row and col
        ngb = np.delete(ngb, 13, axis=1)
        # deletes all the neighbors that are not inside the object
        #val = np.where(np.logical_not(mask[ngb[0, :], ngb[1, :], ngb[2, :]]))[0]
        #ngb = np.delete(ngb, val, axis=1)

        # ids of all the neighbors
        ng_ids = ngb[0, :] * mask.shape[1] * mask.shape[2] + ngb[1, :] * mask.shape[2] + ngb[2, :]

        # creates the nodes and edges for each pair of neighbors
        for i in range(ngb.shape[1]):
            # add the neighbor node
            mask_graph.add_node(ng_ids[i], cords=(ngb[0, i], ngb[1, i], ngb[2, i]))

            weight = (weight_map[row, col, dep] + weight_map[ngb[0, i], ngb[1, i], ngb[2, i]]) / 2.0
            mask_graph.add_edge(node_id, ng_ids[i], energy=weight)

    return mask_graph


def cpma_3d(mask, verbose=False, **kwargs):
    """

    :return:
    """
    if 'enforce_connectivity' not in kwargs:
        kwargs['enforce_connectivity'] = False

    if 'return_scores' not in kwargs:
        kwargs['return_scores'] = False

    if 'pruning' not in kwargs:
        kwargs['pruning'] = 'threshold'

    return_scores = kwargs['return_scores']
    enforce_connectivity = kwargs['enforce_connectivity']
    pruning = kwargs['pruning']

    if 'num_cpu' not in kwargs:
        kwargs['num_cpu'] = None

    num_cpu = kwargs['num_cpu']

    tau = 0.5
    #gt_medial_axis = skeletonize_3d(mask)
    gt_medial_axis = sato_2000_3d(mask, return_distance_function=False)
    d = scipy.ndimage.morphology.distance_transform_edt(mask)

    ct = dct(dct(dct(mask, norm='ortho', axis=0), norm='ortho', axis=1), norm='ortho', axis=2)

    max_f = max(mask.shape[0], mask.shape[1], mask.shape[2])

    # Compute in parallel the reconstruction of the image using the cosine transform
    if num_cpu is None:
        pool = mp.Pool()
    else:
        pool = mp.Pool(num_cpu)

    step = int(mask.shape[0] / 50)
    ans = np.array(pool.map(partial(f_3d, ct=ct), range(3, int(0.63*max_f), step))) # 0.63 impliys 25% of all the frequencies
    #ans = np.array([f_3d(i, ct=ct) for i in range(2, int(0.63*max_f), 1)])

    pool.close()

    # The score function is the mean of all reconstructions
    scores = ans.mean(axis=0)

    if pruning == 'incremental':
        # build the connected skeleton incremental from the highest values in the score function
        cpma = np.zeros_like(gt_medial_axis).astype(bool)
        scheduled = np.zeros_like(gt_medial_axis).astype(bool)
        rows = []
        cols = []
        ii, jj = np.where(np.logical_and(scores == scores.max(), scores >= tau))

        scheduled[ii, jj] = True

        rows.extend(ii.tolist())
        cols.extend(jj.tolist())
        while len(rows) > 0:
            r = rows.pop(0)
            c = cols.pop(0)
            cpma[r, c] = True

            ii, jj = np.where(
                np.logical_and(
                    np.logical_not(scheduled[r - 1:r + 2, c - 1:c + 2]),
                    scores[r - 1:r + 2, c - 1:c + 2] > tau
                )
            )

            scheduled[ii + r - 1, jj + c - 1] = True

            rows.extend((ii + r - 1).tolist())
            cols.extend((jj + c - 1).tolist())

    elif pruning == 'in_medial':
        # Computes the cpma as a subset of the medial axis
        cpma = gt_medial_axis.copy()
        # We remove the end points that has a value lower than tau
        selem = np.ones((3, 3))
        fil_1 = sum_filter(cpma.astype(int), selem)
        # get a mask with values where the end points are
        end_points_mask = np.logical_and(gt_medial_axis, fil_1 == 2)
        ii, jj = np.where(np.logical_and(end_points_mask, scores < 0.5))
        while ii.size > 0:
            cpma[ii, jj] = False
            fil_1 = sum_filter(cpma.astype(int), selem)
            # get a mask with values where the end points are
            end_points_mask = np.logical_and(gt_medial_axis, fil_1 == 2)
            ii, jj = np.where(np.logical_and(end_points_mask, scores < tau))
    elif pruning == 'threshold':
        cpma = scores > tau

    # Enforce connectivity
    if enforce_connectivity:
        cpma = _compute_connected_medial_axis_3d(cpma, mask, scores)

    if verbose:
        plot_skel_3d(cpma)
        plt.show()

    if return_scores:
        return cpma, d, scores
    else:
        return cpma, d
