import kimimaro
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.morphology import label
from skimage.measure import regionprops
from cloudvolume import PrecomputedSkeleton
from phdcode.utils.utils import plot_skel_3d


def sato_2000_3d(mask, verbose=False, **kwargs):

    if 'return_distance_function' not in kwargs:
        kwargs['return_distance_function'] = True

    return_distance_function = kwargs['return_distance_function']

    if mask.astype(int).sum() == 0.0:
        return np.zeros(mask.shape)

    # Gets the biggest element in the array
    labels = label(mask)
    props = regionprops(labels)
    if len(props) > 0:
        max_label_obj = max(props, key=lambda o: o.area)
        labels = np.asfortranarray(labels == max_label_obj.label)
    else:
        labels = mask

    teasar = np.zeros_like(mask)

    if return_distance_function:
        d = scipy.ndimage.morphology.distance_transform_edt(mask)

    skels = kimimaro.skeletonize(
        labels,
        teasar_params={
            'scale': 1.00,
            'const': 0.0,  # physical units
            'pdrf_exponent': 4,
            'pdrf_scale': 100000,
            'soma_detection_threshold': 1100,  # physical units
            'soma_acceptance_threshold': 3500,  # physical units
            'soma_invalidation_scale': 1.0,
            'soma_invalidation_const': 300,  # physical units
            'max_paths': None,  # default None
        },
        # object_ids=[ ... ], # process only the specified labels
        dust_threshold=0,  # skip connected components with fewer than this many voxels
        anisotropy=(1, 1, 1),  # default True
        fix_branching=False,  # default True
        fix_borders=True,  # default True
        progress=False,  # default False, show progress bar
        parallel=1,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
    )
    #skels[1].viewer()

    if len(skels.keys()) > 0:
        idx = skels[1].vertices[:, 0].astype(int), skels[1].vertices[:, 1].astype(int), skels[1].vertices[:, 2].astype(int)
        teasar[idx] = True

    if verbose:
        plot_skel_3d(teasar)
        plt.show()

    if return_distance_function:
        return teasar, d
    else:
        return teasar

