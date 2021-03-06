import os
import imageio
import time

import numpy as np
from medial_axis import cpma, cpma_3d
from matplotlib import pyplot as plt
from skimage.morphology import medial_axis as baseline_medial_axis
from tqdm import tqdm


def plot_image(axis, image, title, cmap='gray', interpolation=None):
    axis.axis('off')
    axis.imshow(image, cmap=cmap, interpolation=interpolation)
    axis.set_title(title, fontsize=8)


if __name__ == '__main__':

    BASE_FOLDER = 'data'
    RESULTS_FOLDER = 'results'

    # get all images in the base folder
    all_images = [f for f in os.listdir(BASE_FOLDER) if any([f.endswith(ext) for ext in ['tif', 'png', 'jpg']])]

    # create the results dir
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    n_rows = 2
    n_cols = 3
    fig, axs = plt.subplots(2, 3, dpi=300)
    for im_file in tqdm(all_images):
        im = imageio.imread(os.path.join(BASE_FOLDER, im_file))

        # Lets use only one channel
        im = im[..., 0] if len(im.shape) > 2 else im

        # Simple thresholding to create a binary mask for the shape
        mask = im > 128

        # adds padding to both sides of the image
        ld = int(1.1 * max(*mask.shape))
        aux_mask = np.zeros([ld, ld], dtype=bool)
        idx = np.where(mask)
        idx = (idx[0] + (ld - mask.shape[0]) // 2, idx[1] + (ld - mask.shape[1]) // 2)
        aux_mask[idx] = True
        mask = aux_mask

        # Computes the CPMA and the C-CPMA
        start = time.time()
        ma, dist, score_function = cpma(
            mask,
            tau=0.47,
            enforce_connectivity=False,
            return_scores=True
        )
        cpma_time = (time.time() - start)

        start = time.time()
        # NOTE: This uses the modified energy function. To achieve the paper's results set energy_func='inverse'.
        connected_ma, dist = cpma(
            mask,
            tau=0.47,
            enforce_connectivity=True,
            connect_max_iter=50,
            energy_func='exponential',
            alpha=10.0,
        )
        c_cpma_time = (time.time() - start)

        # We use scikit image implementation as the baseline medial axis
        start = time.time()
        noise_medial_axis = baseline_medial_axis(mask)
        baseline_ma_time = (time.time() - start)

        # Plot the results
        plot_image(axs[0, 0], mask, 'Mask')
        plot_image(axs[0, 1], score_function, title='Score function', cmap='viridis')
        plot_image(axs[0, 2], dist, title='Distance transform', cmap='viridis')
        plot_image(axs[1, 0], mask.astype(int) + noise_medial_axis.astype(int), title=f'Baseline ({round(baseline_ma_time, 2)} s)', cmap='viridis')
        plot_image(axs[1, 1], mask.astype(int) + ma.astype(int), title=f'CPMA ({round(cpma_time, 2)} s)', cmap='viridis')
        plot_image(axs[1, 2], mask.astype(int) + connected_ma.astype(int), title=f'C-CPMA ({round(c_cpma_time, 2)} s)', cmap='viridis')

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_FOLDER, f'medial_axis_figure_{os.path.splitext(im_file)[0]}.png'))

