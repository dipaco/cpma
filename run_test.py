import os
import imageio
import time
from medial_axis import cpma, cpma_3d
from matplotlib import pyplot as plt
from skimage.morphology import medial_axis as baseline_medial_axis

if __name__ == '__main__':

    BASE_FOLDER = 'data'
    RESULTS_FOLDER = 'results'

    # get all images in the base folder
    all_images = [f for f in os.listdir(BASE_FOLDER) if any([f.endswith(ext) for ext in ['tif', 'png', 'jpg']])]

    # create the results dir
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    n_rows = 2
    n_cols = 3
    plt.axis('off')
    plt.tight_layout()
    fig = plt.figure(dpi=300)
    for im_file in all_images:
        im = imageio.imread(os.path.join(BASE_FOLDER, im_file))

        # Simple thresholding to create a binary mask for the shape
        mask = im > 128

        # d
        start = time.time()
        ma, dist, scores_function = cpma(mask, return_scores=True)
        cpma_time = (time.time() - start)

        start = time.time()
        connected_ma, dist = cpma(mask, enforce_connectivity=True)
        c_cpma_time = (time.time() - start)

        # We use scikit image implementation as the baseline medial axis
        start = time.time()
        noise_medial_axis = baseline_medial_axis(mask)
        baseline_ma_time = (time.time() - start)

        ax = plt.subplot(n_rows, n_cols, 1)
        plt.imshow(mask, cmap='gray', interpolation=None)
        ax.set_title('Mask')

        ax = plt.subplot(n_rows, n_cols, 2)
        plt.imshow(mask, cmap='gray', interpolation=None)
        ax.set_title('Mask')

        ax = plt.subplot(n_rows, n_cols, 3)
        plt.imshow(scores_function, interpolation=None)
        ax.set_title('Score function')

        ax = plt.subplot(n_rows, n_cols, 4)
        plt.imshow(mask.astype(int) + noise_medial_axis.astype(int), interpolation=None)
        ax.set_title(f'Baseline ({round(baseline_ma_time, 2)} s)')

        ax = plt.subplot(n_rows, n_cols, 5)
        plt.imshow(mask.astype(int) + ma.astype(int), interpolation=None)
        ax.set_title(f'CPMA ({round(cpma_time, 2)} s)')

        ax = plt.subplot(n_rows, n_cols, 6)
        plt.imshow(mask.astype(int) + connected_ma.astype(int), interpolation=None)
        ax.set_title(f'C-CPMA ({round(c_cpma_time, 2)} s)')

        plt.savefig(os.path.join(RESULTS_FOLDER, f'medial_axis_figure_{os.path.splitext(im_file)[0]}.png'))

