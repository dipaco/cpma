# CPMA: Cosine-Pruned Medial Axis algorithm

This repository contains the official implementations of the CPMA/C-CPMA medial axis pruning. The CPMA, is a method for medial axis pruning with noise robustness and equivariance to isometric transformations. It leverages the Discrete Cosine Transform to create smooth versions of a shape S. We use the smooth shapes to compute a score function F(x, S) that filters out spurious branches from the medial axis of the original shape. Our method generalizes to n-dimensional shapes given the properties of the Discrete Cosine Transform. 

[ [Link to the paper](https://ieeexplore.ieee.org/document/9402852) ]

# Installation

After cloning the repository, you will need to install all the necessary packages. We recommend creating a new conda environment using the `environment.yml` file we provide:

```angular2html
conda env create -f environment.yml
conda activate cpma 
```

# 2D Tests

After you install all the dependencies and create the conda environment, you can run the 2D test file as:

```angular2html
python run_2d_test.py
```

The command should create a new folder named `results`. Inside this folder you will see a comparative figure for every image in the `data` folder. The comparative images should look like this:

![alt text](https://github.com/dipaco/cpma/blob/main/docs/medial_axis_figure_elephant12.png?raw=true)

# 3D Tests

COMING SOON!

If you find our code or our paper useful, please consider citing it.

```bibtex
@article{Patino2021CPMA,
    title={Cosine-Pruned Medial Axis: A New Method for Isometric Equivariant and Noise-Free Medial Axis Extraction},
    author={Diego Alberto Patiño Cortes and John Willian Branch},
    journal={IEEE Access},
    year={2021},
    volume={9},
    pages={65466-65481}
}
```
