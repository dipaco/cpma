# CPMA: Cosine-Pruned Medial Axis algorithm

The CPMA, is a method for medial axis pruning with noise robustness and equivariance to isometric transformations. It leverages the Discrete Cosine Transform to create smooth versions of a shape <inline-formula> <tex-math notation="LaTeX">$\Omega $ </tex-math></inline-formula>. We use the smooth shapes to compute a score function <inline-formula> <tex-math notation="LaTeX">$\mathcal {F}_{\Omega }$ </tex-math></inline-formula> that filters out spurious branches from the medial axis of the original shape <inline-formula> <tex-math notation="LaTeX">$\Omega $ </tex-math></inline-formula>. Our method generalizes to <inline-formula> <tex-math notation="LaTeX">$n$ </tex-math></inline-formula>-dimensional shapes given the properties of the Discrete Cosine Transform. 

# Installation

After cloning the repository, you will need to install all the necesary packages. We recommend creating a new conda environment using the `environment.yml` file we provide:

```angular2html
conda env create -f environment.yml
conda activate cpma 
```

After you install all the dependencies and create the conda environment, you can run the test file as:

```angular2html
python run_2d_test.py
```

The command should create a new folder named `results`. Inside this folder you will see a comparative figure for every image in the `data` folder. The comparative images should look like this:



If you find our code or paper useful, please consider citing

    @article{PatioCortes2021CosinePrunedMA,
        title={Cosine-Pruned Medial Axis: A New Method for Isometric Equivariant and Noise-Free Medial Axis Extraction},
        author={Diego Alberto Pati{\~n}o Cortes and John Willian Branch},
        journal={IEEE Access},
        year={2021},
        volume={9},
        pages={65466-65481}
    }
