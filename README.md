# CPMA: Cosine-Pruned Medial Axis algorithm

The CPMA, is a method for medial axis pruning with noise robustness and equivariance to isometric transformations. It leverages the Discrete Cosine Transform to create smooth versions of a shape <inline-formula> <tex-math notation="LaTeX">$\Omega $ </tex-math></inline-formula>. We use the smooth shapes to compute a score function <inline-formula> <tex-math notation="LaTeX">$\mathcal {F}_{\Omega }$ </tex-math></inline-formula> that filters out spurious branches from the medial axis of the original shape <inline-formula> <tex-math notation="LaTeX">$\Omega $ </tex-math></inline-formula>. Our method generalizes to <inline-formula> <tex-math notation="LaTeX">$n$ </tex-math></inline-formula>-dimensional shapes given the properties of the Discrete Cosine Transform. 

If you find our code or paper useful, please consider citing

    @article{PatioCortes2021CosinePrunedMA,
        title={Cosine-Pruned Medial Axis: A New Method for Isometric Equivariant and Noise-Free Medial Axis Extraction},
        author={Diego Alberto Pati{\~n}o Cortes and John Willian Branch},
        journal={IEEE Access},
        year={2021},
        volume={9},
        pages={65466-65481}
    }
