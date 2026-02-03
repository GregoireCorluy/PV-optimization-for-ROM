![Python Version](https://img.shields.io/badge/python-3.10.10-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
<!--![DOI](https://zenodo.org/)(https://zenodo.org/) -->

# Progress variable optimization: Effects on the manifold topology and reduced-order modeling
This repository contains the code, data and results for the paper:

> G. Corlùy, K. Zdybał, A. Parente - [*Progress variable optimization: Effects on the manifold topology and reduced-order modeling*](doi...), under review for "Energy&AI", volume, (2026) page

To cite this publication:

```
@article{corluy2026progress,
  title={Progress variable optimization: Effects on the manifold topology and reduced-order modeling},
  author={Corlùy, Grégoire and Zdybał, Kamila and Parente, Alessandro},
  journal={Energy&AI},
  volume={},
  pages = {},
  issn = {},
  year={2026},
  publisher={},
  doi={},
}
```

## Background and motivation

Reduced-order models (ROMs) have become increasingly popular in many engineering and energy applications as a tool to design and optimize systems at a reduced computational cost. In the field of reacting flows, ROMs can be built by projecting the high-dimensional state space onto a low-dimensional manifold. In the past, the manifold parametrization was defined by physical variables based on expert knowledge. Recently, the encoder-decoder architecture has emerged as a tool to automatically optimize the manifold parametrization. Many successful implementations exist in the literature applying the encoder-decoder to find a manifold parametrization and perform a ROM simulation with it. However, there lacks a formal analysis to understand how an optimized parametrization performs better than a heuristic parametrization for a ROM simulation. Hence, this work compares the optimized parametrization with a heuristic one both *a priori* as *a posteriori* for a 0D hydrogen flame dataset. This work illustrates step-by-step how and why the optimized parametrization is better than the heuristic one and how it translates to the ROM performance.

## Graphical abstract

![Screenshot](Figures/graphical-abstract.png)

## Data

Describe what type of data and provide script to generate the data... Comes from code provided by Kamila

## Code

Describe different directories and files...

## Installation

- **Python version:** 3.10.10  
- **Dependencies:** All necessary packages are listed in `requirements_PV_ROM.txt` at the root of the repository.  
- To install them:

```bash
pip install -r requirements_PV_ROM.txt
```

> Note: This project also depends on the **[PCA-Fold](https://github.com/kamilazdybal/PCAfold)** library, which must be installed separately.

## Acknowledgements

This work was supported by the Walloon Region through the Fonds de La Recherche Scientifique - FNRS for the FRFS-WEL-T under Grant n. WEL-T-CR-2023 A - 07.

A special thank to Kamila Zdybał who provided many snippets of codes or even complete notebooks for this work. Moreover, she also provided invaluable advice for the experiments, the figures, the graphical abstract and the work in general.

## Contact

For any questions or inquiries, please contact Grégoire Corlùy at: gregoire.stephane.corluy@ulb.be

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
