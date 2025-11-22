# CUSP
[![Status](https://img.shields.io/badge/Status-Published-success.svg)](#citation)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md) 

This repository is the official implementation for the paper: 

**"CUSP: Complex Spike Sorting from Multi-electrode Array Recordings with U-net Sequence-to-Sequence Prediction"**.

You can access the published paper here:
[https://doi.org/10.1016/j.jneumeth.2025.110631](https://doi.org/10.1016/j.jneumeth.2025.110631)


## Example Usage
To run the CUSP complex spike sorting on your data, you can use the provided Jupyter notebook `run_CSsort.ipynb`. This notebook demonstrates how to set up the parameters, load data (e.g., channel*sample .bin file with `int16` type), and execute the sorting algorithm.

## Dataset

The data associated with this project has been officially published and is openly available for access and reuse. You can find the full, versioned dataset on **Zenodo** by following this link: [https://zenodo.org/records/17673850](https://zenodo.org/records/17673850).

## Citation

If you use the methods described in our paper or our published dataset, please cite the following publication:

Bao C, Mildren RL, Charles AS, Cullen KE. CUSP: Complex Spike Sorting from Multi-electrode Array Recordings with U-net Sequence-to-Sequence Prediction. **J Neurosci Methods**. 2025 Nov 18:110631. doi: 10.1016/j.jneumeth.2025.110631.

```bibtex
@article{Bao2025CUSP,
    title = {{CUSP: Complex Spike Sorting from Multi-electrode Array Recordings with U-net Sequence-to-Sequence Prediction}},
    author = {Bao, C and Mildren, RL and Charles, AS and Cullen, KE},
    journal = {{J Neurosci Methods}},
    year = {2025},
    month = {Nov 18},
    doi = {10.1016/j.jneumeth.2025.110631},
    publisher = {Elsevier},
}
```

---
