# Self-Organizing Maps — From Scratch in NumPy

A from-scratch implementation of Kohonen Self-Organizing Maps (SOM) in NumPy, with experiments on the Iris and synthetic blob datasets. Course project for Introduction to Machine Learning, IIT Kharagpur.

## Overview

The SOM is implemented end to end with the three classic phases — **competition** (best-matching-unit search), **cooperation** (Gaussian neighborhood function), and **adaptation** (weight update with exponentially decaying learning rate α and neighborhood radius σ).

## Results

- **98% cluster purity on Iris** (Setosa/Versicolor/Virginica)
- Grid-size ablation (Iris quantization error): 6×6 → 0.123, 8×8 → 0.110, 10×10 → 0.118 (best: 8×8)
- 100% separation on synthetic 4-cluster blob data
- Visualizations: U-matrix, component planes, and hit maps

See `24AI10039_SOMs.pdf` for the full report and `experiment_results.txt` for raw numbers.

## Files

| File | Description |
|---|---|
| `som_implementation.py` | From-scratch SOM class + experiment driver |
| `24AI10039_SOMs.pdf` | Full project report |
| `som_comprehensive_results.png` | Combined results figure |
| `experiment_results.txt` | Quantitative results dump |
| `iris_umatrix.pdf`, `iris_component_planes.pdf` | Iris visualizations |
| `blobs_umatrix.pdf`, `hit_maps_comparison.pdf` | Blob + hit-map visualizations |

## Run

```bash
pip install numpy matplotlib scikit-learn
python som_implementation.py
```

## Method

```
for each iteration t:
    pick input x
    BMU = argmin_i ||x - w_i||          # competition
    h_ci = exp(-d(c,i)^2 / (2 σ(t)^2))  # cooperation
    w_i += α(t) · h_ci · (x - w_i)      # adaptation
    α(t), σ(t) decay exponentially
```

Implemented with `numpy.random.seed(42)` for reproducibility.
