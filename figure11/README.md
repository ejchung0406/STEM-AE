# Reproducing Figure 11: Error bound sweep of STEM

This directory contains the necessary components to reproduce **Figure 11** from our MICRO 2025 paper.

---

## Figure Description

Impact of varying the error bound (ε) on speedup and sampling error for STEM. This figure demonstrates STEM's parameter sensitivity by showing how larger ε values enhance speedup with increased error. The analysis validates STEM's robustness across different error bound settings, allowing users to tune the trade-off between simulation speed and accuracy according to their requirements.

---

## Contents

- `sweep.py` - Script for drawing figure 11.

- `1_kernel_sample.sh` - Script for collecting the result.
- `2_create_figure11.sh` - Script for drawing figure 11.

---

## Reproducing the Figure

### 1. Run the Experiment

Run `./1_kernel_sample.sh` to generate the results. The output will be saved to `figure11/results/stem-casio-sweep.csv`.

### 2. Plot the Figure

Run `./2_create_figure11.sh` to draw Figure 11.

> **Note:** This script does **not** automatically extract values from the CSV file.  
> The values are manually hardcoded in `sweep.py`.
