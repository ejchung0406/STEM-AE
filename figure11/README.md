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

### 1. Run 

Run `./1_kernel_sample.sh` to obtain the results. The results are stored in `figure11/results/stem-casio-sweep.csv`.

### 2. Draw the figure

Run `./2_create_figure11.sh` to draw figure 11. The process of gathering values from the aforementioned CSV file to this plot drawing script is not automated so the script already has the values hard coded in `sweep.py`. 