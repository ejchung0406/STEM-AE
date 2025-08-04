# Reproducing Figure 11: Error bound sweep of STEM

This directory contains the necessary components to reproduce **Figure 11** from our MICRO 2025 paper.

---

## Figure Description

TODO

---

## Contents

- `sweep.py` - script for drawing figure 11.

- `1_kernel_sample.sh` - Script for collecting the result.
- `2_create_figure11.sh` - script for drawing figure 11.

---

## Reproducing the Figure

### 1. run 

run `./1_kernel_sample.sh` to obtain the results. the results are stored in `figure11/results/stem-casio-sweep.csv`

### 2. draw the figure

run `./2_create_figure11.sh` to draw the figure 11. the process of gathering values from the aformentioned csv file to this plot drawing script is not automted so the script already has the values hard coded in `sweep.py`. 