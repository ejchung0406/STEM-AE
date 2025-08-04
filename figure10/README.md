# Reproducing Figure 10: Execution time histograms of clusters in previous works

This directory contains the necessary components to reproduce **Figure 10** from our MICRO 2025 paper.

---

## Figure Description

Distribution of execution times for kernels grouped as "identical" by previous sampling techniques, using the DLRM workload from the CASIO benchmark suite. The plot reveals that kernels classified as identical by traditional methods actually exhibit significantly different execution time distributions, demonstrating why these approaches introduce sampling error. This motivates the need for more sophisticated kernel sampling methodologies like STEM+ROOT that can better distinguish between kernels with similar signatures but different performance characteristics.

---

## Contents

- `prevworks.py` - Script for drawing the histograms. 
- `prevworks.csv` - Kernel execution time data from each cluster of sampling methods.

- `1_create_figure10.sh` - Script for drawing the scatterplot. 

---

## Reproducing the Figure

### 1. (Optional) Obtain execution time data for kernel clusters

We first have to collect execution time data for kernels that are considered to be in the same cluster by prior kernel sampling methods. This is possible by using the `export_for_figures` flag in Python scripts, such as using the `debug_pka()` function in `figure7/sampling_methods/pka.py`. Note that by using the command `uv run python -m figure7.sampling_methods.pka`, one can easily access the kernel IDs in a certain cluster. 

However, we found this process is not fully automated and is very burdensome to do for all workloads and sampling methods, so we provided a CSV file of `prevworks.csv` that we used for drawing the figure in the paper. `prevworks.csv` contains kernel execution time data for 6 clusters that were used during the kernel sampling of Rodinia and CASIO workload suites.

### 2. Draw the scatterplot

We can use the aforementioned CSV file to draw the histogram by running the command `./1_create_figure10.sh`.



