# Reproducing Figure 1: Kernel Execution Time Histograms

This directory contains the necessary components to reproduce **Figure 1** from our MICRO 2025 paper.

---

## Figure Description

Figure 1 shows **execution time histograms** of repeatedly executed GPU kernels from ML workloads in the CASIO benchmark suite. Kernel names are shown above each plot. The plot highlights runtime heterogeneity in GPU kernel execution.

---

## Contents

- `profile_with_nsys.py` - python script to run CASIO workloads with NVIDIA Nsight Systems (`nsys`) to generate kernel profiling data.
- `runtime-example.csv` - A pre-collected CSV file containing kernel execution time data on an **RTX 2080 GPU**.
- `runtime.py` - Python script that reads `runtime-example.csv` and generates the histogram figure.
- `runtime.pdf` - Output figure that matches Figure 1 in the paper.

- `1_profile_with_nsys.sh` - Script for running the nsys script. 
- `2_create_figure1.sh` - script for running the plot create script.
---

## Reproducing the Figure

### 1. Prerequisites
You need a GPU to run profiling.
You do **not** need a GPU or to rerun profiling to generate the figure.

### 2. Profile the workloads

run `1_profile_with_nsys.sh` to obtain nsys profiles for each workload. they are saved in `results_nsys` directory with each names. `nsys_<workload_name>.csv` has execution time information for every kernel call in the workload.

### 3. Generate the figure

we have not automated the process of collecting the execution times and drawing it into a figure. instead, we provide a csv file named `runtime-example.csv` that has collections of 6 kernels and their execution times from workloads in CASIO. `runtime.py` uses these numbers to draw the histogram shown in the paper. run `2_create_figure1.sh` to obtain the same figure as the paper. 
