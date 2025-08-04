# Reproducing Figure 1: Kernel Execution Time Histograms

This directory contains the components needed to reproduce **Figure 1** from our MICRO 2025 paper.

---

## Figure Description

Figure 1 shows **execution time histograms** of repeatedly executed GPU kernels from ML workloads in the CASIO benchmark suite. Kernel names are shown above each plot. The figure highlights runtime heterogeneity in GPU kernel execution.

---

## Contents

- `profile_with_nsys.py` — Runs CASIO workloads with NVIDIA Nsight Systems (`nsys`) to generate kernel profiling data.
- `runtime-example.csv` — Pre-collected kernel execution time data on an **RTX 2080 GPU**.
- `runtime.py` — Reads `runtime-example.csv` and generates the histogram figure.
- `runtime.pdf` — Output figure matching Figure 1 from the paper.
- `1_profile_with_nsys.sh` — Shell script to run the profiling.
- `2_create_figure1.sh` — Shell script to generate the figure.

---

## Reproducing the Figure

### 1. Prerequisites

- A GPU is required to run profiling.
- A GPU is **not** required to generate the figure from the provided CSV.

### 2. Profile the Workloads

Run `1_profile_with_nsys.sh`. This will generate Nsight profiles for each workload in the `results_nsys/` directory. Each profile includes a `nsys_<workload_name>.csv` file with kernel execution time data.

### 3. Generate the figure

We have not automated the process of collecting the execution times and drawing them into a figure. Instead, we provide a CSV file named `runtime-example.csv` that contains collections of 6 kernels and their execution times from workloads in CASIO. `runtime.py` uses these numbers to draw the histogram shown in the paper. Run `2_create_figure1.sh` to obtain the same figure as the paper. 


