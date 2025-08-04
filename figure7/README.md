# Reproducing Figure 7, 8 and Table 3: Speedup & Error evaluation of STEM

This directory contains the necessary components to reproduce **Figure 7, 8** and **Table 3, 5** from our MICRO 2025 paper.

---

## Figure Description

Speedup comparison of four kernel sampling methods on the Rodinia and CASIO benchmark suites. The speedup is presented in log-scale, with the average speedup shown on the far right. Sampling error comparison of four sampling methods on Rodinia and CASIO suites. STEM+ROOT shows near-zero sampling error on CASIO suite as it leverages the massive number of kernel calls and their execution time distributions. Average speedup (×) and error (%) of 5 kernel sampling methods on 3 GPU benchmark suites. Some values not available due to excessive overhead.

---

## Contents

- `profile_workloads.py` - Profiles workloads to get metrics used in PKA, Sieve, Photon, and STEM. Uses nsys, ncu, and custom nvbit tool. 
- `kernel_sample.py` - Script for running kernel sampling on various workloads and sampling methodologies. 

- `1_profile_workloads.sh` - Script for running the profiling script. 
- `2_download_profiles.sh` - Script for downloading and extracting the profiles. 
- `3_kernel_sample.sh` - Script for running kernel sampling.

---

## Reproducing the Figure

### 1. Prerequisites
You need a GPU to run profiling.
You do **not** need a GPU to do kernel sampling from our provided workload profiles.

### 2. Profile the workloads

Run `1_profile_workloads.sh` to obtain workload profiles using nsys, ncu, and nvbit tool for Photon kernel signatures. The results are saved in `results_ncu/` and `results_photon/`. Please check generated nsys, ncu files as well as `bbv.csv` files to see how kernel signatures are generated for each kernel. 

### 3. Download the profiles for more workloads

The script above only runs through workloads in the Rodinia suite with small input configuration since it takes over months to run the full Rodinia and CASIO workloads. Also, it is infeasible to run ncu and Photon on Huggingface workloads. Instead, we provide our profiled results that can be downloaded from Dropbox. Run `2_download_profiles.sh` to download and extract the workload profiles. 

### 4. Run kernel sampling

Run `3_kernel_sample.sh` to do kernel sampling using downloaded workload profiles. The results are stored in `results/` with each CSV file for Rodinia, CASIO, and Huggingface suites. How we use this CSV file to draw the figures is not automated. We take the average from each iteration and used matplotlib to draw the figures. The default script has iter=1 to save your time. If you want to run the full evaluation, set iter=10 for all suites in `kernel_sample.py` and average the results from each iteration. One iteration takes around 30 minutes to run.

### 5. Gathering the timing information to fill in Table 5

`profile_workloads.py` has a feature where it can measure the time required for collecting each stat used for nsys, ncu, and nvbit tool for Photon. We used nvbit for measuring the time for Sieve and it requires using the `nvbit/instr_count_bb` tool for time measurement but it is not included here as ncu can already measure the instruction count.