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

- `figure7.xlsx` - MS Excel project file for generating the plots with result csv files. This generates similar plots as Figure 7 and 8 on the original paper. 

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

### 5. Measuring Profiling Overhead for Table 5

The overhead of each method and the associated profiler is calculated as the **time difference** between running a workload with and without its required profiler. We used the `figure7/profile_workloads.py` script to automate these measurements. The script executes each workload, captures the elapsed time via functions like `run_nsys()` and `run_ncu()`, and prints the raw timing data to standard output.

The specific profiling tool used for each kernel sampling method is as follows:

- **PKA**: NVIDIA's **Nsight Compute** (`ncu`) to collect 12 hardware performance metrics.
- **Sieve**: The `instr_count_bb` tool from the **NVBit library** to count kernel instructions.
- **Photon**: A custom-designed Basic Block Vector (BBV) extractor built on **NVBit**.
- **STEM (Ours)**: NVIDIA's **Nsight Systems** (`nsys`).

We collected the timing data from these scripts to obtain the data for Table 5. 

---

### Discrepancies with the Original Paper
#### Randomness Involved
Please note that the data obtained from the AE code may differ slightly from the results shown in the paper. This is because many of the evaluated kernel sampling methods involve randomness, and our AE code only runs a single iteration with default settings. You can modify this by changing the function arguments where `iter=1`. Even without this change, however, you should still observe trends consistent with those shown in the paper.
#### PKA and Sieve Sampling Methods on Some Rodinia Workloads
As noted in Section 8 of the paper, for the _gaussian_ and _heartwall_ workloads, we manually tuned previous methods (PKA and Sieve) to sample kernels randomly instead of always selecting the first chronological kernel, as this yielded better performance. We did this to ensure our method was compared against the best possible versions of prior methods, even if that required additional fine-tuning. However, this tuning is not implemented in our AE code, so PKA and Sieve may show larger errors on these workloads compared to what is reported in the paper.
#### Sieve Sampling Method on CASIO Workloads
Please also note that a naïve implementation of Sieve (one of the previous methods) shows very poor speedup on CASIO workloads. This is also stated on page 9 of the original paper:

> For CASIO workloads, we disabled Sieve’s additional clustering using KDE (kernel density estimation), as it led to oversampling and limited speedups to below 2–5× on each workload.

In the provided `.xlsx` file, the Sieve data is generated with KDE clustering enabled, as this is the default setting in our AE code. To obtain speedup and error numbers closer to those in the paper, you should disable KDE clustering only for CASIO workloads. To do this, please edit `figure7/sampling_methods/sieve.py` by uncommenting line 82 and commenting out lines 83–97.