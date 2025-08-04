# Reproducing Figure 12 and Table 4: Speedup & Error evaluation of STEM on cycle-level simulator with DSE configs

This directory contains the necessary components to reproduce **Figure 12** and **Table 4** from our MICRO 2025 paper.

---

## Figure Description

TODO

---

## Contents

- `macsim/` - directory for macsim simulator and results
 
- `1_trace.sh` - script for downloading macsim traces
- `2_run_macsim.sh` - copy macsim binary to each directory and run macsim in parallel for each workload
- `3_kernel_sample.sh` - using the macsim output, estimate the total number of cycles with each kernel sampling technique.
---

## Reproducing the Figure

### 1. download traces

run `1_trace.sh` to download the traces. we excluded bloom and gpt2 as their trace sizes were too big to share with `gdown`. instead, consider downloading traces of [gemma](https://www.dropbox.com/scl/fi/ewcyrogwv7odc6soi9v6n/gemma_nvbit.tar.gz?rlkey=arifvlad3kj9tcw6ogze7n04m&st=a06uyay0&dl=0) and [gpt2](https://www.dropbox.com/scl/fi/qn72hfwyeo5qq120kyade/gpt2_nvbit.tar.gz?rlkey=pal8q77bwf4iarypfts2osus3&st=97rzuvab&dl=0) on these dropbox links and do the experiments -- copy subdirectories in `macsim/` and add the new workloads in `workloads` in `macsim.py`. 

### 2. run simulation

run `2_run_macsim.sh` to run macsim and see how each kernel sampling techniques estimate the total number of cycles. 

### 3. (optional) profiling and running kernel sampling

we have to run profiling on each workloads and also run kernel sampling with each sampling techniques to get the sample ids for each workloads and each sampling technique. however, to reduce the burden, in `pkl/`, we saved the list of kernel ids that we obtained from running the same script as `figure7/sampling_methods/*.py`. using this, we can calculate the total number of cycles with each sampling scheme. 

### 4. calculating the total number of cycles for each sampling technique

by using the macsim's output file that notes number of cycles per each kernel, this python script calculates the total number of cycles only with the kernels that are sampled from each sampling technique. information about the sampled kernels are obtained from the `.pkl` files. the results are saved in `results.csv` file. 