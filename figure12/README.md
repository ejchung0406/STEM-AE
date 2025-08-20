# Reproducing Figure 12 and Table 4: Speedup & Error evaluation of STEM on cycle-level simulator with DSE configs

This directory contains the necessary components to reproduce **Figure 12** and **Table 4** from our MICRO 2025 paper.

---

## Figure Description

Cycle count comparison between sampled and full simulation across GPU microarchitecture changes using various kernel sampling methods and workloads. This figure demonstrates STEM's effectiveness in cycle-level simulation environments by showing how different sampling techniques estimate total execution cycles when integrated with the MacSim simulator. The results validate STEM's accuracy for design space exploration scenarios where precise cycle counts are critical for architectural decisions.

---

## Contents

- `macsim/` - Directory for macsim simulator and results.
- `macsim.py` - Script for running macsim.
- `kernel_sample.py` - Script for reading the sample kernel IDs from pkl files and using the macsim simulator output to calculate the estimate of total number of cycles for each workload and compare the error.

- `1_trace.sh` - Script for downloading macsim traces.
- `2_run_macsim.sh` - Copy macsim binary to each directory and run macsim in parallel for each workload.
- `3_kernel_sample.sh` - Using the macsim output, estimate the total number of cycles with each kernel sampling technique.

- `figure12.xlsx` - MS Excel project file for generating the plots with result csv files. This generates similar plots as Figure 12 on the original paper, but only with a subset of workloads due to trace size overhead. 
---

## Reproducing the Figure

### 1. Download traces

Run `1_trace.sh` to download the traces. We excluded bloom and gpt2 as their trace sizes were too big to share with `gdown`. Instead, consider downloading traces of [gemma](https://www.dropbox.com/scl/fi/ewcyrogwv7odc6soi9v6n/gemma_nvbit.tar.gz?rlkey=arifvlad3kj9tcw6ogze7n04m&st=a06uyay0&dl=0) and [gpt2](https://www.dropbox.com/scl/fi/qn72hfwyeo5qq120kyade/gpt2_nvbit.tar.gz?rlkey=pal8q77bwf4iarypfts2osus3&st=97rzuvab&dl=0) on these Dropbox links and do the experiments -- copy subdirectories in `macsim/` and add the new workloads in `workloads` in `macsim.py`. The Bloom workload is not included due to its trace file exceeding 10 GB in size.

### 2. Run simulation

Run `2_run_macsim.sh` to run macsim and see how each kernel sampling technique estimates the total number of cycles.

### 3. Profiling and running kernel sampling

We have to run profiling on each workload and also run kernel sampling with each sampling technique to get the sample IDs for each workload and each sampling technique. However, to reduce the burden, in `pkl/`, we saved the list of kernel IDs that we obtained from running the same script as `figure7/sampling_methods/*.py`. Using this, we can calculate the total number of cycles with each sampling scheme. 

### 4. Calculate the total number of cycles for each sampling technique

By using macsim's output file that notes the number of cycles per each kernel, this Python script calculates the total number of cycles only with the kernels that are sampled from each sampling technique. Information about the sampled kernels is obtained from the `.pkl` files. The results are saved in the `results.csv` file. 

**Note**: The results in Table 4 were averaged over 11 Rodinia workloads and 6 LLM workloads. In this artifact evaluation, we include only 4 Rodinia workloads with relatively high error and, optionally, 2 LLM workloads (if the traces from the Dropbox link were used). As a result, individual workloads may exhibit larger errors or lower speedups than the averages reported in Table 4. However, the results should still reflect the overall trends presented in Figure 12. We appreciate your understanding that we could not share all trace files due to size constraints.
