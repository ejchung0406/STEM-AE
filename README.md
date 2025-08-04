# MICRO '25 Artifact Evaluation for the Paper "Swift and Trustworthy Large-Scale GPU Simulation"

## Abstract

This artifact accompanies our MICRO '25 paper "Swift and Trustworthy Large-Scale GPU Simulation". The goal is to replicate key results from the paper — Figures 1, 7--12 and Tables 3, 4.

## Prerequisites

- `pip`
- Nsight Systems CLI and Nsight Compute CLI  
  - Install [Nsight Systems here](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)
- NVIDIA GPU with driver supporting CUDA 12.4 or higher

**Note:** A GPU is required for profiling workloads but is **not** required to generate figures from provided CSV data files.

## Artifact Structure

This artifact is organized into directories corresponding to paper figures and tables:

- **`figure1/`** - Kernel execution time histograms showing runtime heterogeneity in GPU kernels
- **`figure7/`** - Speedup and error evaluation of STEM vs. other sampling methods (includes Table 3 & 5)
- **`figure9/`** - Speedup vs. error scatter plots for CASIO and Huggingface suites  
- **`figure10/`** - Execution time histograms of kernel clusters from previous sampling works
- **`figure11/`** - Error bound sweep analysis of STEM methodology
- **`figure12/`** - Cycle-level simulator validation with MacSim (includes Table 4)
- **`workloads/`** - Benchmark suites: Rodinia, CASIO, Huggingface workloads
- **`macsim/`** - MacSim cycle-level GPU simulator
- **`nvbit/`** - NVBit-based profiling tools for kernel signature extraction

## Quick Start

```bash
# Install uv
pip install uv

# Download dependencies
uv venv

# Initialize submodules
git submodule update --init --recursive

# Build MacSim
uv run ./build.py --ramulator -j32

# Compile Rodinia
cd workloads/rodinia/gpu-rodinia
make
cd -

# Download Rodinia data
cd workloads/rodinia
gdown 1wPapvsb14v3Nn5DMxb_vU3xjgiMR6pTa
tar -xvf rodinia-data.tar.gz
rm rodinia-data.tar.gz
cd -

# Build NVBit tools
cd nvbit/photon
make
cd ../macsim-tracer
make
cd ../instr_count_bb
make
cd ../..
``` 

## Evaluation Workflow

### 1. Setup and Building (Required for all experiments)
Follow the Quick Start instructions above to set up the environment and build necessary components.

### 2. Figure Reproduction Guide

Each figure directory contains:
- Profiling scripts to collect workload data
- Pre-collected CSV files with results 
- Figure generation scripts
- Individual README with detailed instructions

### 3. Key Experiments

**Figure 1**: Demonstrates kernel execution time heterogeneity
- Shows why traditional sampling methods fail
- Can be reproduced with provided `runtime-example.csv`

**Figures 7-8 & Tables 3,5**: Core STEM evaluation
- Compares 5 sampling methods across 3 benchmark suites
- Pre-profiled data available via download scripts
- Full profiling takes months for complete suites

**Figure 9**: Speedup vs. accuracy trade-offs
- Scatter plots showing STEM's superior performance
- Uses results from Figure 7 experiments

**Figure 10**: Analysis of previous sampling method limitations
- Shows execution time distributions within "identical" kernel clusters
- Demonstrates why traditional methods introduce errors

**Figure 11**: STEM parameter sensitivity analysis
- Error bound sweep for STEM methodology
- Validates STEM's robustness

**Figure 12 & Table 4**: Cycle-level simulator validation
- Integration with MacSim for design space exploration
- Demonstrates STEM's effectiveness in simulation workflows

### **CRITICAL: Required CASIO Fix**
**This fix is MANDATORY and must be applied before running any CASIO workloads.**

The CASIO benchmark suite uses a deprecated PyTorch function that will cause workloads to fail. You MUST edit `/fast_data/echung67/casio/rnnt/melfbank.py`:

```python
# Line 124: Change this line to add return_complex=True parameter
OLD: x = torch.sfft(...)  
NEW: x = torch.sfft(..., return_complex=True)

# Line 125: Add this new line immediately after
x = torch.view_as_real(x)
```

**Without this fix, CASIO workloads will fail with PyTorch compatibility errors.**

For detailed instructions on reproducing specific figures, refer to the README files in each directory.