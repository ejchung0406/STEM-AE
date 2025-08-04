# MICRO '25 Artifact Evaluation for the Paper "Swift and Trustworthy Large-Scale GPU Simulation"

## Abstract

This artifact accompanies our MICRO ’25 paper.

Goal: replicate key results from the paper — Figures 1, 7–12 and Tables 3, 4.

## Prerequisites

- `pip`
- Nsight Systems CLI and Nsight Compute CLI  
  - Install [Nsight Systems here](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)
- NVIDIA GPU with driver supporting CUDA 12.4 or higher

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
gdown ???
tar -xvf rodinia-data.tar.gz
rm rodinia-data.tar.gz

# Build NVBit tools
cd nvbit/photon
make
cd ../macsim-tracer
make
cd ../instr_count_bb
make
cd ../..
``` 
### Temporary fix for Casio
Edit `/fast_data/echung67/casio/rnnt/melfbank.py`
Line 124: add `return_complex=True`
Next line: add `x = torch.view_as_real(x)`