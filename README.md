# MICRO '25 Artifact evaluation for the paper "Swift and Trustworthy Large-Scale GPU Simulation"

## Abstract

This artifact accompanies our MICRO ’25 paper.

goal: demonstrates key results from the paper, including figure 1. 

## Prerequisites

- pip 
- nsight systems cli, nsight compute cli
  - Use [this](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html) install the nsight systems. 
- Nvidia GPU with driver supporting cuda 12.4 or higher

## Quick Start

```bash

# Install uv
pip install uv

# Download dependencies
uv venv

# Install macsim and casio
git submodule update --init --recursive

# build macsim
uv run ./build.py --ramulator -j32

# Compile rodinia
cd workloads/rodinia/gpu-rodinia
make
cd -

# Build Macsim
cd macsim
uv run ./build.py --ramulator -j32

# Build NVBit tool for Photon's kernel profiler, Macsim tracer, and Sieve's instruction counter.
cd nvbit/photon
make
cd ../macsim-tracer
make 
cd ../instr_count_bb
make
cd ../..

# fix error in casio
open /fast_data/echung67/casio/rnnt/melfbank.py. in line 124, add `return_complex=True` and at the next line add `x = torch.view_as_real(x)`. 

```

The container includes PyTorch, matplotlib, and NVIDIA Nsight Systems for GPU profiling.

## Contents

there are each directory for each figures that this repo reproduce, and readme files for each figure are in each directory. 