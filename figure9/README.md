# Reproducing Figure 9: Speedup & Error scatter plot

This directory contains the necessary components to reproduce **Figure 9** from our MICRO 2025 paper.

---

## Figure Description

Scatter plot showing the speedup (log scale) and error (%) of different kernel sampling methods on CASIO suite (left) and Huggingface suite (right). The plot demonstrates the trade-off between sampling speedup and accuracy across different workloads, with STEM+ROOT achieving near-zero sampling error on CASIO suite while maintaining significant speedup through its advanced kernel sampling methodology.

---

## Contents

- `scatter.py` - Script for drawing the scatterplot. 
- `scatter.csv`, `scatter-hugging.csv` - Speedup, error values we obtained from kernel sampling.

- `1_create_figure9.sh` - Script for drawing the scatterplot. 

---

## Reproducing the Figure

### 1. (Optional) Obtain speedup and error

We first have to collect the speedup and error values of kernel sampling from CASIO and Huggingface workloads. This can be done with using the scripts that we used in `../figure7/` and collecting them in a CSV file. However, as this process is not automated, we included the CSV file that we used. The value for CASIO is named `scatter.csv` and for Huggingface is `scatter-hugging.csv`. 

### 2. Draw the scatterplot

We can use the two aforementioned CSV files to draw the scatter plot by simply running the command `./1_create_figure9.sh`.



