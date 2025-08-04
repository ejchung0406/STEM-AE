#!/usr/bin/env python
import numpy as np
from scipy.stats import norm
import csv
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]

# Read the data from histogram.csv
with open('prevworks.csv', 'r') as file:
  reader = csv.reader(file)
  header = next(reader)

data = np.genfromtxt('prevworks.csv', delimiter=',', skip_header=1)

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 7))

for i in range(6):
  # Extract the x vectors from the data
  x = data[:, i]

  # Ignore NaN values in x
  x = x[~np.isnan(x)]
  x /= 1000

  # Calculate the mean and standard deviation of the x vectors
  mu = np.mean(x)
  sigma = np.std(x)

  # Plot i
  bin = [50, 50, 70, 70, 20, 20][i]
  n, bins, patches = plt.hist(x, bin, density=0, facecolor='red', alpha=0)
  
  y = norm.pdf(bins, mu, sigma)

  if i == 0:
    axs[i%2, i//2].hist(x, bin, density=0, facecolor='black', alpha=0.75)
    axs[i%2, i//2].set_xlim([2, 12])
    axs[i%2, i//2].set_ylim([0, 60])
  elif i == 1:
    axs[i%2, i//2].hist(x, bin, density=0, facecolor='black', alpha=0.75)
    axs[i%2, i//2].set_xlim([1, 5])
    axs[i%2, i//2].set_ylim([0, 1100])
  elif i == 2:
    axs[i%2, i//2].hist(x, bin, density=0, facecolor='black', alpha=0.75)
    axs[i%2, i//2].set_xlim([4.5, 8.5])
    axs[i%2, i//2].set_ylim([0, 40])
  elif i == 3: 
    axs[i%2, i//2].hist(x, bin, density=0, facecolor='black', alpha=0.75)
    axs[i%2, i//2].set_xlim([0, 13])
    axs[i%2, i//2].set_ylim([0, 50])
  elif i == 4:
    axs[i%2, i//2].hist(x, bin, density=0, facecolor='black', alpha=0.75)
    axs[i%2, i//2].set_xlim([2, 2.75])
    axs[i%2, i//2].set_ylim([0, 750])
  elif i == 5:
    axs[i%2, i//2].hist(x, bin, density=0, facecolor='black', alpha=0.75)
    axs[i%2, i//2].set_xlim([1.25, 1.55])
    axs[i%2, i//2].set_ylim([0, 40])
  axs[i%2, i//2].set_title(f"{header[i]}", fontsize=28, fontweight='bold')
  axs[i%2, i//2].tick_params(labelsize=18, rotation=0)
  axs[i%2, i//2].grid(True)
  # axs[i%2, i//2].set_yticks(axs[i%2, i//2].get_yticks()[::2])

fig.supxlabel('Kernel Execution Time (us)', fontsize=24)
fig.supylabel('# Kernel Calls', fontsize=24)

# Adjust the spacing to prevent labels from being cut off
plt.tight_layout()

# Save the figure
plt.savefig('prevworks.pdf')

