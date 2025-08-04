#!/usr/bin/env python
import numpy as np
from scipy.stats import norm
import csv
import matplotlib.pyplot as plt

# Set matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]

# Read the data from histogram.csv
with open('runtime-example.csv', 'r', encoding='utf-8') as file:
  reader = csv.reader(file)
  header = next(reader)

# Read data with missing value handling
data = np.genfromtxt('runtime-example.csv', delimiter=',', skip_header=1, encoding='utf-8', missing_values='', filling_values=np.nan)

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

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
  bin = [40, 30, 20, 50, 30, 50][i]
  n, bins, patches = plt.hist(x, bin, density=0, facecolor='red', alpha=0)
  
  y = norm.pdf(bins, mu, sigma)

  if i == 0:
    axs[i//3, i%3].set_xlim([18, 20.4])
    axs[i//3, i%3].hist(x, bin, density=0, facecolor='black', alpha=0.75)
    axs[i//3, i%3].plot(bins, y * (sum(n) * np.diff(bins)[0]), 'r--', linewidth=2)
  elif i == 1:
    x1 = x[(x < 5) & (x > 0)]
    x2 = x[(x > 5) & (x < 7.5)]
    x3 = x[x > 10]
    axs[i//3, i%3].hist(x1, bin, density=0, facecolor='black', alpha=0.75)
    axs[i//3, i%3].hist(x2, bin, density=0, facecolor='black', alpha=0.75)
    axs[i//3, i%3].hist(x3, bin, density=0, facecolor='black', alpha=0.75)
    mu1 = np.mean(x1)
    sigma1 = np.std(x1)
    mu2 = np.mean(x2)
    sigma2 = np.std(x2)
    mu3 = np.mean(x3)
    sigma3 = np.std(x3)
    n1, bins1, patches1 = plt.hist(x1, bin, density=0, facecolor='red', alpha=0)
    n2, bins2, patches2 = plt.hist(x2, bin, density=0, facecolor='red', alpha=0)
    n3, bins3, patches3 = plt.hist(x3, bin, density=0, facecolor='red', alpha=0)
    y1 = norm.pdf(bins1, mu1, sigma1)
    y2 = norm.pdf(bins2, mu2, sigma2)
    y3 = norm.pdf(bins3, mu3, sigma3)
    axs[i//3, i%3].plot(bins1, y1 * (sum(n1) * np.diff(bins1)[0]), 'r--', linewidth=2)
    axs[i//3, i%3].plot(bins2, y2 * (sum(n2) * np.diff(bins2)[0]), 'r--', linewidth=2)
    axs[i//3, i%3].plot(bins3, y3 * (sum(n3) * np.diff(bins3)[0]), 'r--', linewidth=2)
  elif i == 2:
    x1 = x[x < 2.3]
    x2 = x[x >= 2.3]
    axs[i//3, i%3].hist(x1, 7, density=0, facecolor='black', alpha=0.75)
    axs[i//3, i%3].hist(x2, 15, density=0, facecolor='black', alpha=0.75)
    mu1 = np.mean(x1)
    sigma1 = np.std(x1)
    mu2 = np.mean(x2)
    sigma2 = np.std(x2)
    n1, bins1, patches1 = plt.hist(x1, 14, density=0, facecolor='red', alpha=0)
    n2, bins2, patches2 = plt.hist(x2, 30, density=0, facecolor='red', alpha=0)
    y1 = norm.pdf(bins1, mu1, sigma1)
    y2 = norm.pdf(bins2, mu2, sigma2)
    axs[i//3, i%3].plot(bins1, y1 * (sum(n1) * np.diff(bins1)[0]) * 2, 'r--', linewidth=2)
    axs[i//3, i%3].plot(bins2, y2 * (sum(n2) * np.diff(bins2)[0]) * 2, 'r--', linewidth=2)
    axs[i//3, i%3].set_xlim([1.5, 3.5])
    axs[i//3, i%3].set_ylim([0, 350])
  elif i == 3:
    axs[i//3, i%3].set_xlim([45, 80])
    axs[i//3, i%3].set_ylim([0, 50])
    axs[i//3, i%3].hist(x, bin, density=0, facecolor='black', alpha=0.75)
    axs[i//3, i%3].plot(bins, y * (sum(n) * np.diff(bins)[0]), 'r--', linewidth=2)
  elif i == 4:
    x1 = x[x < 20]
    x2 = x[x > 20]
    axs[i//3, i%3].hist(x1, bin, density=0, facecolor='black', alpha=0.75)
    axs[i//3, i%3].hist(x2, bin, density=0, facecolor='black', alpha=0.75)
    mu1 = np.mean(x1)
    sigma1 = np.std(x1)
    mu2 = np.mean(x2)
    sigma2 = np.std(x2)
    n1, bins1, patches1 = plt.hist(x1, bin, density=0, facecolor='red', alpha=0)
    n2, bins2, patches2 = plt.hist(x2, bin, density=0, facecolor='red', alpha=0)
    y1 = norm.pdf(bins1, mu1, sigma1)
    y2 = norm.pdf(bins2, mu2, sigma2)
    axs[i//3, i%3].plot(bins1, y1 * (sum(n1) * np.diff(bins1)[0]), 'r--', linewidth=2)
    axs[i//3, i%3].plot(bins2, y2 * (sum(n2) * np.diff(bins2)[0]), 'r--', linewidth=2)
  elif i == 5:
    axs[i//3, i%3].hist(x, bin, density=0, facecolor='black', alpha=0.75)
    axs[i//3, i%3].set_xlim([6, 8.5])
    axs[i//3, i%3].set_ylim([0, 8000])

  axs[i//3, i%3].set_title(f"{header[i]}", fontsize=24, fontweight='bold')
  axs[i//3, i%3].tick_params(labelsize=16, rotation=0)
  axs[i//3, i%3].grid(True)
  # axs[i//3, i%3].set_yticks(axs[i//3, i%3].get_yticks()[::2])

fig.supxlabel('Kernel Execution Time (us)', fontsize=24)
fig.supylabel('# Kernel Calls', fontsize=24)

# Adjust the spacing to prevent labels from being cut off
plt.tight_layout()

# Save the figure
plt.savefig('runtime.pdf')

