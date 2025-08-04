import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# Set default font size
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]

# Read CSV file
# Assuming the CSV file has columns: label, name, xval, yval
df = pd.read_csv('scatter.csv')

# Separate the data by label
label_1 = df[df['label'] == "PKA"]
label_2 = df[df['label'] == "STEM"]
label_3 = df[df['label'] == "Sieve"]
label_4 = df[df['label'] == "Photon"]

label_5 = df[df['label'] == "PKA-mean"]
label_6 = df[df['label'] == "STEM-mean"]
label_7 = df[df['label'] == "Sieve-mean"]
label_8 = df[df['label'] == "Photon-mean"]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Create scatter plot for each label
axs[0].scatter(label_1['xval'], label_1['yval'], color='r', label='PKA')
axs[0].scatter(label_3['xval'], label_3['yval'], color='b', label='Sieve')
axs[0].scatter(label_4['xval'], label_4['yval'], color='y', label='Photon')
axs[0].scatter(label_2['xval'], label_2['yval'], color='g', label='STEM+ROOT')

axs[0].scatter(label_5['xval'], label_5['yval'], color='k', marker='x', s=200, linewidths=5)
axs[0].scatter(label_6['xval'], label_6['yval'], color='k', marker='x', s=200, linewidths=5)
axs[0].scatter(label_7['xval'], label_7['yval'], color='k', marker='x', s=200, linewidths=5)
axs[0].scatter(label_8['xval'], label_8['yval'], color='k', marker='x', s=200, linewidths=5)

# Add the names as text labels for each mean
axs[0].text(label_5['xval'] * 1.2, label_5['yval'] + 2, "PKA mean", fontsize=18, fontweight='bold')
axs[0].text(label_6['xval'] * 1.4, label_6['yval'] - 2, "STEM+ROOT mean", fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
axs[0].text(label_7['xval'] * 0.3, label_7['yval'] + 4, "Sieve mean", fontsize=18, fontweight='bold')
axs[0].text(label_8['xval'] * 1.2, label_8['yval'] + 3, "Photon mean", fontsize=18, fontweight='bold')

# Add labels and title
axs[0].set_xscale('log')
# plt.yscale('log')

formatter = ScalarFormatter()
formatter.set_scientific(False)  # Disable scientific notation
formatter.set_useOffset(False)
plt.gca().xaxis.set_major_formatter(formatter)

axs[0].set_title('ML Workloads (Casio)', fontweight='bold')
axs[0].legend(loc='upper right')

####################################################################################################

df = pd.read_csv('scatter-hugging.csv')

# Clear the current figure
# plt.clf()

# Separate the data by label
label_2 = df[df['label'] == "STEM"]
label_3 = df[df['label'] == "Random"]

label_5 = df[df['label'] == "STEM-mean"]
label_6 = df[df['label'] == "Random-mean"]

# Create scatter plot for each label
axs[1].scatter(label_2['xval'], label_2['yval'], color='y', label='STEM+ROOT')
axs[1].scatter(label_3['xval'], label_3['yval'], color='g', label='Random')

axs[1].scatter(label_5['xval'], label_5['yval'], color='k', marker='x', s=200, linewidths=5)
axs[1].scatter(label_6['xval'], label_6['yval'], color='k', marker='x', s=200, linewidths=5)

# Add the names as text labels for each mean
axs[1].text(label_5['xval'] / 2, label_5['yval'] + 0.25, "STEM+ROOT mean", fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
axs[1].text(label_6['xval'] * 1.2, label_6['yval'] + 0.1, "Random mean", fontsize=18, fontweight='bold')


# Add labels and title
axs[1].set_xscale('log')
# plt.yscale('log')
axs[1].set_title('LLM & ML Workloads (Huggingface)', fontweight='bold')

fig.supxlabel('Speedup (log scale)', fontweight='bold')
fig.supylabel('Error (%)', fontweight='bold')

plt.gca().xaxis.set_major_formatter(formatter)
# Add a legend
plt.legend()

# Save the plot
plt.savefig('scatter.pdf')
