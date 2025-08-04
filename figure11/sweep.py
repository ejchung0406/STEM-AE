import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 1, figsize=(6, 4))

#################################################

metrics = ["bert-infer", "bert-train", "dlrm-infer", "dlrm-train", "resnet-infer", "resnet-train", "rnnt-train", "ssdrn34-infer", "ssdrn34-train", "unet-infer", "unet-train", "mean"]
y = {
  "ε=3%": (821.8339123,813.9872375,92.08994608,90.49503373,148.4913131,134.1041539,229.2352225,52.77482571,31.19827119,40.60647329,39.46862399,76.45637443),
  "ε=5%": (924.1427113,1046.040598,99.56268991,114.2279517,207.3322684,172.5186436,321.4905809,90.86040032,63.02563352,48.8818698,47.33157451,106.771173),
  "ε=10%": (1144.586283,1549.790621,119.1967023,129.6378337,253.832126,204.141264,424.9168716,143.4668302,145.3750739,97.21036799,91.79682965,172.2965846),
  "ε=25%": (1738.027064,2174.403024,161.7103897,150.0932312,293.612392,260.5806329,799.1358298,254.5171756,251.8404366,113.657472,110.3902105,228.5316157),
}

x = np.arange(len(metrics))
width = 0.20
multiplier = -0.5

for attribute, measurement in y.items():
  offset = width * multiplier
  rects = axs[0].bar(x + offset, measurement, width, label=attribute)

  last_bar = rects[-1]
  height = last_bar.get_height()
  xpos = last_bar.get_x() + last_bar.get_width() / 2

  # Manually annotate just the last bar
  axs[0].text(xpos, height, f"{height:.2f}", ha='center', va='bottom', fontsize=8)
  multiplier += 1

axs[0].set_ylabel('Speed Up (X)', fontsize=12)
# axs[0].set_title('DRAM read/write access')
axs[0].set_xticks(x + width, metrics, rotation=20)
axs[0].grid('grid', linestyle='dashed', color='gray', zorder=0, axis='y')
axs[0].set_axisbelow(True)
axs[0].legend(loc='upper left', ncols=5)
axs[0].set_ylim(1, 100000)
axs[0].set_yscale("log")

#################################################

metrics = ["bert-infer", "bert-train", "dlrm-infer", "dlrm-train", "resnet-infer", "resnet-train", "rnnt-train", "ssdrn34-infer", "ssdrn34-train", "unet-infer", "unet-train", "mean"]
y = {
  "ε=3%": (0.147419986,0.185924998,0.222138505,0.225853244,0.162137563,0.12644845,0.238662962,0.213435426,0.146719821,0.138487813,0.208858078,0.183280622),
  "ε=5%": (0.232904213,0.232891317,0.721024975,0.372910073,0.490432989,0.122765267,0.768186605,0.436750322,0.387111899,0.42539111,0.306584774,0.408813959),
  "ε=10%": (1.504114838,0.769047729,0.831472122,0.690857985,0.348489848,0.297949088,1.429542499,1.505505544,0.40877788,0.80073639,0.243043606,0.80268523),
  "ε=25%": (3.134817527,1.638453811,1.60046987,1.861675301,1.070518849,1.333800086,3.40706325,3.331246562,1.33068776,2.017512781,1.275722404,2.000178927),
}

x = np.arange(len(metrics))
width = 0.20
multiplier = -0.5

for attribute, measurement in y.items():
  offset = width * multiplier
  rects = axs[1].bar(x + offset, measurement, width, label=attribute)
  last_bar = rects[-1]
  height = last_bar.get_height()
  xpos = last_bar.get_x() + last_bar.get_width() / 2

  # Manually annotate just the last bar
  axs[1].text(xpos, height, f"{height:.2f}", ha='center', va='bottom', fontsize=8)
  multiplier += 1

axs[1].set_ylabel('Error (%)', fontsize=12)
# axs[0].set_title('DRAM read/write access')
axs[1].set_xticks(x + width, metrics, rotation=20)
axs[1].grid('grid', linestyle='dashed', color='gray', zorder=0, axis='y')
axs[1].set_axisbelow(True)
axs[1].legend(loc='upper left', ncols=5)
axs[1].set_ylim(0, 5)

plt.tight_layout()
plt.savefig("error_sweep.pdf")