from types import ModuleType
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sampling_methods.utils as utils
import numpy as np
import csv
import os
import pickle 

pka_metrics = [
  "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", # Coalesced Global Load
  "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", # Coalesced Global Store 
  "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", # Coalesced Local Load
  "smsp__inst_executed_op_global_ld.sum", # Thread Global Load
  "smsp__inst_executed_op_global_st.sum", # Thread Global Store
  "smsp__inst_executed_op_local_ld.sum", # Thread Local Load
  "smsp__inst_executed_op_shared_ld.sum", # Thread Shared Load
  "smsp__inst_executed_op_shared_st.sum", # Thread Shared Store
  "smsp__sass_inst_executed_op_global_atom.sum", # Thread Global Atomic
  "smsp__inst_executed.sum", # Total Instructions Executed (Also used in Sieve)
  "smsp__thread_inst_executed_per_inst_executed.ratio", # Divergence Efficiency
  "launch__grid_size", # Number of thread blocks
]

def run_pka(suite_name:str, name:str, subdir:str, use_nsys_only: bool, print_samples: bool, export_for_figures: bool = False) -> tuple[float, float, int]:
  if use_nsys_only:
    return 0, 0, 0

  path_to_nsys_csv = f"results_ncu-example/{name}/{subdir}/nsys_{name}.csv"
  cols_nsys, data_nsys = utils.read_csv(path_to_nsys_csv)
  total_exe_time = sum([int(row[cols_nsys.index("Kernel Dur (ns)")]) for row in data_nsys])

  path_to_ncu_csv = f"results_ncu-example/{name}/{subdir}/ncu_{name}_flush.csv"
  metric_names, metric_map = utils.read_ncu_csv(path_to_ncu_csv)
  data_np = np.array([[float(x.replace(",", "")) for x in metric_map[metric]] for metric in pka_metrics]).T

  # Perform PCA
  pca = PCA(n_components=0.9995)
  new_data = pca.fit_transform(data_np)

  # Perform KMeans clustering sweep
  min_error = float('inf')
  for i in range(1, 21):
    if i > len(new_data) or i > len(new_data[0]):
      break

    total_samples = []

    total_sampled_exe_time = 0
    predicted_total_exe_time = 0

    kmeans = KMeans(n_clusters=i)
    kmeans.fit(new_data)
    labels = kmeans.labels_

    # among all kernels in the cluster, find the one that has smallest index
    for j in range(i):
      cluster_indices = np.where(labels == j)[0]
      # first_kernel = cluster_indices[np.argmin(cluster_indices)]
      first_kernel = np.random.choice(cluster_indices)
      num_same_label_clusters = len(cluster_indices)
      total_sampled_exe_time += int(data_nsys[first_kernel][cols_nsys.index("Kernel Dur (ns)")])
      predicted_total_exe_time += int(data_nsys[first_kernel][cols_nsys.index("Kernel Dur (ns)")]) * num_same_label_clusters
      total_samples.append((first_kernel, num_same_label_clusters))

    error = abs(total_exe_time - predicted_total_exe_time) / total_exe_time
    if error < min_error:
      min_error = error
      min_speedup = total_exe_time / total_sampled_exe_time
      min_error_i = i

      if export_for_figures:
        os.makedirs("stem-figures", exist_ok=True)
        with open("stem-figures/pka.csv", mode="w", newline="") as file:
          rows_to_export = []
          for ii in range(min_error_i):
            subset = [data_nsys[idx][cols_nsys.index("Kernel Dur (ns)")] for idx in range(len(labels)) if labels[idx] == ii]
            rows_to_export.append([f"Cluster {ii}"] + subset)
          writer = csv.writer(file)
          writer.writerows(rows_to_export)

  print(f"Method: PKA")
  print(f"Suite: {suite_name}, Name: {name}, Subdir: {subdir}")
  print(f"Speedup: {min_speedup} when num of clusters = {min_error_i}") 
  print(f"Error: {min_error * 100:.5f}%")
  if print_samples:
    total_samples.sort()
    print(f"Kernel sample IDs: {total_samples}")
    with open(f"{name}-pka.pkl", "wb") as f:
      pickle.dump(total_samples, f)
  
  return min_speedup, min_error, min_error_i

def kernel_sample(suite_module: ModuleType, use_nsys_only: bool, num_iter: int, verbose: bool) -> list[tuple]:
  ret = []
  for name in suite_module.names:
    for subdir in list(suite_module.subdirs[name])[-1:]:
      for iter in range(num_iter):
        s, p, i = run_pka(suite_name = suite_module.__name__, name = name, subdir = subdir, use_nsys_only = use_nsys_only, print_samples=verbose)
        ret.append((f"iter {iter}", f"PKA-{i}", suite_module.__name__, name, subdir, s, p))
  return ret

def debug_pka():
  suite_name = "casio"
  name = "dlrm-infer"
  subdir = "default"

  speedup, prediction_error, c = run_pka(
    suite_name=suite_name,
    name=name,
    subdir=subdir,
    use_nsys_only=False,
    print_samples=False,
    export_for_figures=True,
  )
  return

if __name__ == "__main__":
  debug_pka()
  