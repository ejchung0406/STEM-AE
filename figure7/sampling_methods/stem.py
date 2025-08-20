import math, random, statistics
import sampling_methods.utils as utils
import numpy as np, pickle
import pickle
import importlib
import os

from sklearn.cluster import KMeans
from types import ModuleType

def kmeans_1d(l_list: list[tuple[int, str]]) -> dict[int, list[tuple[int, str]]]:
  # Reshape the list to make it suitable for clustering
  data = [[int(x[1])] for x in l_list]

  # Normalize data to fit in the range [0, 1]
  data = np.array(data)
  data = (data - data.min()) / (data.max() - data.min() + 1e-6)

  # Initialize the KMeans model with k=2
  kmeans = KMeans(n_clusters=2)

  # Fit the model to the data
  kmeans.fit(data)

  # Get the cluster labels for each data point
  cluster_labels = kmeans.labels_

  # Group data points based on their cluster labels
  clustered_data = {}
  for i, label in enumerate(cluster_labels):
    if label not in clustered_data:
      clustered_data[label] = []
    clustered_data[label].append(l_list[i])

  return clustered_data

def regkmeans_1d(l_list: list[tuple[int, str]], lambda_: float = 0.1) -> tuple[dict[int, list[tuple[int, str]]], int]:
  # Reshape the list to make it suitable for clustering
  data = [[int(x[1])] for x in l_list]

  idx, _, counts = utils.regkmeans(np.array(data), lambda_arg=lambda_)

  # Group data points based on their cluster labels
  clustered_data = {}
  for i, label in enumerate(idx):
    if label not in clustered_data:
      clustered_data[label] = []
    clustered_data[label].append(l_list[i])

  return clustered_data, len(counts)

def cluster_recursive(key: str, values: list, c: float,
                      kernel_info: dict[str, tuple[int, list[int]]], simtime_old_: float, n_samples_old_: int) -> None:
  
  if len(values) == 1:
    kernel_info[key] = 1, [values[0][0]]
    return
  
  if simtime_old_ == -1.0:
    avg_old = statistics.mean(list(map(float, [v[1] for v in values])))
    stdev_old = statistics.pstdev(list(map(float, [v[1] for v in values])))
    n_samples_old = math.ceil((c * stdev_old / avg_old) ** 2)
    simtime_old = n_samples_old * avg_old
  else:
    simtime_old = simtime_old_
    n_samples_old = n_samples_old_

  if simtime_old == 0:
    kernel_info[key] = n_samples_old, [v[0] for v in values]
    return

  cluster = kmeans_1d(values)

  sqrtaibi = 0
  nimui = 0
  n_samples_new = []
  n_samples_new_map = {}
  simtime_new_map = {}

  for kkey, value in cluster.items():
    avg_i = statistics.mean(list(map(float, [v[1] for v in value])))
    stdev_i = statistics.pstdev(list(map(float, [v[1] for v in value]))) if len(value) > 1 else 0
    sqrtaibi += len(value) * stdev_i * math.sqrt(avg_i)
    nimui += len(value) * avg_i
    n_samples_new.append(len(value)*stdev_i/math.sqrt(avg_i))
    n_samples_new_map[kkey] = len(value) * stdev_i / math.sqrt(avg_i)
  
  n_samples_new = [math.ceil(sqrtaibi / ((1/c * nimui) ** 2) * n_i) for n_i in n_samples_new]
  for kkey, value in n_samples_new_map.items():
    n_samples_new_map[kkey] = math.ceil(sqrtaibi / ((1/c * nimui) ** 2) * value)

  simtime_new = 0
  for kkey, value in cluster.items():
    simtime_new_map[kkey] = n_samples_new_map[kkey] * statistics.mean(list(map(float, [v[1] for v in value])))
    simtime_new += n_samples_new_map[kkey] * statistics.mean(list(map(float, [v[1] for v in value])))

  # print(f"key: {key}, simtime_old: {simtime_old}, simtime_new: {simtime_new}, n_samples_old: {n_samples_old}, n_samples_new: {n_samples_new}")

  if simtime_new > simtime_old:
    kernel_info[key] = n_samples_old, [v[0] for v in values]
  else:
    for kkey, value in cluster.items():
      new_key = key + str(kkey)
      cluster_recursive(new_key, value, c, kernel_info, simtime_old_=simtime_new_map[kkey], n_samples_old_=n_samples_new_map[kkey])

def parse_stem(suite_name: str, name: str, subdir: str, error_bound: float, 
                  use_nsys_only: bool, print_samples: bool=False) -> tuple[float, float]:
  c = 1.96 / error_bound
  path_to_nsys_csv = f"results_ncu-example/{name}/{subdir}/nsys_{name}.csv"
  cols, data = utils.read_csv(path_to_nsys_csv)
  total_exe_time = sum([int(row[cols.index("Kernel Dur (ns)")]) for row in data])

  # exe_map[key = gridsize + blocksize + kernel name] = [kernel id, kernel exetime]
  exe_map = utils.dur_map(cols, data)

  # kernel_info[key = gridsize + blocksize + kernel name + cluster_ids] = n_samples, [kernel ids]
  kernel_info = {}

  for key, value in exe_map.items():
    exetimes = list(map(float, [v[1] for v in value]))
    cluster_recursive(key, value, c, kernel_info, simtime_old_=-1.0, n_samples_old_=-1)

  total_n_sampled_kernels = 0
  for key, value in kernel_info.items():
    total_n_sampled_kernels += math.ceil(value[0])

  for iter in range(1):
    total_sampled_exe_time = 0
    predicted_total_exe_time = 0
    n_sampled_kernels = 0
    total_samples = []

    for key, value in kernel_info.items():
      # from a list value[1], randomly pick math.ceil(value[0]) elements (with duplicates)
      m_min = min(math.ceil(value[0]), len(value[1]))
      n_sampled_kernels += m_min
      sample_ids = random.sample(value[1], k=m_min)
      sample_ids = list(map(int, sample_ids))
      for sample_id in sample_ids:
        total_sampled_exe_time += int(data[sample_id][cols.index("Kernel Dur (ns)")])
        predicted_total_exe_time += int(data[sample_id][cols.index("Kernel Dur (ns)")]) * len(value[1]) / m_min
      
        weight = float(len(value[1]) / m_min)
        total_samples.append((sample_id, weight))

    speedup = total_exe_time / total_sampled_exe_time
    prediction_error = abs(predicted_total_exe_time - total_exe_time) / total_exe_time

    print(f"Method: STEM")
    print(f"Original number of kernels: {len(data)}")
    print(f"Total number of sampled kernels: {n_sampled_kernels}")
    print(f"Suite: {suite_name}, Name: {name}, Subdir: {subdir}")
    print(f"Error bound: {error_bound * 100:.0f}%")
    print(f"Speedup: {speedup}") 
    print(f"Error: {prediction_error * 100:.5f}%")
    if print_samples:
      total_samples.sort()
      print(f"Kernel sample IDs: {total_samples}")
      with open(f"{name}-stem.pkl", "wb") as f:
        pickle.dump(total_samples, f)
    print("==============================")

  return speedup, prediction_error

def kernel_sample(suite_module: ModuleType, use_nsys_only: bool, num_iter: int, verbose: bool) -> list[tuple]:
  ret = []
  for name in suite_module.names:
    for subdir in list(suite_module.subdirs[name])[-1:]:
      for iter in range(num_iter):
        s, p = parse_stem(suite_name = suite_module.__name__, name = name, subdir = subdir, \
                      error_bound = 0.05, use_nsys_only = use_nsys_only, print_samples=verbose) # from paper: error_bound = 5%, n_th = 50, n_min = 30
        ret.append((f"iter {iter}", "STEM", suite_module.__name__, name, subdir, s, p))

  return ret

### Debugging

def debug_stem():
  suite_name = "rodinia"
  name = "srad_v2"
  subdir = "10"
  s, p = parse_stem(
    suite_name=suite_name,
    name=name,
    subdir=subdir,
    error_bound=0.01,
    use_nsys_only=True,
    print_samples=True
  )
  print(f"Speedup: {s}, Prediction Error: {p}")

  return

def kernel_sample_sweep(suite_module: ModuleType, use_nsys_only: bool, num_iter: int) -> list[tuple]:
  ret = []
  for e in [0.03, 0.05, 0.1, 0.25]:
    for name in suite_module.names:
      for subdir in list(suite_module.subdirs[name])[-1:]:
        for iter in range(num_iter):
          s, p = parse_stem(suite_name = suite_module.__name__, name = name, subdir = subdir, \
                        error_bound = e, use_nsys_only = use_nsys_only) # from paper: error_bound = e
          ret.append((f"iter {iter}", "STEM", suite_module.__name__, name, subdir, s, p))

  return ret

def parse_stem_diff_hw(suite_name: str, name: str, subdir: str, error_bound: float, 
                  print_samples: bool=False) -> tuple[float, float, float]:
  c = 1.96 / error_bound
  path_to_nsys_csv = f"results_diff_hws/results/h100/{name}/h100.csv"
  cols, data = utils.read_csv(path_to_nsys_csv)
  path_to_new_nsys_csv = f"results_diff_hws/results/h200/{name}/h200.csv"
  _, data_new = utils.read_csv(path_to_new_nsys_csv)
  total_exe_time = sum([int(row[cols.index("Kernel Dur (ns)")]) for row in data])

  # exe_map[key = gridsize + blocksize + kernel name] = [kernel id, kernel exetime]
  exe_map = utils.dur_map(cols, data)

  # kernel_info[key = gridsize + blocksize + kernel name + cluster_ids] = n_samples, [kernel ids]
  kernel_info = {}

  for key, value in exe_map.items():
    exetimes = list(map(float, [v[1] for v in value]))
    cluster_recursive(key, value, c, kernel_info, simtime_old_=-1.0, n_samples_old_=-1)

  # print(f"Total number of Clusters: {len(kernel_info)}")
  # for key, value in kernel_info.items():
  #   print(f"Cluster: {key}, n_samples: {value[0]}")

  total_n_sampled_kernels = 0
  for key, value in kernel_info.items():
    total_n_sampled_kernels += math.ceil(value[0])

  for iter in range(1):
    total_sampled_exe_time = 0
    predicted_total_exe_time = 0
    n_sampled_kernels = 0
    total_samples = []

    for key, value in kernel_info.items():
      # from a list value[1], randomly pick math.ceil(value[0]) elements (with duplicates)
      m_min = min(math.ceil(value[0]), len(value[1]))
      n_sampled_kernels += m_min
      sample_ids = random.sample(value[1], k=m_min)
      sample_ids = list(map(int, sample_ids))
      for sample_id in sample_ids:
        total_sampled_exe_time += int(data_new[sample_id][cols.index("Kernel Dur (ns)")])
        predicted_total_exe_time += int(data_new[sample_id][cols.index("Kernel Dur (ns)")]) * len(value[1]) / m_min
      
        weight = float(len(value[1]) / m_min)
        total_samples.append((sample_id, weight))

    speedup = total_exe_time / total_sampled_exe_time
    prediction_error = abs(predicted_total_exe_time - total_exe_time) / total_exe_time

    print(f"Method: STEM")
    print(f"Original number of kernels: {len(data)}")
    print(f"Total number of sampled kernels: {n_sampled_kernels}")
    print(f"Suite: {suite_name}, Name: {name}, Subdir: {subdir}")
    print(f"Error bound: {error_bound * 100:.0f}%")
    print(f"Speedup: {speedup}") 
    print(f"Error: {prediction_error * 100:.5f}%")
    if print_samples:
      total_samples.sort()
      print(f"Kernel sample IDs: {total_samples}")
      with open(f"{name}-stem.pkl", "wb") as f:
        pickle.dump(total_samples, f)
    print("==============================")

  return speedup, prediction_error

def kernel_sample_diff_hw(suite_module: ModuleType, num_iter: int, verbose: bool) -> list[tuple]:
  ret = []
  for name in suite_module.names:
    for subdir in list(suite_module.subdirs[name])[-1:]:
      for iter in range(num_iter):
        s, p = parse_stem_diff_hw(suite_name = suite_module.__name__, name = name, subdir = subdir, \
                      error_bound = 0.05, print_samples=verbose) # from paper: error_bound = 5%, n_th = 50, n_min = 30
        ret.append((f"iter {iter}", "STEM", suite_module.__name__, name, subdir, s, p))

  return ret

if __name__ == "__main__":
  # debug_stem()
  # kernel_sample(importlib.import_module("suites.rodinia"))
  ret = kernel_sample_sweep(importlib.import_module("suites.casio"), use_nsys_only=True, num_iter=1)
  os.makedirs("../figure11/results", exist_ok=True)
  utils.save_results_to_csv(ret, "../figure11/results/stem-casio-sweep.csv")
  # kernel_sample(importlib.import_module("suites.fastertransformer"))
