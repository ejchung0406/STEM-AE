from types import ModuleType
from sklearn.neighbors import KernelDensity
import sampling_methods.utils as utils 
import numpy as np
from scipy.signal import argrelextrema
import csv
import pickle
import random


sieve_metrics = [
  "ID",
  "Kernel Name",
  "Block Size",
  "Grid Size",
  "smsp__inst_executed.sum", # Total Instructions Executed
]

def tuple_to_int(tuple: str) -> int:
  """ input: string '(a, b, c)' 
      output: int a*b*c """
  tuple_str = tuple.strip('()').replace(' ', '')
  
  elements = tuple_str.split(',')
  
  product = 1
  for elem in elements:
      product *= int(elem)
  
  return product

def my_cv(num_instrs: list[int]) -> float:
  if len(num_instrs) < 2: return 0.0
  mean = np.mean(num_instrs)
  std = np.std(num_instrs)
  return std / mean if mean != 0 else 0.0

def my_kde(num_instrs: list[int]) -> list[int]:
  X = np.array(num_instrs).reshape(-1, 1)
  kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(X)
  scores = kde.score_samples(X)  # Evaluate KDE only at original points
  local_minima_indices = argrelextrema(scores, np.less_equal)[0]
  return list(local_minima_indices)

def run_sieve(suite_name:str, name:str, subdir:str, use_nsys_only: bool, print_samples: bool, export_for_figures: bool=False) -> tuple[float, float]:
  if use_nsys_only:
    return 0, 0

  path_to_ncu_csv = f"results_ncu-example/{name}/{subdir}/ncu_{name}_flush.csv"
  metric_names, metric_map = utils.read_ncu_csv(path_to_ncu_csv)
  data_flush = [[x.replace(",", "") for x in metric_map[metric]] for metric in sieve_metrics]
  # change dimension 0 and 1 of data
  data_flush = list(map(list, zip(*data_flush)))

  path_to_nsys_csv = f"results_ncu-example/{name}/{subdir}/nsys_{name}.csv"
  cols_nsys, data_nsys = utils.read_csv(path_to_nsys_csv)
  total_exe_time = sum([int(row[cols_nsys.index("Kernel Dur (ns)")]) for row in data_nsys])

  new_data = {}
  for row in data_flush:
    key = row[sieve_metrics.index("Kernel Name")]
    if key not in new_data:
      new_data[key] = []
    # new_row = (ID, Block Size * Grid Size, Total Instructions Executed)
    new_row = (int(row[sieve_metrics.index("ID")]), 
               tuple_to_int(row[sieve_metrics.index("Block Size")]) * \
                  tuple_to_int(row[sieve_metrics.index("Grid Size")]),
               int(row[sieve_metrics.index("smsp__inst_executed.sum")]))
    new_data[key].append(new_row)

  tier1 = {}
  tier2 = {}
  tier3 = {}

  for key, value in new_data.items():
    cv = my_cv([v[2] for v in value])
    if cv == 0:
      tier1[key] = [v[0] for v in value]
    elif cv < 0.4:
      tier2[key] = [v[0] for v in value]
    else:
      # tier3[key] = [v[0] for v in value]
      value.sort(key=lambda x: x[2])
      num_instrs = [v[2] for v in value]
      local_minima_idx = my_kde(num_instrs)
      local_minima_idx = [0] + local_minima_idx + [len(num_instrs)] 
      # cluster based on local minima boundaries and assign to tier3 with different keys
      for idx in range(len(local_minima_idx) - 1):
        new_num_instrs = num_instrs[local_minima_idx[idx]:local_minima_idx[idx+1]]
        cv_new = my_cv(new_num_instrs)
        if cv_new > 0.5:
          new_local_minima_idx = my_kde(new_num_instrs)
          new_local_minima_idx = [0] + new_local_minima_idx + [len(new_num_instrs)]
          for new_idx in range(len(new_local_minima_idx) - 1):
            tier3[key + "/" + str(idx) + "/" + str(new_idx)] = [v[0] for v in value[local_minima_idx[idx]:local_minima_idx[idx+1]][new_local_minima_idx[new_idx]:new_local_minima_idx[new_idx+1]]]
        else:
          tier3[key + "/" + str(idx)] = [v[0] for v in value[local_minima_idx[idx]:local_minima_idx[idx+1]]]

  # Sample from each tier groups
  total_sampled_exe_time = 0
  predicted_total_exe_time = 0
  total_samples = []

  for key, value in tier1.items():
    if len(value) == 0:
      continue
    # Sample first kernel
    # print(f"tier 1 - sampled kernel id: {value[0]}")
    sample = value[0]
    # sample = random.choice(value)
    total_sampled_exe_time += int(data_nsys[sample][cols_nsys.index("Kernel Dur (ns)")])
    predicted_total_exe_time += int(data_nsys[sample][cols_nsys.index("Kernel Dur (ns)")]) * len(value)
    total_samples.append((sample, len(value)))

  for key, value in tier2.items():
    if len(value) == 0:
      continue
    value_counts = {}
    for v in value:
      # num_of_threads is new_data[key][??][1] where new_data[key][??][0] == v
      num_of_threads = [vv[1] for vv in new_data[key] if vv[0] == v][0]
      if num_of_threads not in value_counts:
        value_counts[num_of_threads] = 0
      value_counts[num_of_threads] += 1
    
    dominant_value = max(value_counts, key=value_counts.get)
    # first kernel id with the most dominant num of threads
    sampled_kernel_id = min([v[0] for v in new_data[key] if v[1] == dominant_value], key=lambda x: x)
    # sampled_kernel_id = random.choice([v[0] for v in new_data[key] if v[1] == dominant_value])
  
    total_sampled_exe_time += int(data_nsys[sampled_kernel_id][cols_nsys.index("Kernel Dur (ns)")])
    predicted_total_exe_time += int(data_nsys[sampled_kernel_id][cols_nsys.index("Kernel Dur (ns)")]) * len(value)
    total_samples.append((sampled_kernel_id, len(value)))

  for key, value in tier3.items():
    if len(value) == 0:
      continue
    # print(f"tier 3 - sampled kernel id: {value[0]}")
    sample_id = value[0]
    # sample_id = random.choice(value)
    total_sampled_exe_time += int(data_nsys[sample_id][cols_nsys.index("Kernel Dur (ns)")])
    predicted_total_exe_time += int(data_nsys[sample_id][cols_nsys.index("Kernel Dur (ns)")]) * len(value)
    total_samples.append((sample_id, len(value)))

  if export_for_figures:
    with open("stem-figures/sieve.csv", mode="w", newline="") as file:
      rows_to_export = []
      for key, value in tier1.items():
        subset = [data_nsys[idx][cols_nsys.index("Kernel Dur (ns)")] for idx in value]
        rows_to_export.append([f"Tier 1 - {key}"] + subset)
      for key, value in tier2.items():
        subset = [data_nsys[idx][cols_nsys.index("Kernel Dur (ns)")] for idx in value]
        rows_to_export.append([f"Tier 2 - {key}"] + subset)
      for key, value in tier3.items():
        subset = [data_nsys[idx][cols_nsys.index("Kernel Dur (ns)")] for idx in value]
        rows_to_export.append([f"Tier 3 - {key}"] + subset)
      writer = csv.writer(file)
      writer.writerows(rows_to_export)

  if print_samples:
    total_samples.sort()
    print(f"Kernel sample IDs: {total_samples}")
    with open(f"{name}-sieve.pkl", "wb") as f:
      pickle.dump(total_samples, f)

  speedup = total_exe_time / total_sampled_exe_time
  prediction_error = abs(total_exe_time - predicted_total_exe_time) / total_exe_time
  print(f"Method: Sieve")
  print(f"Suite: {suite_name}, Name: {name}, Subdir: {subdir}")
  print(f"Speedup: {speedup}") 
  print(f"Error: {prediction_error * 100:.5f}%")
  
  return speedup, prediction_error

def kernel_sample(suite_module: ModuleType, use_nsys_only: bool, num_iter: int, verbose: bool) -> list[tuple]:
  ret = []
  for name in suite_module.names:
    for subdir in list(suite_module.subdirs[name])[-1:]:
      for iter in range(num_iter):
        s, p = run_sieve(suite_name = suite_module.__name__, name = name, subdir = subdir, use_nsys_only = use_nsys_only, print_samples = verbose)
        ret.append((f"iter {iter}", "Sieve", suite_module.__name__, name, subdir, s, p))
  return ret 

def debug_sieve():
  suite_name = "casio"
  name = "dlrm-infer"
  subdir = "default"

  speedup, prediction_error = run_sieve(
    suite_name=suite_name,
    name=name,
    subdir=subdir,
    use_nsys_only=False,
    export_for_figures=True,
    print_samples=False,
  )
  return

if __name__ == "__main__":
  debug_sieve()