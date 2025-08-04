import csv, pickle
import numpy as np
from types import ModuleType
import sampling_methods.utils as utils

def parse_photon(path_to_csv: str) -> np.ndarray:
  with open(path_to_csv, 'r') as f:
    reader = csv.reader(f)
    # Read the header and kernel information
    next(reader)
    # Kernel information example: Kernel0,3,262144,_Z22bpnn_layerforward_CUDAPfS_S_S_ii
    
    gpu_bbvs = []
    num_warps = []
    try:
      while True:
        kernel_info = next(reader)  # Get the next kernel entry
        # num_lines = math.ceil(int(kernel_info[2]) / 100)  # Compute number of lines to read

        # data = [list(map(int, row[:-1])) for row in islice(reader, num_lines)]

        gpu_bbv = np.array(next(reader)[:-1], dtype=int)
        # gpu_bbv = np.mean(gpu_bbv, axis=0)
        gpu_bbvs.append(gpu_bbv)
        num_warps.append(int(kernel_info[2]))
        # print(f"{kernel_info[0]}, GPU BBV: {gpu_bbv}")
    except StopIteration:
      pass  

    return gpu_bbvs, num_warps
  
def run_photon(suite_name: str, name: str, subdir: str, use_nsys_only: bool=False, print_samples: bool=False) -> tuple[float, float, float]:
  path_to_nsys_csv = f"results_ncu-example/{name}/{subdir}/nsys_{name}.csv"
  cols, data = utils.read_csv(path_to_nsys_csv)
  total_exe_time = sum([int(row[cols.index("Kernel Dur (ns)")]) for row in data])

  # exe_map[key = gridsize + blocksize + kernel name] = [kernel id, kernel exetime]
  exe_map = utils.dur_map(cols, data)

  path_to_photon_csv = f"results_photon-example/{name}/{subdir}/bbv.csv"
  gpu_bbvs, num_warps = parse_photon(path_to_photon_csv)
  max_length = max(len(bbv) for bbv in gpu_bbvs)
  gpu_bbvs = [np.pad(bbv, (0, max_length - len(bbv)), mode='constant') for bbv in gpu_bbvs]
  gpu_bbvs_np = np.array(gpu_bbvs)

  threshold = 0.05  # 5% error threshold
  
  total_sampled_exe_time = 0
  predicted_total_exe_time = 0
  n_sampled_kernels = 0
  total_samples = []
  total_sample_weight = {}

  print(f"Size: {gpu_bbvs_np.shape}")

  distr = {}

  for i in range(len(gpu_bbvs_np)):
    current_bbv = gpu_bbvs_np[i]
    current_warps = num_warps[i]
    candidates = []
    
    min_error_outside_threshold = 1
    # Check all preceding rows
    for j in total_samples:
      prev_bbv = gpu_bbvs_np[j[0]]
      error = np.linalg.norm(current_bbv - prev_bbv) / (np.linalg.norm(current_bbv) + 1e-6)
      
      if error < threshold:
        candidates.append((j[0], error, num_warps[j[0]]))
      else:
        if error < min_error_outside_threshold:
          min_error_outside_threshold = error
    
    if candidates:
      # Find candidate with most similar num_warps
      best_candidate = min(candidates, key=lambda x: abs(x[2] - current_warps))
      # print(f"Row {i}: Most similar BBV is row {best_candidate[0]} with error {best_candidate[1]:.8f}. Exetime: {int(data[i][cols.index('Kernel Dur (ns)')])} ns")
      predicted_total_exe_time += int(data[best_candidate[0]][cols.index("Kernel Dur (ns)")])
      # distr[best_candidate[0]].append(int(data[i][cols.index("Kernel Dur (ns)")]))
      
      for idx, sample in enumerate(total_samples):
        if sample[0] == best_candidate[0]:
          total_samples[idx] = (sample[0], sample[1] + 1)
          break

    else:
      # print(f"Row {i}: No similar BBVs found below threshold. Min error: {min_error_outside_threshold}, Exetime: {int(data[i][cols.index('Kernel Dur (ns)')])} ns")
      n_sampled_kernels += 1
      total_sampled_exe_time += int(data[i][cols.index("Kernel Dur (ns)")])
      predicted_total_exe_time += int(data[i][cols.index("Kernel Dur (ns)")])
      total_samples.append((i, 1))
      # distr[i] = [int(data[i][cols.index("Kernel Dur (ns)")])]

  speedup = total_exe_time / total_sampled_exe_time
  prediction_error = abs(predicted_total_exe_time - total_exe_time) / total_exe_time

  print(f"Method: Photon")
  print(f"Original number of kernels: {len(gpu_bbvs_np)}")
  print(f"Total number of sampled kernels: {n_sampled_kernels}")
  print(f"Suite: {suite_name}, Name: {name}, Subdir: {subdir}")
  print(f"Speedup: {speedup}") 
  print(f"Error: {prediction_error * 100:.5f}%")
  if print_samples:
    total_samples.sort()
    print(f"Kernel sample IDs: {total_samples}")
    with open(f"/fast_data/echung67/sandbox/llm-for-macsim/{name}-photon.pkl", "wb") as f:
      pickle.dump(total_samples, f)
  print("==============================")

  # for key, value in distr.items():
  #   print(f"Kernel {key}: {value}")

  return speedup, prediction_error

def kernel_sample(suite_module: ModuleType, use_nsys_only:bool, num_iter: int, verbose: bool) -> list[tuple]:
  ret = []
  for name in suite_module.names:
    for subdir in list(suite_module.subdirs[name])[-1:]:
      for iter in range(num_iter):
        s, p = run_photon(suite_name = suite_module.__name__, name = name, subdir = subdir, use_nsys_only = use_nsys_only, print_samples=verbose)
        ret.append((f"iter {iter}", "Photon", suite_module.__name__, name, subdir, s, p))

  return ret

### Debugging

def debug_photon():
  suite_name = "casio"
  name = "dlrm-infer"
  subdir = "default"

  speedup, prediction_error = run_photon(
    suite_name=suite_name,
    name=name,
    subdir=subdir,
    print_samples=True
  )
  return

if __name__ == "__main__":
  debug_photon()

