import os
import subprocess
import re
import pickle
import subprocess
import csv

configs = [
  "naive",
  "cache_double",
  "cache_half",
  "core_double",
  "core_half",
]

workloads =[
  "bfs",
  "gaussian",
  "particlefilter_naive",
  "srad_v1",
]

def sample(iter=10):
  results = []

  sampling_methods = ["pka", "sieve", "photon", "stem"]

  for _ in range(iter):
    sample_ids = {}

    for method in sampling_methods:
      for work in workloads:
        with open(f"pkl/{work}-tiny-{method}.pkl", "rb") as f:
          sample_ids[f"{work}"] = pickle.load(f)

      for work in workloads:
        for config in configs:
          print(f"Sampling method: {method}, Workload: {work}, Config: {config}")

          macsim_path = f"macsim/{work}/{config}/macsim.out"

          if not os.path.exists(f"macsim/{work}/{config}/general.stat.out"):
            print(f"File {macsim_path} does not exist.")
            continue

          num_of_cycles_cum = extract_num_of_cycles(macsim_path)
          num_of_cycles = []

          num_of_cycles.append(num_of_cycles_cum[0])
          for i in range(1, len(num_of_cycles_cum)):
            num_of_cycles.append(num_of_cycles_cum[i] - num_of_cycles_cum[i-1])

          predicted_total_cycles = 0
          for id, weight in sample_ids[work]:
            predicted_total_cycles += num_of_cycles[id] * weight

          error = abs(sum(num_of_cycles) - predicted_total_cycles) / sum(num_of_cycles) 

          print("Predicted Total Cycles: ", predicted_total_cycles)
          print("Actual Total Cycles: ", sum(num_of_cycles))
          print("Error: ", error * 100, "%")

          results.append((method, work, config, predicted_total_cycles, sum(num_of_cycles), error * 100))

  # Save results to a CSV file
  with open('results.csv', 'a', newline='') as csvfile:
    fieldnames = ['Sampling method', 'Workload', 'Config', 'Predicted #cycles', 'Total #cycles', 'Error (%)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header only if the file is new
    if csvfile.tell() == 0:
      writer.writeheader()
    for method, work, config, pred, total, error in results:
      writer.writerow({'Sampling method': method, 'Workload': work, 'Config': config, 'Predicted #cycles': pred, 'Total #cycles': total, 'Error (%)': error})


def extract_num_of_cycles(filename):
  kernel_cycles = {}
  current_kernel = None
  kernel_index = None
  thread_cycles = []
  
  # Process file line by line to avoid loading the entire file into memory
  with open(filename, 'r') as file:
    for line in file:
      # Check for kernel setup
      kernel_match = re.search(r'setup_process:0\s+([^\s]+)\s+current_index:(\d+)', line)
      if kernel_match:
        # If we found a new kernel and had threads from previous kernel, record the max
        if current_kernel and thread_cycles:
          kernel_cycles[current_kernel] = max(thread_cycles)
          thread_cycles = []  # Reset for new kernel
        
        kernel_path = kernel_match.group(1)
        kernel_index = kernel_match.group(2)
        current_kernel = kernel_path.split('/')[-2]  # Extract Kernel0, Kernel1, etc.
      
      # Check for thread finished line
      thread_match = re.search(r'\*\*Core \d+ Thread \d+ Finished:\s+insts:\d+\s+cycles:(\d+)', line)
      if thread_match and current_kernel:
        cycles = int(thread_match.group(1))
        thread_cycles.append(cycles)
  
  # Add the last kernel's data if it exists
  if current_kernel and thread_cycles:
    kernel_cycles[current_kernel] = max(thread_cycles)
  
  return [cycle for _, cycle in kernel_cycles.items()]

if __name__ == "__main__":
  sample(iter=1)