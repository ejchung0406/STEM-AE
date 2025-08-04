import csv
import numpy as np
import matplotlib.pyplot as plt
import io

ncu_metrics = [
  "l1tex__t_sector_hit_rate.pct",  # L1 Cache Hit Rate
  "lts__t_sector_op_read_hit_rate.pct",  # L2 Cache Hit Rate (Read Access Only, all Writes are Hits)

  # PKA and Sieve
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

  # Execution Time
  "gpu__cycles_elapsed.avg", # Average GPU Clocks
]

val_metrics = [
  "dram__bytes_read.sum", # dram read bytes
  "dram__bytes_write.sum", # dram write bytes

  "smsp__inst_executed_op_local_ld.sum", # local loads
  "smsp__inst_executed_op_local_st.sum", # local stores
  "smsp__inst_executed_op_shared_ld.sum", # shared loads
  "smsp__inst_executed_op_shared_st.sum", # shared stores
  "smsp__inst_executed_op_global_ld.sum", # global loads
  "smsp__inst_executed_op_global_st.sum", # global stores

  "l1tex__t_sector_hit_rate.pct", # l1 hit
  "lts__t_sector_op_read_hit_rate.pct", # l2 read hit

  "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum", # 16bit FP instrs
  "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum", # 32bit FP instrs
  "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum", # 64bit FP instrs
  "smsp__sass_thread_inst_executed_op_integer_pred_on.sum", # integer instrs
  
  "smsp__thread_inst_executed_per_inst_executed.ratio", # warp execution efficiency
  "smsp__sass_average_branch_targets_threads_uniform.pct", # branch efficiency
]

def read_csv(filename: str) -> tuple[list, list]:
  data = []
  with open(filename, 'r') as file:
    line = next(file)
    # TODO: Fix Hardcoded values
    while '"ID","Process ID","Process Name"' not in line and \
          'API Start (ns),API Dur (ns),Queue Start (ns)' not in line and \
          'Time (%),Total Time (ns),Instances' not in line:
      line = next(file)

    column_names = line.replace('"', '').strip().split(',')

    # Read the rest of the file as the 2D table
    csv_reader = csv.reader(file)
    for row in csv_reader:
      if not ''.join(row).strip():  # Check if the row, when joined and stripped, is empty
        continue  # Skip appending empty rows
      data.append(row)

  return column_names, data

def exe_time_map(cols:list, data:dict[str, list[str]]) -> dict[str, list[tuple[int, str]]]:
  ret_map = {}
  for kernel_id in map(int, data["ID"]):
    # key = "x".join(data["Block Size"][kernel_id].strip('()').replace(' ', '').split(',')) + "/" \
    #     + "x".join(data["Grid Size"][kernel_id].strip('()').replace(' ', '').split(',')) + "/" \
    #     + data["Kernel Name"][kernel_id].split("(")[0].strip()
    key = data["Kernel Name"][kernel_id].split("(")[0].strip()
    if key not in ret_map:
      ret_map[key] = []
    ret_map[key].append((kernel_id, data["gpu__cycles_elapsed.avg"][kernel_id]))

  return ret_map

def dur_map(cols:list, data:dict[str, list[str]]) -> dict[str, list[tuple[int, str]]]:
  ret_map = {}
  for kernel_id, row in enumerate(data):
      # key = "x".join(row[-3].split()) + "/" + "x".join(row[-2].split()) + "/" + row[-1].split("<")[0].strip()
      key = row[cols.index("Kernel Name")].split("(")[0].strip()
      if key not in ret_map:
          ret_map[key] = []
      ret_map[key].append((kernel_id, row[cols.index("Kernel Dur (ns)")]))

  return ret_map

def read_ncu_csv(path_to_csv:str) -> tuple[list[str], dict[str, list[str]]]:
  cols, data = read_csv(path_to_csv)

  metric_map = {}
  metric_names = []

  metric_map["ID"] = []
  metric_map["Kernel Name"] = []
  metric_map["Block Size"] = []
  metric_map["Grid Size"] = []

  for idx, row in enumerate(data):
    # Collect unique metric names
    metric_name = row[cols.index("Metric Name")]
    if metric_name not in metric_names:
      metric_names.append(metric_name)
    if metric_name not in metric_map:
      metric_map[metric_name] = []

    if idx % len(metric_names) == 1:
      metric_map["ID"].append(row[cols.index("ID")])
      metric_map["Kernel Name"].append(row[cols.index("Kernel Name")])
      metric_map["Block Size"].append(row[cols.index("Block Size")])
      metric_map["Grid Size"].append(row[cols.index("Grid Size")])
    
    metric_value = row[cols.index("Metric Value")]
    metric_map[metric_name].append(metric_value.replace(",", ""))

  metric_names.extend(["ID", "Kernel Name", "Block Size", "Grid Size"])
  return metric_names, metric_map

def save_results_to_csv(results: list[tuple], filename: str) -> None:
  with open(filename, 'w') as file:
    file.write("Iteration,Sampling Method,Suite,Name,Subdir,Speedup,Prediction Error\n")
    for result in results:
      file.write(f"{result[0]},{result[1]},{result[2]},{result[3]},{result[4]},{result[5]},{result[6]}\n")
  return

def dist2(x, y):
  return np.sum((x - y) ** 2)

def regkmeans(X, lambda_arg, max_iter=100, tol=1e-4):
  """
  Optimized Regularized K-means clustering algorithm in Python.
  
  Parameters:
  X : numpy.ndarray
      Data matrix, each row is an observation (n_samples, n_features).
  lambda_ : float
      Regularization parameter.
  max_iter : int, optional (default=100)
      Maximum number of iterations.
  tol : float, optional (default=1e-4)
      Tolerance to declare convergence.
  
  Returns:
  idx : numpy.ndarray
      Cluster label of each data point (n_samples,).
  centers : numpy.ndarray
      Final centroids of each cluster (n_clusters, n_features).
  cluster_sizes : numpy.ndarray
      Number of elements in each cluster (n_clusters,).
  """
  n_samples, n_features = X.shape

  # Normalize X
  if len(X) != 1: X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
  else: X_norm = np.ones_like(X)
  
  # Initialize: every datum in one cluster
  idx = np.zeros(n_samples, dtype=int)
  lambda_ = lambda_arg

  for iteration in range(max_iter):
    old_idx = idx.copy()
    # lambda_ = lambda_ * 1.015

    unique_labels, inverse = np.unique(idx, return_inverse=True)
    n_clusters = len(unique_labels)
    centers = np.array([X_norm[idx == label].mean(axis=0) for label in unique_labels])
    cluster_sizes = np.array([np.sum(idx == label) for label in unique_labels])

    # n_clusters = np.max(idx) + 1
    # centers = np.array([X_norm[idx == label].mean(axis=0) if np.any(idx == label) else np.zeros_like(X_norm[0]) for label in range(n_clusters)])
    # cluster_sizes = np.array([np.sum(idx == label) for label in range(n_clusters)])

    # print(f"Iteration {iteration + 1}: {n_clusters} clusters")
    # print(f"Cluster sizes: {cluster_sizes}")
    # print(f"Unique labels: {np.unique(idx)}")
    
    for datum in range(n_samples):
      deltaE = np.zeros((n_clusters + 1), dtype=float)

      i = inverse[datum]

      for j in range(n_clusters):
        if cluster_sizes[i] > 1 and i != j:
          deltaE[j] = lambda_  * (1 / (cluster_sizes[i] * (cluster_sizes[i] - 1)) - 1 / (cluster_sizes[j] * (cluster_sizes[j] + 1))) \
                        + (dist2(X_norm[datum], centers[j]) * cluster_sizes[j] / (cluster_sizes[j] + 1)) \
                        - (dist2(X_norm[datum], centers[i]) * cluster_sizes[i] / (cluster_sizes[i] - 1))
        else:
          deltaE[j] = -lambda_ * (1 + 1 / (cluster_sizes[j] * (cluster_sizes[j] + 1))) \
                        + (dist2(X_norm[datum], centers[j]) * cluster_sizes[j] / (cluster_sizes[j] + 1))

      # When j = n_clusters (new cluster)
      if cluster_sizes[i] > 1:
        deltaE[n_clusters] = lambda_ * (1 / (cluster_sizes[i] * (cluster_sizes[i] - 1)) + 1) \
                              - (dist2(X_norm[datum], centers[i]) * cluster_sizes[i] / (cluster_sizes[i] - 1))
        
      # print(deltaE)

      # Update cluster assignment
      if np.min(deltaE) < 0:
        idx[datum] = np.argmin(deltaE)
    
    # Check for convergence
    if np.all(idx == old_idx) or iteration == max_iter - 1:
      break
  
  # Final update of cluster information
  unique_labels, counts = np.unique(idx, return_counts=True)
  centers = np.array([X[idx == label].mean(axis=0) for label in unique_labels])
  
  return idx, centers, counts

def regkmeans_debug():
  # Example usage
  np.random.seed(42)  # for reproducibility
  X = np.random.rand(1000, 2)  # Random 2D dataset
  lambda_ = 0.1

  idx, centers, stdevs, cluster_sizes = regkmeans(X, lambda_arg)

  print(f"Number of clusters: {len(cluster_sizes)}")
  print("Cluster sizes:", cluster_sizes)
  print("Cluster centers:\n", centers)

  # Visualization (uncomment if matplotlib is available)
  plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='viridis')
  plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
  for center, stdev in zip(centers, stdevs):
    ellipse = plt.matplotlib.patches.Ellipse(center, 2*stdev[0], 2*stdev[1], edgecolor='red', facecolor='none', linestyle='--')
    plt.gca().add_patch(ellipse)
  plt.title('Regularized K-means Clustering')
  plt.savefig('regkmeans.png')

if __name__ == "__main__":
  regkmeans_debug()