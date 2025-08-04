from types import ModuleType
import sampling_methods.utils as utils
import random, math, importlib

# Random Kernel Sampling
# Sample (N / desired_speedup) number of kernels
def kernel_sample(suite_module: ModuleType, use_nsys_only: bool, num_iter: int, verbose: bool, desired_speedup: float = 1000.0) -> list[tuple]:

  ret = []
  for name in suite_module.names:
    for subdir in list(suite_module.subdirs[name])[-1:]:
      for iter in range(num_iter):
        path_to_nsys_csv = f"results_ncu-example/{name}/{subdir}/nsys_{name}.csv"
        cols, data = utils.read_csv(path_to_nsys_csv)
        total_exe_time = sum([int(row[cols.index("Kernel Dur (ns)")]) for row in data])

        total_sampled_exe_time = 0
        predicted_total_exe_time = 0

        for i in range(math.ceil(len(data) / desired_speedup)):
          sample_id = random.randint(0, len(data) - 1)
          total_sampled_exe_time += int(data[sample_id][cols.index("Kernel Dur (ns)")])
          predicted_total_exe_time += int(data[sample_id][cols.index("Kernel Dur (ns)")]) \
                                      * len(data) / math.ceil(len(data) / desired_speedup)

        prediction_error = abs(total_exe_time - predicted_total_exe_time) / total_exe_time
        speedup = total_exe_time / total_sampled_exe_time

        print(f"Method: Random Sampling")
        print(f"Suite: {suite_module.__name__}, Name: {name}, Subdir: {subdir}")
        print(f"Speedup: {speedup}")
        print(f"Error: {prediction_error * 100:.5f}%")

        ret.append((f"iter {iter}", "Random", suite_module.__name__, name, subdir, speedup, prediction_error))

  return ret

if __name__ == "__main__":
  suite_module = importlib.import_module(f'suites.hugging')
  results = kernel_sample(suite_module, use_nsys_only=True, num_iter=10)
  utils.save_results_to_csv(results, "results/hugging-random.csv")