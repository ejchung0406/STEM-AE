import os, importlib
import time

from suites import *
from sampling_methods import modules as sampling_modules
from sampling_methods.utils import ncu_metrics, save_results_to_csv

def kernel_sample(suite_name: str, use_nsys_only: bool=False, sampling_methods: list[str] = None, num_iter: int = 10, verbose: bool = False) -> list[tuple]:
  ret = []
  try:
    suite_module = importlib.import_module(f'suites.{suite_name}')
    for sampling_module in sampling_modules:
      if sampling_methods == None or sampling_module.__name__.split('.')[-1] in sampling_methods:
        start_time = time.time()
        ret += sampling_module.kernel_sample(suite_module, use_nsys_only=use_nsys_only, num_iter=num_iter, verbose=verbose)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[SAMPLING] {sampling_module.__name__.split('.')[-1]} elapsed time: {elapsed_time}")
  except ModuleNotFoundError:
    print(f"Error: Module '{suite_name}' not found in the 'suites' package.")
  return ret

def main() -> None:
  os.makedirs("results", exist_ok=True)

  results = kernel_sample("rodinia", sampling_methods = ["random_sampling", "pka", "sieve", "photon", "stem"], num_iter=1)
  save_results_to_csv(results, "results/rodinia.csv")

  results = kernel_sample("casio", sampling_methods = ["random_sampling", "pka", "sieve", "photon", "stem"], num_iter=1)
  save_results_to_csv(results, "results/casio.csv")

  results = kernel_sample("hugging", sampling_methods = ["random_sampling", "stem"], num_iter=1)
  save_results_to_csv(results, "results/hugging.csv")

  return

if __name__ == '__main__':
  main()