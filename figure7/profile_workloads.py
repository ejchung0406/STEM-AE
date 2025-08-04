import os, importlib
from suites import *
from sampling_methods import modules as sampling_modules
from sampling_methods.utils import ncu_metrics, save_results_to_csv
import subprocess
import time

def run_nsys(suite_name: str, device_id: int, nsys_overwrite_flag: bool) -> None:
  try:
    module = importlib.import_module(f'suites.{suite_name}')
    for name in module.names:
      for arg, subdir in list(zip(module.args[name], module.subdirs[name]))[-2:]:
        print(f"Running {name} {arg} in {subdir}")
        result_dir = f"results_ncu/{name}/{subdir}"
        os.makedirs(result_dir, exist_ok=True)

        result_dir = os.path.join(os.getcwd(), result_dir)
        cwd = os.path.join(module.path_to_bin, "" if module.cwd == None else module.cwd[name])
        env = os.environ
        if suite_name == "casio":
          env["CASIO"] = module.path_to_bin
          env["PLAT"] = "GPU"
          env["DEV"] = f"cuda:{0}"
          env["SAMP"] = "all"
        
        # Nsys profiling
        if module.cmd != None:
          cmd = f"{module.cmd[name]} {arg}"
        else:
          cmd = f"./{name} {arg}"
        cmd += f" > {result_dir}/nsys_{name}.out 2> {result_dir}/nsys_{name}.err"

        start_time = time.time()
        subprocess.run([f"CUDA_VISIBLE_DEVICES={device_id} nsys profile -o {result_dir}/nsys_{name} {'--force-overwrite true' if nsys_overwrite_flag else ''} {cmd}"], shell=True, cwd = cwd, env = env)
        subprocess.run([f"CUDA_VISIBLE_DEVICES={device_id} nsys stats -r cuda_kern_exec_trace -f csv --force-export true {result_dir}/nsys_{name}.nsys-rep > {result_dir}/nsys_{name}.csv"], shell=True, cwd = cwd, env = env)
        subprocess.run([f"CUDA_VISIBLE_DEVICES={device_id} nsys stats -r cuda_gpu_kern_gb_sum -f csv --force-export true {result_dir}/nsys_{name}.nsys-rep > {result_dir}/nsys_{name}_summary.csv"], shell=True, cwd = cwd, env = env)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[PROFILE] nsys elapsed time: {elapsed_time}")
  except ModuleNotFoundError:
    print(f"Error: Module '{suite_name}' not found in the 'suites' package.")
  return

def run_ncu(suite_name: str, device_id: int) -> None:
  try:
    module = importlib.import_module(f'suites.{suite_name}')
    for name in module.names:
      for arg, subdir in list(zip(module.args[name], module.subdirs[name]))[-2:]:
        print(f"Running {name} {arg} in {subdir}")
        result_dir = f"results_ncu/{name}/{subdir}"
        os.makedirs(result_dir, exist_ok=True)

        result_dir = os.path.join(os.getcwd(), result_dir)
        cwd = os.path.join(module.path_to_bin, "" if module.cwd == None else module.cwd[name])
        env = os.environ
        if suite_name == "casio":
          env["CASIO"] = module.path_to_bin
          env["PLAT"] = "GPU"
          env["DEV"] = f"cuda:{0}"
          env["SAMP"] = "all"
        
        # Cache Flush between Kernel calls
        if module.cmd != None:
          cmd = f"{module.cmd[name]} {arg}"
        else:
          cmd = f"./{name} {arg}"
        cmd += f" > {result_dir}/ncu_{name}_flush.csv 2> {result_dir}/ncu_{name}.err"
        start_time = time.time()
        subprocess.run([f"CUDA_VISIBLE_DEVICES={device_id} ncu --metrics {','.join(ncu_metrics)} --cache-control all --replay-mode kernel --csv {cmd}"], shell=True, cwd = cwd, env = env)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[PROFILE] ncu elapsed time: {elapsed_time}")
        
  except ModuleNotFoundError:
    print(f"Error: Module '{suite_name}' not found in the 'suites' package.")
  return

def run_photon(suite_name: str, device_id: int) -> None:
  try:
    module = importlib.import_module(f'suites.{suite_name}')
    for name in module.names:
      for arg, subdir in list(zip(module.args[name], module.subdirs[name]))[-2:]:
        print(f"Running {name} {arg} in {subdir}")
        result_dir = f"results_photon/{name}/{subdir}"
        os.makedirs(result_dir, exist_ok=True)

        result_dir = os.path.join(os.getcwd(), result_dir)
        cwd = os.path.join(module.path_to_bin, "" if module.cwd == None else module.cwd[name])
        env = os.environ
        if suite_name == "casio":
          env["CASIO"] = module.path_to_bin
          env["PLAT"] = "GPU"
          env["DEV"] = f"cuda:{0}"
          env["SAMP"] = "all"
        
        # NVBit profiling
        if module.cmd != None:
          cmd = f"{module.cmd[name]} {arg}"
        else:
          cmd = f"./{name} {arg}"
        cmd += f" > {result_dir}/{name}.out 2> {result_dir}/{name}.err"

        if os.path.exists(os.path.join(result_dir, 'bbv.csv')):
          os.remove(os.path.join(result_dir, 'bbv.csv'))

        start_time = time.time()
        subprocess.run([f"CUDA_VISIBLE_DEVICES={device_id} \
                          CUDA_INJECTION64_PATH={os.getcwd()}/../nvbit-photon/photon/photon.so \
                          BBV_FILE_PATH={result_dir}/bbv.csv \
                          {cmd}"], shell=True, cwd = cwd, env = env)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[PROFILE] photon elapsed time: {elapsed_time}")

  except ModuleNotFoundError:
    print(f"Error: Module '{suite_name}' not found in the 'suites' package.")
  return

def main() -> None:
  # os.environ['TMPDIR'] = "temp"

  run_nsys("rodinia-tiny", device_id = 0, nsys_overwrite_flag = True)
  run_ncu("rodinia-tiny", device_id = 0)
  run_photon("rodinia-tiny", device_id = 0)

  return

if __name__ == '__main__':
  main()