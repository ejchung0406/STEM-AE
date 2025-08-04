import os
import subprocess

names = [
  "bert-infer",
  "bert-train",
  "dlrm-infer",
  "dlrm-train",
  "resnet-infer",
  "resnet-train",
  "rnnt-train",
  "ssdrn34-infer",
  "ssdrn34-train",
  "unet-infer",
  "unet-train",
]

cmds = {
  "bert-infer": "python3 -m bert.infer",
  "bert-train": "python3 -m bert.train",
  "dlrm-infer": "python3 -m dlrm.infer",
  "dlrm-train": "python3 -m dlrm.train",
  "resnet-infer": "python3 -m resnet50.infer",
  "resnet-train": "python3 -m resnet50.train",
  "rnnt-train": "python3 -m rnnt.train",
  "ssdrn34-infer": "python3 -m ssdrn34.infer",
  "ssdrn34-train": "python3 -m ssdrn34.train",
  "unet-infer": "python3 -m unet.infer",
  "unet-train": "python3 -m unet.train",
}

path_to_bin = "../workloads/casio/"

def run_nsys(device_id: int, nsys_overwrite_flag: bool) -> None:
  for name in names:
    print(f"Running {name} with nsys")
    cwd = path_to_bin
    result_dir = os.getcwd() + "/results_nsys"
    os.makedirs(result_dir, exist_ok=True)

    env = os.environ
    env["CASIO"] = path_to_bin
    env["PLAT"] = "GPU"
    env["DEV"] = f"cuda:{0}"
    env["SAMP"] = "all"
      
    # Nsys profiling
    cmd = f"{cmds[name]}"
    cmd += f" > {result_dir}/nsys_{name}.out 2> {result_dir}/nsys_{name}.err"

    subprocess.run([f"CUDA_VISIBLE_DEVICES={device_id} nsys profile -o {result_dir}/nsys_{name} {'--force-overwrite true' if nsys_overwrite_flag else ''} {cmd}"], shell=True, cwd = cwd, env = env)
    subprocess.run([f"CUDA_VISIBLE_DEVICES={device_id} nsys stats -r cuda_kern_exec_trace -f csv --force-export true {result_dir}/nsys_{name}.nsys-rep > {result_dir}/nsys_{name}.csv"], shell=True, cwd = cwd, env = env)
    subprocess.run([f"CUDA_VISIBLE_DEVICES={device_id} nsys stats -r cuda_gpu_kern_gb_sum -f csv --force-export true {result_dir}/nsys_{name}.nsys-rep > {result_dir}/nsys_{name}_summary.csv"], shell=True, cwd = cwd, env = env)

if __name__ == "__main__":
  run_nsys(0, True)