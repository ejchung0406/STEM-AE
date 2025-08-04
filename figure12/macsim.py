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

def run_macsim():
  for work in workloads:
    for config in configs:
      macsim_path = f"macsim/{work}/{config}/"

      os.system(f"cp ../macsim/bin/macsim {macsim_path}")

      f = open(f"{macsim_path}/trace_file_list", "w")
      f.write(f"1\n../../../trace_nvbit/{work}/kernel_config.txt")
      f.close()

      cmd = f"./macsim > macsim.out 2>macsim.err"
      print(cmd)
      subprocess.Popen(cmd, shell=True, cwd=macsim_path)

if __name__ == "__main__":
  run_macsim()