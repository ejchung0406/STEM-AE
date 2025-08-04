import os

path_to_bin = "../workloads/rodinia/gpu-rodinia/bin/linux/cuda/"

names = [
  "backprop-tiny",
  "bfs-tiny",
  "dwt2d-tiny",
  "gaussian-tiny",
  "lud_cuda-tiny",
  "needle-tiny",
  "particlefilter_float-tiny",
  "particlefilter_naive-tiny",
  "pathfinder-tiny",
  "sc_gpu-tiny",
  "srad_v1-tiny",
  "srad_v2-tiny",
]

args = {
  "backprop-tiny": ["524288"],
  "bfs-tiny": ["../../../../rodinia-data/bfs/graph256k.txt"],
  "dwt2d-tiny": ["../../../../rodinia-data/dwt2d/rgb.bmp -d 1024x1024 -f -5 -l 3"],
  "gaussian-tiny": ["-f ../../../../rodinia-data/gaussian/matrix128.txt"],
  "lud_cuda-tiny": ["-i ../../../../rodinia-data/lud/64.dat"],
  "needle-tiny": ["64 10"],
  "particlefilter_float-tiny": ["-x 64 -y 64 -z 5 -np 10"],
  "particlefilter_naive-tiny": ["-x 128 -y 128 -z 10 -np 1000"],
  "pathfinder-tiny": ["50000 500 100"],
  "sc_gpu-tiny": ["10 20 16 64 16 100 none none 1"],
  "srad_v1-tiny": ["10 0.5 64 64"],
  "srad_v2-tiny": ["64 64 0 32 0 32 0.5 10"],
}

subdirs = {
  "backprop-tiny": [""],
  "bfs-tiny": [""],
  "dwt2d-tiny": [""],
  "gaussian-tiny": [""],
  "lud_cuda-tiny": [""],
  "needle-tiny": [""],
  "particlefilter_float-tiny": [""],
  "particlefilter_naive-tiny": [""],
  "pathfinder-tiny": [""],
  "sc_gpu-tiny": [""],
  "srad_v1-tiny": [""],
  "srad_v2-tiny": [""],  
}

cwd = None
cmd = {
  "backprop-tiny": "./backprop",
  "bfs-tiny": "./bfs",
  "dwt2d-tiny": "./dwt2d",
  "gaussian-tiny": "./gaussian",
  "lud_cuda-tiny": "./lud_cuda",
  "needle-tiny": "./needle",
  "particlefilter_float-tiny": "./particlefilter_float",
  "particlefilter_naive-tiny": "./particlefilter_naive",
  "pathfinder-tiny": "./pathfinder",
  "sc_gpu-tiny": "./sc_gpu",
  "srad_v1-tiny": "./srad_v1",
  "srad_v2-tiny": "./srad_v2",
}