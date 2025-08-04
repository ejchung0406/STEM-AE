import os

path_to_bin = "../workloads/casio/"

names = [
  "bert-infer",
  "bert-train",
  "dlrm-infer",
  "dlrm-train",
  # "muzero",
  "resnet-infer",
  "resnet-train",
  # "rnnt-infer",
  "rnnt-train",
  "ssdrn34-infer",
  "ssdrn34-train",
  # "swin-transformer",
  "unet-infer",
  "unet-train",
]


cmd = {
  "bert-infer": "python3 -m bert.infer",
  "bert-train": "python3 -m bert.train",
  "dlrm-infer": "python3 -m dlrm.infer",
  "dlrm-train": "python3 -m dlrm.train",
  "muzero": "python muzero/minimuzero.py",
  "resnet-infer": "python3 -m resnet50.infer",
  "resnet-train": "python3 -m resnet50.train",
  "rnnt-infer": "python3 -m rnnt.infer",
  "rnnt-train": "python3 -m rnnt.train",
  "ssdrn34-infer": "python3 -m ssdrn34.infer",
  "ssdrn34-train": "python3 -m ssdrn34.train",
  "swin-transformer": "python Swin-Transformer/main.py",
  "unet-infer": "python3 -m unet.infer",
  "unet-train": "python3 -m unet.train",
}

cwd = {
  "bert-infer": "",
  "bert-train": "",
  "dlrm-infer": "",
  "dlrm-train": "",
  "muzero": "",
  "resnet-infer": "",
  "resnet-train": "",
  "rnnt-infer": "",
  "rnnt-train": "",
  "ssdrn34-infer": "",
  "ssdrn34-train": "",
  "swin-transformer": "",
  "unet-infer": "",
  "unet-train": "",
}

args = {
  "bert-infer": [""],
  "bert-train": [""],
  "dlrm-infer": [""],
  "dlrm-train": [""],
  "muzero": ["tictactoe", "atari"],
  "resnet-infer": [""],
  "resnet-train": [""],
  "rnnt-infer": [""],
  "rnnt-train": [""],
  "ssdrn34-infer": [""],
  "ssdrn34-train": [""],
  "swin-transformer": ["--cfg Swin-Transformer/configs/swinv2/swinv2_base_patch4_window12_192_22k.yaml", 
                       "--cfg Swin-Transformer/configs/swinv2/swinv2_base_patch4_window16_256.yaml"],
  "unet-infer": [""],
  "unet-train": [""],
}

subdirs = {
  "bert-infer": ["default"],
  "bert-train": ["default"],
  "dlrm-infer": ["default"],
  "dlrm-train": ["default"],
  "muzero": ["tictactoe", "atari"],
  "resnet-infer": ["default"],
  "resnet-train": ["default"],
  "rnnt-infer": ["default"],
  "rnnt-train": ["default"],
  "ssdrn34-infer": ["default"],
  "ssdrn34-train": ["default"],
  "swin-transformer": ["finetuning", "default"],
  "unet-infer": ["default"],
  "unet-train": ["default"],
}
