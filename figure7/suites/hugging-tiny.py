path_to_bin = "../workloads/hugging-tiny/"

names = [
  "bert-tiny",
  "bloom-tiny",
  "deit-tiny",
  "gemma-tiny",
  "gpt2-tiny",
  "resnet50-tiny",
]

cmd = {
  "bert-tiny": "python3 bert.py",
  "bloom-tiny": "python3 bloom.py",
  "deit-tiny": "python3 deit.py",
  "gemma-tiny": "python3 gemma.py",
  "gpt2-tiny": "python3 gpt2.py",
  "resnet50-tiny": "python3 resnet50.py",
}

subdirs = {
  "bert-tiny": [""],
  "bloom-tiny": [""],
  "deit-tiny": [""],
  "gemma-tiny": [""],
  "gpt2-tiny": [""],
  "resnet50-tiny": [""],
}

args = {
  "bert-tiny": [""],
  "bloom-tiny": [""],
  "deit-tiny": [""],
  "gemma-tiny": [""],
  "gpt2-tiny": [""],
  "resnet50-tiny": [""],
}

cwd = None