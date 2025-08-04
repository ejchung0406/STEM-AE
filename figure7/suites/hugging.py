path_to_bin = "../workloads/hugging/"

names = [
  "bert_medium",
  "bloom",
  "deit",
  "gemma",
  "gpt2",
  "resnet50",
]

cmd = {
  "bert_medium": "python3 bert_medium.py",
  "bloom": "python3 bloom.py",
  "deit": "python3 deit.py",
  "gemma": "python3 gemma.py",
  "gpt2": "python3 gpt2.py",
  "resnet50": "python3 resnet50.py",
}

cwd = None

args = {
  "bert_medium": [""],
  "bloom": [""],
  "deit": [""],
  "gemma": [""],
  "gpt2": [""],
  "resnet50": [""],
}

subdirs = {
  "bert_medium": ["default"],
  "bloom": ["default"],
  "deit": ["default"],
  "gemma": ["default"],
  "gpt2": ["default"],
  "resnet50": ["default"],
}