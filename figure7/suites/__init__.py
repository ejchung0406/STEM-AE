import os
import importlib

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

modules = []

# Iterate over all files in the directory
for filename in os.listdir(current_dir):
  # Check if the file is a Python file and not __init__.py
  if filename.endswith('.py') and filename != '__init__.py':
    # Module name is the filename without the .py extension
    module_name = filename[:-3]
    
    # Import the module dynamically
    importlib.import_module(f'.{module_name}', package=__name__)
    modules.append(modules)