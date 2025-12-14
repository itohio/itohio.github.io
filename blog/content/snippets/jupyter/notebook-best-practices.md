---
title: "Jupyter Notebook Best Practices"
date: 2024-12-12T20:10:00Z
draft: false
description: "Best practices for research notebooks"
type: "snippet"
tags: ["jupyter", "notebook", "best-practices", "research", "jupyter-knowhow"]
category: "jupyter"
---



Best practices for creating reproducible, maintainable research notebooks. Follow these guidelines to make your notebooks easier to understand, share, and reproduce.

## Use Case

Use these practices when you need to:
- Create reproducible research
- Share notebooks with collaborators
- Document experiments
- Present results

---

## Docker Setup

### Docker Run

```bash
# Run Jupyter Lab
docker run -d \
  --name jupyter \
  -p 8888:8888 \
  -v $(pwd):/home/jovyan/work \
  -e JUPYTER_ENABLE_LAB=yes \
  jupyter/scipy-notebook

# Get token
docker logs jupyter
```

### Docker Compose

```yaml
version: '3.8'

services:
  jupyter:
    image: jupyter/scipy-notebook:latest
    container_name: jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
    environment:
      JUPYTER_TOKEN: "mytoken"
    restart: unless-stopped

volumes:
  jupyter-data:
```

---

## Best Practices

### 1. Structure Your Notebook

```python
# Cell 1: Title and Description
"""
# Experiment: Algorithm Performance Analysis
**Date:** 2024-12-12
**Author:** Your Name
**Goal:** Compare performance of algorithms A, B, and C
"""

# Cell 2: Imports (all at the top)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Cell 3: Configuration and Constants
RANDOM_SEED = 42
DATA_PATH = "data/input.csv"
OUTPUT_PATH = "results/"

np.random.seed(RANDOM_SEED)

# Cell 4: Helper Functions
def load_data(path):
    """Load and preprocess data."""
    return pd.read_csv(path)

# Cell 5+: Analysis sections with markdown headers
```

### 2. Use Markdown Cells Liberally

```markdown
## Data Loading

Load the dataset and perform initial exploration.

**Expected outcome:** Dataset with 1000 samples, 10 features
```

### 3. Magic Commands

```python
# Time cell execution
%%time
result = expensive_computation()

# Profile memory usage
%load_ext memory_profiler
%memit large_array = np.zeros((10000, 10000))

# Reload modules automatically
%load_ext autoreload
%autoreload 2

# Display all outputs (not just last)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Set matplotlib inline
%matplotlib inline
```

### 4. Version Control Integration

```bash
# Install nbstripout to remove output from commits
pip install nbstripout

# Set up for repository
nbstripout --install

# Or manually before commit
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

### 5. Reproducibility Checklist

```python
# Cell 1: Environment info
import sys
import platform

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")

# Cell 2: Set all random seeds
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# If using PyTorch
# import torch
# torch.manual_seed(SEED)

# If using TensorFlow
# import tensorflow as tf
# tf.random.set_seed(SEED)
```

## Examples

### Example 1: Experiment Template

```python
"""
# Experiment: [Name]
**Date:** YYYY-MM-DD
**Hypothesis:** [What you're testing]
**Expected Outcome:** [What you expect to find]
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Configuration
SEED = 42
np.random.seed(SEED)

# Load Data
data = load_data()
print(f"Data shape: {data.shape}")

# Preprocessing
processed_data = preprocess(data)

# Analysis
results = run_experiment(processed_data)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(results)
plt.title("Experiment Results")
plt.show()

# Conclusions
"""
## Results
- Finding 1: [Description]
- Finding 2: [Description]

## Next Steps
- [ ] Investigate edge case
- [ ] Run with larger dataset
"""
```

### Example 2: Debugging Setup

```python
# Enable detailed error messages
%xmode Verbose

# Post-mortem debugging
%pdb on

# Interactive debugger
from IPython.core.debugger import set_trace

def problematic_function(x):
    set_trace()  # Debugger will stop here
    result = x / 0
    return result
```

### Example 3: Progress Bars

```python
from tqdm.notebook import tqdm
import time

# For loops
for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.01)

# For pandas operations
tqdm.pandas(desc="Applying function")
df['result'] = df['column'].progress_apply(lambda x: expensive_function(x))
```

### Example 4: Interactive Widgets

```python
import ipywidgets as widgets
from IPython.display import display

def plot_function(frequency=1.0, amplitude=1.0):
    x = np.linspace(0, 10, 1000)
    y = amplitude * np.sin(frequency * x)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.title(f"Sine Wave (f={frequency}, A={amplitude})")
    plt.show()

# Create interactive controls
widgets.interact(
    plot_function,
    frequency=(0.1, 5.0, 0.1),
    amplitude=(0.1, 2.0, 0.1)
)
```

## Notes

- Keep notebooks focused - one experiment per notebook
- Use descriptive cell outputs (print statements, plots)
- Document assumptions and decisions
- Include negative results - they're valuable too
- Export final results to separate files

## Gotchas/Warnings

- ⚠️ **Cell order**: Notebooks can be run out of order - test by "Restart & Run All"
- ⚠️ **Hidden state**: Variables persist between cells - can cause confusion
- ⚠️ **Large outputs**: Clear output of cells with large data to reduce file size
- ⚠️ **Git conflicts**: Notebook JSON is hard to merge - use nbdime or strip outputs