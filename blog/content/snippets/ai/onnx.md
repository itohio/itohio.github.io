---
title: "ONNX Model Conversion"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "onnx", "model-conversion", "interoperability"]
---


ONNX (Open Neural Network Exchange) for converting models between frameworks.

---

## Installation

```bash
pip install onnx onnxruntime
pip install tf2onnx  # TensorFlow to ONNX
pip install onnx2pytorch  # ONNX to PyTorch
```

---

## PyTorch to ONNX

```python
import torch
import torch.onnx

# Load PyTorch model
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

---

## TensorFlow to ONNX

```bash
# Command line
python -m tf2onnx.convert \
    --saved-model tensorflow_model/ \
    --output model.onnx \
    --opset 13

# From Keras
python -m tf2onnx.convert \
    --keras model.h5 \
    --output model.onnx \
    --opset 13
```

```python
# Python API
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name='input'),)
output_path = 'model.onnx'

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
```

---

## Run ONNX Model

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('model.onnx')

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Prepare input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run([output_name], {input_name: input_data})
print(outputs[0])
```

---

## Validate ONNX Model

```python
import onnx

# Load model
model = onnx.load('model.onnx')

# Check model
onnx.checker.check_model(model)
print('Model is valid!')

# Print model info
print(onnx.helper.printable_graph(model.graph))
```

---

## Optimize ONNX Model

```python
from onnxruntime.transformers import optimizer

# Optimize
optimized_model = optimizer.optimize_model(
    'model.onnx',
    model_type='bert',
    num_heads=12,
    hidden_size=768
)

optimized_model.save_model_to_file('model_optimized.onnx')
```

---

## ONNX to PyTorch

```python
import onnx
from onnx2pytorch import ConvertModel

# Load ONNX model
onnx_model = onnx.load('model.onnx')

# Convert to PyTorch
pytorch_model = ConvertModel(onnx_model)

# Use like normal PyTorch model
import torch
input_tensor = torch.randn(1, 3, 224, 224)
output = pytorch_model(input_tensor)
```

---

## ONNX Runtime Providers

```python
import onnxruntime as ort

# List available providers
print(ort.get_available_providers())

# Use specific provider
session = ort.InferenceSession(
    'model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

---