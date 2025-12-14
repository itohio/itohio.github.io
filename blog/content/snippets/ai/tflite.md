---
title: "TensorFlow Lite"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "tflite", "mobile", "edge-computing", "tensorflow"]
---


TensorFlow Lite for deploying models on mobile and embedded devices.

---

## Convert Model to TFLite

```python
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model('my_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## Optimization

```python
# Post-training quantization (Dynamic range)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Float16 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()

# Integer quantization
def representative_dataset():
    for data in dataset.take(100):
        yield [tf.dtypes.cast(data, tf.float32)]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_int8_model = converter.convert()
```

---

## Run Inference (Python)

```python
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

---

## Android Integration

```kotlin
// build.gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
}

// Kotlin code
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer

class TFLiteModel(private val modelBuffer: MappedByteBuffer) {
    private val interpreter = Interpreter(modelBuffer)
    
    fun predict(input: FloatArray): FloatArray {
        val output = Array(1) { FloatArray(10) }
        interpreter.run(input, output)
        return output[0]
    }
    
    fun close() {
        interpreter.close()
    }
}
```

---

## iOS Integration

```swift
import TensorFlowLite

class TFLiteModel {
    private var interpreter: Interpreter
    
    init(modelPath: String) throws {
        interpreter = try Interpreter(modelPath: modelPath)
        try interpreter.allocateTensors()
    }
    
    func predict(input: [Float]) throws -> [Float] {
        let inputData = Data(copyingBufferOf: input)
        try interpreter.copy(inputData, toInputAt: 0)
        try interpreter.invoke()
        
        let outputTensor = try interpreter.output(at: 0)
        let results = [Float](unsafeData: outputTensor.data) ?? []
        return results
    }
}
```

---

## Benchmark

```bash
# Install benchmark tool
pip install tensorflow

# Benchmark model
python -m tensorflow.lite.tools.benchmark_model \
    --graph=model.tflite \
    --num_threads=4
```

---