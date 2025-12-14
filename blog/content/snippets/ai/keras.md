---
title: "Keras Essentials"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "keras", "deep-learning", "python"]
---


High-level Keras API for building neural networks quickly.

---

## Installation

```bash
# Keras is included in TensorFlow 2.x
pip install tensorflow

# Standalone Keras (multi-backend)
pip install keras
```

---

## Sequential Model

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

---

## Functional API

```python
# Multi-input model
input1 = keras.Input(shape=(32,), name='input1')
input2 = keras.Input(shape=(64,), name='input2')

x1 = layers.Dense(64, activation='relu')(input1)
x2 = layers.Dense(64, activation='relu')(input2)

# Concatenate
combined = layers.concatenate([x1, x2])
output = layers.Dense(1, activation='sigmoid')(combined)

model = keras.Model(inputs=[input1, input2], outputs=output)
```

---

## Custom Layers

```python
class MyLayer(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

---

## Custom Model

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()
```

---

## Callbacks

```python
# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_schedule = keras.callbacks.LearningRateScheduler(scheduler)

# Custom callback
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: loss = {logs['loss']:.4f}")

# Use callbacks
model.fit(
    x_train, y_train,
    epochs=50,
    callbacks=[early_stop, checkpoint, lr_schedule]
)
```

---

## Data Augmentation

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Include in model
model = keras.Sequential([
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu'),
    # ... rest of model
])
```

---