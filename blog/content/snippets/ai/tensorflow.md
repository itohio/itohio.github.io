---
title: "TensorFlow Essentials"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "tensorflow", "deep-learning", "python"]
---


Essential TensorFlow operations and patterns for deep learning.

---

## Installation

```bash
# CPU version
pip install tensorflow

# GPU version
pip install tensorflow[and-cuda]

# Check installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Basic Tensor Operations

```python
import tensorflow as tf

# Create tensors
scalar = tf.constant(42)
vector = tf.constant([1, 2, 3])
matrix = tf.constant([[1, 2], [3, 4]])

# Tensor operations
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Element-wise operations
c = a + b
d = a * b

# Matrix multiplication
e = tf.matmul(a, b)

# Reshaping
reshaped = tf.reshape(a, [4, 1])
```

---

## Building Models (Keras API)

```python
from tensorflow import keras
from tensorflow.keras import layers

# Sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Functional API
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Training

```python
# Train model
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)

# Predict
predictions = model.predict(x_new)
```

---

## Custom Training Loop

```python
import tensorflow as tf

# Define model, loss, optimizer
model = MyModel()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

# Training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        loss = train_step(x_batch, y_batch)
```

---

## Saving and Loading

```python
# Save entire model
model.save('my_model.h5')
model.save('my_model')  # SavedModel format

# Load model
loaded_model = keras.models.load_model('my_model.h5')

# Save weights only
model.save_weights('weights.h5')
model.load_weights('weights.h5')
```

---

## TensorBoard

```python
# Setup TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
)

model.fit(x_train, y_train, callbacks=[tensorboard_callback])

# View in terminal
# tensorboard --logdir=./logs
```

---