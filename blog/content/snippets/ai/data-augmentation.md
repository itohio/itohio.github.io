---
title: "Data Augmentation"
date: 2024-12-13
draft: false
description: "Data augmentation techniques for Keras and PyTorch"
tags: ["ai", "ml", "keras", "pytorch", "data-augmentation", "computer-vision", "deep-learning"]
---

Comprehensive guide to data augmentation techniques for training robust deep learning models.

---

## Why Data Augmentation?

**Benefits**:
- Increases training data diversity
- Reduces overfitting
- Improves model generalization
- Makes models robust to variations
- Cost-effective alternative to collecting more data

**Common Use Cases**:
- Computer vision (images)
- Natural language processing (text)
- Audio processing
- Time series data

---

## Keras Data Augmentation

### Basic Image Augmentation (Legacy)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,           # Random rotation ±20 degrees
    width_shift_range=0.2,       # Horizontal shift 20%
    height_shift_range=0.2,      # Vertical shift 20%
    shear_range=0.2,            # Shear transformation
    zoom_range=0.2,             # Random zoom
    horizontal_flip=True,        # Random horizontal flip
    vertical_flip=False,         # No vertical flip
    fill_mode='nearest',         # Fill strategy for new pixels
    brightness_range=[0.8, 1.2], # Brightness adjustment
    rescale=1./255               # Normalize to [0,1]
)

# Fit on training data
datagen.fit(X_train)

# Generate augmented batches
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
    # Train model on augmented batch
    model.fit(X_batch, y_batch)
```

### Modern Keras Layers (Recommended)

```python
from tensorflow import keras
from tensorflow.keras import layers

# Build augmentation pipeline as model layers
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# Integrate into model
model = keras.Sequential([
    # Augmentation layers (only active during training)
    data_augmentation,
    
    # Model architecture
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Augmentation automatically applied during training
model.fit(X_train, y_train, epochs=10)
```

### Advanced Augmentation Techniques

```python
import tensorflow as tf
from tensorflow.keras import layers

class MixupLayer(layers.Layer):
    """Mixup augmentation: blend two images and their labels"""
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        images, labels = inputs
        batch_size = tf.shape(images)[0]
        
        # Sample lambda from Beta distribution
        lam = tf.random.uniform([batch_size, 1, 1, 1], 0, 1)
        lam = tf.maximum(lam, 1 - lam)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
        
        # Mix labels
        lam_labels = tf.reshape(lam, [batch_size, 1])
        mixed_labels = lam_labels * labels + (1 - lam_labels) * tf.gather(labels, indices)
        
        return mixed_images, mixed_labels

class CutMixLayer(layers.Layer):
    """CutMix augmentation: cut and paste patches between images"""
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        images, labels = inputs
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        
        # Sample lambda
        lam = tf.random.uniform([], 0, 1)
        
        # Random box
        cut_ratio = tf.sqrt(1 - lam)
        cut_h = tf.cast(cut_ratio * tf.cast(height, tf.float32), tf.int32)
        cut_w = tf.cast(cut_ratio * tf.cast(width, tf.float32), tf.int32)
        
        cx = tf.random.uniform([], 0, width, dtype=tf.int32)
        cy = tf.random.uniform([], 0, height, dtype=tf.int32)
        
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, width)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, height)
        
        # Shuffle and mix
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Create mask
        mask = tf.ones([batch_size, height, width, 1])
        mask = tf.tensor_scatter_nd_update(
            mask,
            [[i, j, k, 0] for i in range(batch_size) 
             for j in range(y1, y2) for k in range(x1, x2)],
            tf.zeros([batch_size * (y2-y1) * (x2-x1)])
        )
        
        mixed_images = mask * images + (1 - mask) * shuffled_images
        
        # Adjust lambda based on actual box size
        lam = 1 - (tf.cast((x2-x1)*(y2-y1), tf.float32) / 
                   tf.cast(height*width, tf.float32))
        mixed_labels = lam * labels + (1 - lam) * shuffled_labels
        
        return mixed_images, mixed_labels

# Use in model
inputs = keras.Input(shape=(224, 224, 3))
labels = keras.Input(shape=(10,))

# Apply augmentation
x, y = MixupLayer()([inputs, labels])

# Continue with model architecture
x = layers.Conv2D(64, 3, activation='relu')(x)
# ... rest of model
```

### Custom Augmentation Pipeline

```python
import tensorflow as tf

def augment_image(image, label):
    """Custom augmentation function"""
    # Random crop and resize
    image = tf.image.random_crop(image, size=[200, 200, 3])
    image = tf.image.resize(image, [224, 224])
    
    # Color augmentation
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)
    
    # Geometric augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    
    # Normalize
    image = tf.clip_by_value(image, 0, 1)
    
    return image, label

# Apply to dataset
train_dataset = train_dataset.map(
    augment_image,
    num_parallel_calls=tf.data.AUTOTUNE
)
```

---

## PyTorch Data Augmentation

### Basic Transforms

```python
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Define augmentation pipeline
train_transform = transforms.Compose([
    # Resize
    transforms.Resize(256),
    transforms.RandomCrop(224),
    
    # Geometric transformations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10
    ),
    
    # Color augmentation
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),
    
    # Advanced
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    
    # Normalize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Validation transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Apply to dataset
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder('data/train', transform=train_transform)
val_dataset = ImageFolder('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Advanced Augmentation with Albumentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Powerful augmentation library
train_transform = A.Compose([
    # Resize and crop
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                       rotate_limit=15, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    
    # Color and lighting
    A.RandomBrightnessContrast(brightness_limit=0.2, 
                               contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, 
                         sat_shift_limit=30, 
                         val_shift_limit=20, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, 
               b_shift_limit=15, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    
    # Blur and noise
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
    ], p=0.3),
    
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.ISONoise(),
    ], p=0.3),
    
    # Weather effects
    A.OneOf([
        A.RandomRain(p=1.0),
        A.RandomFog(p=1.0),
        A.RandomSunFlare(p=1.0),
    ], p=0.1),
    
    # Cutout/Erasing
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                    fill_value=0, p=0.5),
    
    # Normalize and convert
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Custom dataset with albumentations
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
```

### Mixup and CutMix in PyTorch

```python
import numpy as np

def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Random box
    _, _, H, W = x.size()
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    # Adjust lambda
    lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# Training loop with mixup/cutmix
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Apply augmentation
        if np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
        else:
            images, labels_a, labels_b, lam = cutmix_data(images, labels)
        
        # Forward pass
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### AutoAugment and RandAugment

```python
from torchvision.transforms import autoaugment, RandAugment

# AutoAugment (learned policies)
auto_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    autoaugment.AutoAugment(
        policy=autoaugment.AutoAugmentPolicy.IMAGENET
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# RandAugment (simpler, often better)
rand_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## Text Data Augmentation

### Keras Text Augmentation

```python
import tensorflow as tf
import random

def augment_text(text, label):
    """Text augmentation techniques"""
    # Synonym replacement (requires nlpaug or similar)
    # Back translation (translate to another language and back)
    # Random insertion, deletion, swap
    
    # Simple example: random word dropout
    words = tf.strings.split(text).numpy()
    if len(words) > 3:
        # Drop 10% of words randomly
        keep_prob = 0.9
        words = [w for w in words if random.random() < keep_prob]
    
    augmented_text = ' '.join(words)
    return augmented_text, label
```

### PyTorch Text Augmentation

```python
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

# Synonym replacement
syn_aug = naw.SynonymAug(aug_src='wordnet')

# Contextual word embeddings
bert_aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute"
)

# Back translation
back_trans_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en'
)

# Apply augmentation
text = "The movie was absolutely fantastic"
augmented_texts = [
    syn_aug.augment(text),
    bert_aug.augment(text),
    back_trans_aug.augment(text)
]
```

---

## Best Practices

### When to Use Augmentation

```python
# ✅ Good practices
# 1. Use augmentation only on training data
train_dataset = train_dataset.map(augment)
val_dataset = val_dataset  # No augmentation

# 2. Start with simple augmentations
simple_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ColorJitter(0.1, 0.1, 0.1)
])

# 3. Gradually add complexity
# 4. Monitor validation performance
# 5. Use domain-appropriate augmentations
```

### Augmentation Strength

```python
# Light augmentation (high-quality data)
light_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

# Medium augmentation (standard)
medium_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                          saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

# Heavy augmentation (small datasets)
heavy_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                          saturation=0.3, hue=0.2),
    transforms.RandomErasing(p=0.5),
    transforms.ToTensor()
])
```

### Performance Optimization

```python
# Keras: Use prefetching
train_dataset = train_dataset.map(
    augment,
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# PyTorch: Use multiple workers
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

---

## Common Pitfalls

```python
# ❌ Don't augment validation/test data
# ❌ Don't use unrealistic augmentations
#    (e.g., vertical flip for text/faces)
# ❌ Don't over-augment (can hurt performance)
# ❌ Don't forget to normalize after augmentation
# ❌ Don't apply augmentation twice accidentally

# ✅ Do validate augmentation visually
import matplotlib.pyplot as plt

def visualize_augmentations(dataset, num_samples=5):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i, (image, label) in enumerate(dataset.take(num_samples)):
        # Original
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Original: {label}')
        axes[0, i].axis('off')
        
        # Augmented
        aug_image, _ = augment(image, label)
        axes[1, i].imshow(aug_image)
        axes[1, i].set_title('Augmented')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

---

## Related Topics

- **Transfer Learning**: Pre-trained models with augmentation
- **Semi-Supervised Learning**: Augmentation for unlabeled data
- **Test-Time Augmentation**: Average predictions over augmented test samples
- **Adversarial Training**: Augmentation with adversarial examples

