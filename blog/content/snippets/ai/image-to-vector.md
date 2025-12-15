---
title: "Image to Vector Embeddings"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "embeddings", "computer-vision", "cnn", "vision-transformers"]
---

Image embeddings convert visual content into dense vector representations that capture semantic and visual features, enabling similarity search, classification, and retrieval.

## Core Idea

**Image embeddings** map images to fixed-size vectors in a high-dimensional space where semantically similar images are close together. The embedding function $E: \mathcal{I} \rightarrow \mathbb{R}^d$ transforms an image $I \in \mathcal{I}$ into a vector $\mathbf{v} \in \mathbb{R}^d$.

### Mathematical Foundation

**Inference (Forward Pass):**

For a CNN-based encoder:
$$\mathbf{v} = E(I) = \text{GlobalPool}(\text{CNN}(I))$$

For a Vision Transformer (ViT):
$$\mathbf{v} = E(I) = \text{CLS}_{\text{pool}}(\text{ViT}(\text{PatchEmbed}(I)))$$

where:
- $I \in \mathbb{R}^{H \times W \times C}$ is the input image
- $\mathbf{v} \in \mathbb{R}^d$ is the output embedding vector
- $d$ is the embedding dimension (typically 512, 768, or 1024)

**Training Objective:**

Contrastive learning with triplet loss:
$$\mathcal{L}_{\text{triplet}} = \max(0, d(\mathbf{v}_a, \mathbf{v}_p) - d(\mathbf{v}_a, \mathbf{v}_n) + \alpha)$$

where:
- $\mathbf{v}_a$ is the anchor embedding
- $\mathbf{v}_p$ is the positive (similar) embedding
- $\mathbf{v}_n$ is the negative (dissimilar) embedding
- $d(\cdot, \cdot)$ is the distance metric (e.g., Euclidean or cosine)
- $\alpha$ is the margin hyperparameter

**Alternative: InfoNCE Loss (used in CLIP):**
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_j) / \tau)}$$

where:
- $\mathbf{v}_i$ is the image embedding
- $\mathbf{t}_i$ is the paired text embedding
- $\text{sim}(\cdot, \cdot)$ is cosine similarity
- $\tau$ is the temperature parameter
- $N$ is the batch size

---

## Architecture Overview

```mermaid
graph LR
    A[Input Image] --> B[Preprocessing]
    B --> C[Feature Extractor]
    C --> D[Global Pooling]
    D --> E[Projection Head]
    E --> F[Embedding Vector]
    F --> G[Similarity Search]
```

---

## PyTorch Implementation

### ResNet-based Image Encoder

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Use pretrained ResNet as backbone
        resnet = models.resnet50(pretrained=True)
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Add projection head
        self.projection = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, x):
        # Extract features: [B, 2048, 1, 1]
        features = self.backbone(x)
        # Flatten: [B, 2048]
        features = features.view(features.size(0), -1)
        # Project to embedding space: [B, embedding_dim]
        embedding = self.projection(features)
        # L2 normalize
        return nn.functional.normalize(embedding, p=2, dim=1)

# Usage
encoder = ImageEncoder(embedding_dim=512)
encoder.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

image = Image.open("image.jpg")
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    embedding = encoder(image_tensor)
    # embedding shape: [1, 512]
```

### Vision Transformer (ViT) Encoder

```python
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor

class ViTImageEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.embedding_dim = self.vit.config.hidden_size
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        # Use CLS token embedding
        embedding = outputs.last_hidden_state[:, 0]
        return nn.functional.normalize(embedding, p=2, dim=1)

# Usage
encoder = ViTImageEncoder()
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    embedding = encoder(**inputs)
    # embedding shape: [1, 768]
```

### Training with Contrastive Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# Training loop
encoder = ImageEncoder()
criterion = TripletLoss(margin=0.5)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

for anchor_img, positive_img, negative_img in dataloader:
    optimizer.zero_grad()
    
    anchor_emb = encoder(anchor_img)
    positive_emb = encoder(positive_img)
    negative_emb = encoder(negative_img)
    
    loss = criterion(anchor_emb, positive_emb, negative_emb)
    loss.backward()
    optimizer.step()
```

---

## LangChain Integration

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import base64
from io import BytesIO

# For multimodal embeddings (image + text)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Using CLIP for image embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/clip-ViT-B-32"
)

# Encode image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        image_bytes = base64.b64encode(f.read()).decode()
    return embeddings.embed_image(image_bytes)

# Create vector store with image embeddings
image_embeddings = [encode_image(f"image_{i}.jpg") for i in range(100)]
documents = [Document(page_content=f"Image {i}") for i in range(100)]

vectorstore = FAISS.from_embeddings(
    texts=[doc.page_content for doc in documents],
    embeddings=image_embeddings,
    embedding=embeddings
)

# Similarity search
query_image_emb = encode_image("query.jpg")
results = vectorstore.similarity_search_by_vector(query_image_emb, k=5)
```

---

## Key Concepts

**Global Pooling**: Aggregates spatial features into a fixed-size vector:
- Average Pooling: $\mathbf{v} = \frac{1}{HW}\sum_{i,j} \mathbf{F}_{i,j}$
- Max Pooling: $\mathbf{v} = \max_{i,j} \mathbf{F}_{i,j}$
- Attention Pooling: $\mathbf{v} = \sum_{i,j} \alpha_{i,j} \mathbf{F}_{i,j}$ where $\alpha = \text{softmax}(\mathbf{W}\mathbf{F})$

**Normalization**: L2 normalization ensures embeddings lie on the unit hypersphere, making cosine similarity equivalent to dot product:
$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{||\mathbf{v}||_2}$$

**Similarity Metrics**:
- Cosine Similarity: $\text{sim}(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{||\mathbf{v}_1|| \cdot ||\mathbf{v}_2||}$
- Euclidean Distance: $d(\mathbf{v}_1, \mathbf{v}_2) = ||\mathbf{v}_1 - \mathbf{v}_2||_2$

