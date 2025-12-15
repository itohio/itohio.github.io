---
title: "Sound to Vector Embeddings"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "embeddings", "audio", "speech", "wav2vec", "whisper"]
---

Audio embeddings convert sound signals (speech, music, environmental sounds) into dense vector representations that capture acoustic and semantic features, enabling similarity search, classification, and retrieval.

## Core Idea

**Audio embeddings** map audio waveforms or spectrograms to fixed-size vectors in a high-dimensional space where semantically similar sounds are close together. The embedding function $E: \mathcal{A} \rightarrow \mathbb{R}^d$ transforms an audio signal $A \in \mathcal{A}$ into a vector $\mathbf{v} \in \mathbb{R}^d$.

### Mathematical Foundation

**Preprocessing: Audio to Spectrogram**

Raw audio waveform $x(t)$ is converted to a spectrogram:
$$S(t, f) = |\text{STFT}(x(t))|^2$$

where:
- $\text{STFT}$ is the Short-Time Fourier Transform
- $S(t, f) \in \mathbb{R}^{T \times F}$ is the time-frequency representation
- $T$ is the number of time frames
- $F$ is the number of frequency bins

**Inference (Forward Pass):**

For CNN-based encoder:
$$\mathbf{v} = E(A) = \text{GlobalPool}(\text{CNN}(\text{MelSpectrogram}(A)))$$

For Transformer-based encoder (Wav2Vec2):
$$\mathbf{v} = E(A) = \text{MeanPool}(\text{Transformer}(\text{CNN}(A)))$$

For sequence-to-vector aggregation:
$$\mathbf{v} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{h}_t$$

where:
- $A \in \mathbb{R}^{L}$ is the raw audio waveform (length $L$ samples)
- $\mathbf{h}_t \in \mathbb{R}^d$ are hidden states at each time step
- $\mathbf{v} \in \mathbb{R}^d$ is the output embedding vector

**Training Objective:**

Self-supervised learning with contrastive loss (Wav2Vec2):
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{c}_t, \mathbf{q}_t) / \tau)}{\sum_{\tilde{q} \in Q_t} \exp(\text{sim}(\mathbf{c}_t, \tilde{\mathbf{q}}) / \tau)}$$

where:
- $\mathbf{c}_t$ is the context vector at time $t$
- $\mathbf{q}_t$ is the quantized target vector
- $Q_t$ is the set of quantized vectors (positive + negatives)
- $\tau$ is the temperature parameter

**Supervised Fine-tuning:**
$$\mathcal{L} = -\sum_{i=1}^{N} \log P(y_i | \mathbf{v}_i)$$

where $y_i$ is the label (e.g., speaker ID, emotion class) and $\mathbf{v}_i = E(A_i)$.

---

## Architecture Overview

```mermaid
graph LR
    A[Raw Audio] --> B[Preprocessing]
    B --> C[Feature Extractor]
    C --> D[Encoder]
    D --> E[Temporal Pooling]
    E --> F[Embedding Vector]
    F --> G[Similarity Search]
```

---

## PyTorch Implementation

### CNN-based Audio Encoder

```python
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=512, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Mel spectrogram extractor
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
        # CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, waveform):
        # waveform: [B, L] where L is audio length
        # Convert to mel spectrogram: [B, 1, T, F]
        mel_spec = self.mel_spectrogram(waveform).unsqueeze(1)
        
        # Extract features: [B, 256, 1, 1]
        features = self.feature_extractor(mel_spec)
        features = features.view(features.size(0), -1)
        
        # Project to embedding: [B, embedding_dim]
        embedding = self.projection(features)
        return nn.functional.normalize(embedding, p=2, dim=1)

# Usage
encoder = AudioEncoder(embedding_dim=512)
encoder.eval()

# Load audio
waveform, sample_rate = torchaudio.load("audio.wav")
# Resample if needed
if sample_rate != 16000:
    resampler = T.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# Ensure fixed length (e.g., 3 seconds)
target_length = 16000 * 3
if waveform.size(1) > target_length:
    waveform = waveform[:, :target_length]
else:
    padding = target_length - waveform.size(1)
    waveform = torch.nn.functional.pad(waveform, (0, padding))

with torch.no_grad():
    embedding = encoder(waveform)
    # embedding shape: [1, 512]
```

### Wav2Vec2-based Encoder

```python
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class Wav2Vec2AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.embedding_dim = self.wav2vec2.config.hidden_size
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    def forward(self, input_values):
        # input_values: [B, L] raw waveform
        outputs = self.wav2vec2(input_values=input_values)
        # Mean pooling over time: [B, L, d] -> [B, d]
        hidden_states = outputs.last_hidden_state
        embedding = hidden_states.mean(dim=1)
        return nn.functional.normalize(embedding, p=2, dim=1)

# Usage
encoder = Wav2Vec2AudioEncoder()
encoder.eval()

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("audio.wav")
# Wav2Vec2 expects 16kHz
if sample_rate != 16000:
    resampler = T.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# Process with Wav2Vec2 processor
inputs = encoder.processor(
    waveform.squeeze().numpy(),
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    embedding = encoder(inputs.input_values)
    # embedding shape: [1, 768]
```

### Whisper-based Encoder

```python
import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperProcessor

class WhisperAudioEncoder(nn.Module):
    def __init__(self, model_name="openai/whisper-base"):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.embedding_dim = self.whisper.config.d_model
        self.processor = WhisperProcessor.from_pretrained(model_name)
    
    def forward(self, input_features):
        # input_features: [B, T, F] mel spectrogram features
        outputs = self.whisper.encoder(input_features=input_features)
        # Mean pooling over time
        hidden_states = outputs.last_hidden_state
        embedding = hidden_states.mean(dim=1)
        return nn.functional.normalize(embedding, p=2, dim=1)

# Usage
encoder = WhisperAudioEncoder()
encoder.eval()

waveform, sample_rate = torchaudio.load("audio.wav")
inputs = encoder.processor(
    waveform.squeeze().numpy(),
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    embedding = encoder(input_features=inputs.input_features)
    # embedding shape: [1, 512]
```

### Training with Contrastive Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# Training loop
encoder = AudioEncoder(embedding_dim=512)
criterion = AudioTripletLoss(margin=0.5)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

for anchor_audio, positive_audio, negative_audio in dataloader:
    optimizer.zero_grad()
    
    anchor_emb = encoder(anchor_audio)
    positive_emb = encoder(positive_audio)
    negative_emb = encoder(negative_audio)
    
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
import torchaudio
import base64

# For audio embeddings, typically use custom models
# Example with Wav2Vec2 via HuggingFace
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class AudioEmbeddings:
    def __init__(self, model_name="facebook/wav2vec2-base"):
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model.eval()
    
    def embed_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding.squeeze().numpy()

# Create embeddings
embeddings_model = AudioEmbeddings()
audio_embeddings = [
    embeddings_model.embed_audio(f"audio_{i}.wav")
    for i in range(100)
]

documents = [Document(page_content=f"Audio {i}") for i in range(100)]

# Store in vector database
vectorstore = FAISS.from_embeddings(
    texts=[doc.page_content for doc in documents],
    embeddings=audio_embeddings
)

# Similarity search
query_emb = embeddings_model.embed_audio("query.wav")
results = vectorstore.similarity_search_by_vector(query_emb, k=5)
```

---

## Key Concepts

**Mel Spectrogram**: Converts linear frequency scale to mel scale (perceptually uniform):
$$m = 2595 \log_{10}(1 + f/700)$$

where $f$ is frequency in Hz and $m$ is mel frequency.

**Temporal Pooling**: Aggregates variable-length sequences:
- Mean Pooling: $\mathbf{v} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{h}_t$
- Max Pooling: $\mathbf{v} = \max_{t} \mathbf{h}_t$
- Attention Pooling: $\mathbf{v} = \sum_{t} \alpha_t \mathbf{h}_t$ where $\alpha = \text{softmax}(\mathbf{W}\mathbf{H})$

**Preprocessing Steps**:
1. **Resampling**: Normalize to target sample rate (typically 16kHz)
2. **Normalization**: Zero-mean, unit-variance: $x' = \frac{x - \mu}{\sigma}$
3. **Padding/Truncation**: Fixed-length sequences for batch processing

**Similarity Metrics**:
- Cosine Similarity: $\text{sim}(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{||\mathbf{v}_1|| \cdot ||\mathbf{v}_2||}$
- Euclidean Distance: $d(\mathbf{v}_1, \mathbf{v}_2) = ||\mathbf{v}_1 - \mathbf{v}_2||_2$

