---
title: "Text to Vector Embeddings"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "embeddings", "nlp", "transformers", "word2vec", "sentence-transformers"]
---

Text embeddings convert textual content into dense vector representations that capture semantic meaning, enabling similarity search, classification, and retrieval in natural language processing.

## Core Idea

**Text embeddings** map text sequences (words, sentences, documents) to fixed-size vectors in a high-dimensional space where semantically similar texts are close together. The embedding function $E: \mathcal{T} \rightarrow \mathbb{R}^d$ transforms text $T \in \mathcal{T}$ into a vector $\mathbf{v} \in \mathbb{R}^d$.

### Mathematical Foundation

**Tokenization and Encoding:**

Text is first tokenized into subword units:
$$T \rightarrow [t_1, t_2, \ldots, t_n]$$

Each token is mapped to an embedding:
$$\mathbf{e}_i = \text{Embedding}(t_i) \in \mathbb{R}^d$$

**Inference (Forward Pass):**

For Transformer-based encoders (BERT, Sentence-BERT):
$$\mathbf{v} = E(T) = \text{Pool}(\text{Transformer}([\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n]))$$

For mean pooling:
$$\mathbf{v} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{h}_i$$

where $\mathbf{h}_i$ are hidden states from the transformer encoder.

For CLS token pooling:
$$\mathbf{v} = \mathbf{h}_{\text{CLS}}$$

**Training Objective:**

Contrastive learning with InfoNCE loss:
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{v}_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{v}_i, \mathbf{v}_j) / \tau)}$$

where:
- $\mathbf{v}_i$ is the anchor embedding
- $\mathbf{v}_i^+$ is the positive (similar) embedding
- $\mathbf{v}_j$ are all embeddings in the batch (including negatives)
- $\text{sim}(\cdot, \cdot)$ is cosine similarity
- $\tau$ is the temperature parameter

**Supervised Fine-tuning (Sentence-BERT):**

For sentence pairs $(s_i, s_j)$ with label $y_{ij}$:
$$\mathcal{L} = -\sum_{(i,j)} y_{ij} \log(\sigma(\text{sim}(\mathbf{v}_i, \mathbf{v}_j))) + (1-y_{ij}) \log(1-\sigma(\text{sim}(\mathbf{v}_i, \mathbf{v}_j)))$$

where $\sigma$ is the sigmoid function.

---

## Architecture Overview

```mermaid
graph LR
    A[Input Text] --> B[Tokenization]
    B --> C[Token Embeddings]
    C --> D[Position Embeddings]
    D --> E[Transformer Encoder]
    E --> F[Pooling]
    F --> G[Embedding Vector]
    G --> H[Similarity Search]
```

---

## PyTorch Implementation

### BERT-based Text Encoder

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERTTextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", pooling="mean"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.embedding_dim = self.bert.config.hidden_size
        self.pooling = pooling
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.pooling == "cls":
            # Use CLS token
            embedding = outputs.last_hidden_state[:, 0]
        elif self.pooling == "mean":
            # Mean pooling with attention mask
            hidden_states = outputs.last_hidden_state
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embedding = sum_embeddings / sum_mask
            else:
                embedding = hidden_states.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return nn.functional.normalize(embedding, p=2, dim=1)

# Usage
encoder = BERTTextEncoder(pooling="mean")
encoder.eval()

text = "This is a sample sentence."
inputs = encoder.tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

with torch.no_grad():
    embedding = encoder(**inputs)
    # embedding shape: [1, 768]
```

### Sentence-BERT Style Encoder

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SentenceBERTEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        hidden_states = outputs.last_hidden_state
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        else:
            embedding = hidden_states.mean(dim=1)
        
        return nn.functional.normalize(embedding, p=2, dim=1)

# Usage
encoder = SentenceBERTEncoder()
encoder.eval()

texts = ["First sentence.", "Second sentence."]
inputs = encoder.tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

with torch.no_grad():
    embeddings = encoder(**inputs)
    # embeddings shape: [2, 384]
```

### Training with Contrastive Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive):
        # anchor: [B, d], positive: [B, d]
        batch_size = anchor.size(0)
        
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Compute similarity matrix: [B, B]
        similarity_matrix = torch.matmul(anchor, positive.t()) / self.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=anchor.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

# Training loop
encoder = SentenceBERTEncoder()
criterion = InfoNCELoss(temperature=0.05)
optimizer = torch.optim.Adam(encoder.parameters(), lr=2e-5)

for anchor_texts, positive_texts in dataloader:
    optimizer.zero_grad()
    
    # Encode anchor texts
    anchor_inputs = encoder.tokenizer(
        anchor_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    anchor_emb = encoder(**anchor_inputs)
    
    # Encode positive texts
    positive_inputs = encoder.tokenizer(
        positive_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    positive_emb = encoder(**positive_inputs)
    
    loss = criterion(anchor_emb, positive_emb)
    loss.backward()
    optimizer.step()
```

---

## LangChain Integration

### Basic Text Embeddings

```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Using OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create embeddings for texts
texts = ["First document.", "Second document.", "Third document."]
text_embeddings = embeddings.embed_documents(texts)

# Single query embedding
query_embedding = embeddings.embed_query("What is the first document?")
```

### Vector Store with Text Embeddings

```python
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Create documents
documents = [
    Document(page_content="Machine learning is a subset of AI."),
    Document(page_content="Deep learning uses neural networks."),
    Document(page_content="Natural language processing enables text understanding.")
]

# Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# Similarity search
results = vectorstore.similarity_search("What is AI?", k=2)

# Similarity search with scores
results_with_scores = vectorstore.similarity_search_with_score(
    "What is AI?",
    k=2
)
```

### Advanced: Custom Embedding Function

```python
from langchain.embeddings.base import Embeddings
from transformers import AutoModel, AutoTokenizer
import torch

class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def embed_documents(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Usage
custom_embeddings = CustomHuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(documents, custom_embeddings)
```

### RAG Pipeline with Text Embeddings

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load and split documents
documents = load_documents()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)

# Query
response = qa_chain.run("What is the main topic?")
```

---

## Key Concepts

**Tokenization**: Converts text to subword units:
- **Word-level**: "machine learning" → ["machine", "learning"]
- **Subword-level (BPE)**: "machine" → ["machine", "##ing"] (handles OOV)
- **SentencePiece**: Handles multilingual text and special characters

**Position Embeddings**: Inject positional information:
$$\mathbf{e}_i = \mathbf{e}_i^{\text{token}} + \mathbf{e}_i^{\text{position}}$$

**Pooling Strategies**:
- **CLS Token**: Use special classification token embedding
- **Mean Pooling**: Average all token embeddings: $\mathbf{v} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{h}_i$
- **Max Pooling**: Element-wise maximum: $\mathbf{v} = \max_i \mathbf{h}_i$
- **Attention Pooling**: Weighted sum: $\mathbf{v} = \sum_{i} \alpha_i \mathbf{h}_i$

**Normalization**: L2 normalization ensures embeddings on unit hypersphere:
$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{||\mathbf{v}||_2}$$

**Similarity Metrics**:
- **Cosine Similarity**: $\text{sim}(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{||\mathbf{v}_1|| \cdot ||\mathbf{v}_2||}$
- **Dot Product**: $\text{sim}(\mathbf{v}_1, \mathbf{v}_2) = \mathbf{v}_1 \cdot \mathbf{v}_2$ (after normalization, equals cosine)
- **Euclidean Distance**: $d(\mathbf{v}_1, \mathbf{v}_2) = ||\mathbf{v}_1 - \mathbf{v}_2||_2$

**Chunking for Long Documents**:
- Fixed-size chunks with overlap to preserve context
- Semantic chunking based on embedding similarity
- Hierarchical chunking (parent-child relationships)

