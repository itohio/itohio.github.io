---
title: "AI/ML Interview Questions - Hard"
date: 2025-12-13
tags: ["ai", "machine-learning", "interview", "hard"]
---

Hard-level AI/ML interview questions covering advanced architectures, optimization, and theoretical concepts.

## Q1: Implement attention mechanism from scratch.

**Answer**:

**How It Works**:

Attention allows model to focus on relevant parts of input when producing output.

**Core Idea**: Compute weighted sum of values, where weights are determined by query-key similarity.

**Formula**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:
- $Q$ = Queries (what we're looking for)
- $K$ = Keys (what's available)
- $V$ = Values (actual content)
- $d_k$ = dimension of keys (for scaling)

**PyTorch Implementation**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: Queries (batch_size, num_heads, seq_len, d_k)
        K: Keys (batch_size, num_heads, seq_len, d_k)
        V: Values (batch_size, num_heads, seq_len, d_v)
        mask: Optional mask (batch_size, 1, 1, seq_len)
    
    Returns:
        output: (batch_size, num_heads, seq_len, d_v)
        attention_weights: (batch_size, num_heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    
    # Calculate attention scores: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided (for padding or future tokens)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        """Split last dimension into (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply attention for each head
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention_weights

# Usage example
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

mha = MultiHeadAttention(d_model, num_heads)

# Create sample input
x = torch.randn(batch_size, seq_len, d_model)

# Self-attention: Q, K, V are all the same
output, attn_weights = mha(x, x, x)

print(f"Output shape: {output.shape}")  # (2, 10, 512)
print(f"Attention weights shape: {attn_weights.shape}")  # (2, 8, 10, 10)
```

**Why It Works**: PyTorch's autograd handles backpropagation through attention, and the model learns optimal attention patterns during training.

---

## Q2: Explain and implement Transformer architecture.

**Answer**:

**How It Works**:

Transformer uses self-attention to process sequences in parallel (unlike RNNs).

**Key Components**:
1. **Multi-Head Attention**: Multiple attention mechanisms in parallel
2. **Position Encoding**: Add positional information (no recurrence)
3. **Feed-Forward Networks**: Process each position independently
4. **Layer Normalization**: Stabilize training
5. **Residual Connections**: Help gradient flow

**Architecture**:
```
Input → Embedding + Positional Encoding
     ↓
[Encoder Block] × N:
  - Multi-Head Self-Attention
  - Add & Norm
  - Feed-Forward Network
  - Add & Norm
     ↓
[Decoder Block] × N:
  - Masked Multi-Head Self-Attention
  - Add & Norm
  - Multi-Head Cross-Attention (with encoder output)
  - Add & Norm
  - Feed-Forward Network
  - Add & Norm
     ↓
Output Linear + Softmax
```

**Positional Encoding**:
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

**PyTorch Implementation**:
```python
import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Output projection
        output = self.fc_out(x)
        return output

# Usage example
vocab_size = 10000
model = Transformer(vocab_size, d_model=512, num_heads=8, num_layers=6)

# Sample input (batch_size=2, seq_len=10)
input_ids = torch.randint(0, vocab_size, (2, 10))
output = model(input_ids)

print(f"Input shape: {input_ids.shape}")   # (2, 10)
print(f"Output shape: {output.shape}")     # (2, 10, 10000)

# Training example
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(input_ids)
    
    # Reshape for loss calculation
    predictions = predictions.view(-1, vocab_size)
    targets = torch.randint(0, vocab_size, (2, 10)).view(-1)
    
    loss = criterion(predictions, targets)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

**Why Transformers Work**:
- **Parallel processing**: Unlike RNNs, all positions processed simultaneously
- **Long-range dependencies**: Attention mechanism sees all positions at once
- **Scalable**: Can scale to billions of parameters efficiently
- **PyTorch autograd**: Handles complex backpropagation automatically

---

## Q3: Explain variational autoencoders (VAE) and implement one.

**Answer**:

**How It Works**:

VAE learns a probabilistic latent representation by encoding to distribution parameters, then sampling.

**Key Idea**: 
- Encoder outputs $\mu$ and $\sigma$ (mean and std of latent distribution)
- Sample from $\mathcal{N}(\mu, \sigma^2)$ using reparameterization trick
- Decoder reconstructs from sample

**Loss Function**:
$$
\mathcal{L} = \mathcal{L}_{\text{reconstruction}} + \beta \cdot \mathcal{L}_{\text{KL}}
$$

where:
- Reconstruction: How well we reconstruct input
- KL divergence: How close latent distribution is to $\mathcal{N}(0, 1)$

**Reparameterization Trick**:
Instead of sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ (not differentiable),
do: $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, 1)$

**Implementation**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

# Training
model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch in dataloader:
        x = batch.view(-1, 784)
        
        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Why VAE vs. Regular Autoencoder**:
- VAE: Smooth latent space, can generate new samples
- AE: Just compression, latent space may have gaps

---

## Q4: Explain Generative Adversarial Networks (GANs).

**Answer**:

**How It Works**:

Two networks compete:
- **Generator**: Creates fake samples
- **Discriminator**: Distinguishes real from fake

**Training Process**:
1. Generator creates fake samples
2. Discriminator tries to classify real vs. fake
3. Generator tries to fool discriminator
4. Both improve through adversarial training

**Loss Functions**:

**Discriminator**:
$$
\max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

**Generator**:
$$
\min_G \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

Or equivalently (non-saturating loss):
$$
\max_G \mathbb{E}_{z \sim p_z}[\log D(G(z))]
$$

**Implementation**:
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Training
generator = Generator(latent_dim=100, img_shape=(1, 28, 28))
discriminator = Discriminator(img_shape=(1, 28, 28))

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

adversarial_loss = nn.BCELoss()

for epoch in range(num_epochs):
    for real_imgs in dataloader:
        batch_size = real_imgs.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        real_loss = adversarial_loss(discriminator(real_imgs), real_labels)
        
        z = torch.randn(batch_size, 100)
        fake_imgs = generator(z)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, 100)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), real_labels)
        
        g_loss.backward()
        optimizer_G.step()
```

**Common Problems**:
- **Mode collapse**: Generator produces limited variety
- **Training instability**: Hard to balance G and D
- **Vanishing gradients**: When D is too good

**Solutions**:
- Wasserstein GAN (WGAN)
- Spectral normalization
- Progressive growing
- StyleGAN architectures

---

## Q5: Implement beam search for sequence generation.

**Answer**:

**How It Works**:

Beam search keeps top-k most likely sequences at each step, exploring multiple paths.

**Algorithm**:
1. Start with k beams (initially just start token)
2. For each beam, generate all possible next tokens
3. Score each candidate (beam score + token log probability)
4. Keep top k candidates as new beams
5. Repeat until all beams end or max length reached

**Implementation**:
```python
import torch
import torch.nn.functional as F
from queue import PriorityQueue

def beam_search(model, start_token, end_token, max_length, beam_width=5, device='cpu'):
    """
    Args:
        model: Sequence generation model
        start_token: Starting token ID
        end_token: End token ID
        max_length: Maximum sequence length
        beam_width: Number of beams to keep
    
    Returns:
        best_sequence: Most likely sequence
        best_score: Log probability of best sequence
    """
    # Initialize beams: (score, sequence, is_complete)
    beams = [(0.0, [start_token], False)]
    completed_beams = []
    
    for step in range(max_length):
        candidates = []
        
        for score, sequence, is_complete in beams:
            if is_complete:
                completed_beams.append((score, sequence))
                continue
            
            # Get model predictions for current sequence
            input_ids = torch.tensor([sequence]).to(device)
            with torch.no_grad():
                logits = model(input_ids)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            
            # Get top k tokens
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            
            # Create candidates
            for log_prob, token_id in zip(top_log_probs[0], top_indices[0]):
                new_score = score + log_prob.item()
                new_sequence = sequence + [token_id.item()]
                is_end = (token_id.item() == end_token)
                
                candidates.append((new_score, new_sequence, is_end))
        
        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)  # Normalize by length
        beams = candidates[:beam_width]
        
        # Check if all beams are complete
        if all(is_complete for _, _, is_complete in beams):
            completed_beams.extend(beams)
            break
    
    # Add remaining beams
    completed_beams.extend(beams)
    
    # Return best sequence
    if completed_beams:
        best_score, best_sequence, _ = max(completed_beams, key=lambda x: x[0] / len(x[1]))
        return best_sequence, best_score
    
    return beams[0][1], beams[0][0]

# Alternative: Batch beam search (more efficient)
def batch_beam_search(model, start_tokens, end_token, max_length, beam_width=5):
    batch_size = start_tokens.size(0)
    
    # Initialize: (batch_size * beam_width, seq_len)
    sequences = start_tokens.unsqueeze(1).repeat(1, beam_width, 1)
    sequences = sequences.view(batch_size * beam_width, -1)
    
    scores = torch.zeros(batch_size, beam_width)
    scores[:, 1:] = float('-inf')  # Only first beam is active initially
    
    for step in range(max_length):
        # Get predictions
        logits = model(sequences)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
        
        # Add to beam scores
        log_probs = log_probs.view(batch_size, beam_width, -1)
        scores = scores.unsqueeze(-1) + log_probs
        
        # Flatten and get top k
        scores_flat = scores.view(batch_size, -1)
        top_scores, top_indices = torch.topk(scores_flat, beam_width, dim=-1)
        
        # Convert flat indices to (beam_idx, token_idx)
        beam_indices = top_indices // log_probs.size(-1)
        token_indices = top_indices % log_probs.size(-1)
        
        # Update sequences
        sequences = sequences.view(batch_size, beam_width, -1)
        sequences = torch.gather(
            sequences, 
            1, 
            beam_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
        )
        sequences = torch.cat([sequences, token_indices.unsqueeze(-1)], dim=-1)
        sequences = sequences.view(batch_size * beam_width, -1)
        
        scores = top_scores
    
    # Return best sequences
    best_scores, best_indices = scores.max(dim=-1)
    best_sequences = sequences.view(batch_size, beam_width, -1)
    best_sequences = torch.gather(
        best_sequences,
        1,
        best_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, best_sequences.size(-1))
    ).squeeze(1)
    
    return best_sequences, best_scores
```

**Beam Search vs. Greedy**:
- Greedy: Always pick most likely token (fast, but suboptimal)
- Beam: Explore multiple paths (better quality, slower)

**Typical beam widths**: 5-10 for translation, 1-3 for chatbots

---

## Summary

Hard AI/ML topics require deep understanding of:
- **Attention mechanisms**: Core of modern NLP
- **Transformers**: Architecture powering GPT, BERT
- **Generative models**: VAE, GAN for creating new data
- **Search algorithms**: Beam search for sequence generation
- **Optimization**: Advanced training techniques

**Key Skills**:
- Implement from scratch (not just use libraries)
- Understand mathematical foundations
- Debug training issues
- Optimize for production

