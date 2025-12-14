---
title: "AI/ML Interview Questions - Medium"
date: 2025-12-13
tags: ["ai", "machine-learning", "interview", "medium"]
---

Medium-level AI/ML interview questions covering neural networks, ensemble methods, and advanced concepts.

## Q1: Explain backpropagation in neural networks.

**Answer**:

**How It Works**:

Backpropagation is the algorithm for training neural networks by computing gradients of the loss with respect to weights.

**Forward Pass**:
1. Input flows through network
2. Each layer applies: $z = Wx + b$, then activation $a = \sigma(z)$
3. Final layer produces prediction
4. Calculate loss: $L = \text{loss}(y_{\text{pred}}, y_{\text{true}})$

**Backward Pass** (Chain Rule):
1. Start from output layer
2. Calculate gradient of loss w.r.t. output: $\frac{\partial L}{\partial a^{(L)}}$
3. Propagate backwards using chain rule:
$$
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}
$$

**Update Weights**:
$$
W^{(l)} = W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}
$$

**PyTorch Implementation**:
```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork, self).__init__()
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x):
        # Forward pass through all layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply sigmoid activation (except for last layer if doing classification)
            if i < len(self.layers) - 1:
                x = torch.sigmoid(x)
        return x

# Training example
def train_network(model, X_train, y_train, epochs=100, lr=0.01):
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Convert to tensors
    X = torch.FloatTensor(X_train)
    y = torch.FloatTensor(y_train)
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass (automatic differentiation)
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Usage
model = NeuralNetwork([784, 256, 128, 10])  # Input, hidden, output layers
X_train = torch.randn(100, 784)  # 100 samples, 784 features
y_train = torch.randn(100, 10)   # 100 samples, 10 classes

train_network(model, X_train, y_train)

# PyTorch automatically handles backpropagation!
# You can also inspect gradients:
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'{name} gradient: {param.grad.norm().item()}')
```

**Why It Works**: PyTorch's autograd automatically computes gradients using chain rule, making backpropagation seamless.

---

## Q2: What are ensemble methods? Explain bagging vs. boosting.

**Answer**:

**Ensemble Methods**: Combine multiple models to improve performance.

### Bagging (Bootstrap Aggregating)

**How It Works**:
1. Create multiple bootstrap samples (random sampling with replacement)
2. Train independent model on each sample
3. Aggregate predictions (voting for classification, averaging for regression)

**Example**: Random Forest
- Each tree trained on different bootstrap sample
- Each split considers random subset of features
- Final prediction: majority vote

**Benefits**:
- Reduces variance (overfitting)
- Parallel training
- Works well with high-variance models (deep trees)

**Implementation**:
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging with decision trees
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,  # Use 80% of data for each sample
    bootstrap=True
)
bagging.fit(X_train, y_train)
```

### Boosting

**How It Works**:
1. Train models sequentially
2. Each model focuses on mistakes of previous models
3. Weight samples based on difficulty
4. Combine with weighted voting

**Example**: AdaBoost
1. Start with equal weights for all samples
2. Train weak learner
3. Increase weights for misclassified samples
4. Repeat, giving more weight to harder examples

**Benefits**:
- Reduces bias (underfitting)
- Often better accuracy than bagging
- Works well with weak learners

**Implementation**:
```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# AdaBoost
adaboost = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
adaboost.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
gb.fit(X_train, y_train)
```

### Comparison

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Weighting | Equal | Adaptive |
| Example | Random Forest | AdaBoost, XGBoost |
| Overfitting | Less prone | More prone |

---

## Q3: Explain gradient descent variants (SGD, Mini-batch, Adam).

**Answer**:

### Batch Gradient Descent

**How It Works**: Use entire dataset to compute gradient.

```python
for epoch in range(n_epochs):
    gradient = compute_gradient(X_train, y_train, weights)
    weights -= learning_rate * gradient
```

**Pros**: Stable convergence, accurate gradient  
**Cons**: Slow for large datasets, memory intensive

### Stochastic Gradient Descent (SGD)

**How It Works**: Use one sample at a time.

```python
for epoch in range(n_epochs):
    for i in range(n_samples):
        gradient = compute_gradient(X_train[i], y_train[i], weights)
        weights -= learning_rate * gradient
```

**Pros**: Fast, can escape local minima  
**Cons**: Noisy updates, unstable convergence

### Mini-batch Gradient Descent

**How It Works**: Use small batches (e.g., 32, 64, 128 samples).

```python
for epoch in range(n_epochs):
    for batch in get_batches(X_train, y_train, batch_size=32):
        X_batch, y_batch = batch
        gradient = compute_gradient(X_batch, y_batch, weights)
        weights -= learning_rate * gradient
```

**Pros**: Balance between speed and stability  
**Cons**: Requires tuning batch size

### Adam (Adaptive Moment Estimation)

**How It Works**: Combines momentum and adaptive learning rates.

**Algorithm**:
```python
# Initialize
m = 0  # First moment (momentum)
v = 0  # Second moment (RMSprop)
t = 0  # Time step

for epoch in range(n_epochs):
    for batch in get_batches(X_train, y_train, batch_size):
        t += 1
        gradient = compute_gradient(batch, weights)
        
        # Update biased first moment
        m = beta1 * m + (1 - beta1) * gradient
        
        # Update biased second moment
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update weights
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
```

**Typical values**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

**Why Adam is popular**: Adaptive learning rates per parameter, works well with default hyperparameters.

---

## Q4: What is batch normalization and why does it help?

**Answer**:

**How It Works**:

Normalize activations within each mini-batch to have mean 0 and variance 1.

**Algorithm**:
1. For each mini-batch:
$$
\mu_B = \frac{1}{m}\sum_{i=1}^m x_i
$$
$$
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2
$$
2. Normalize:
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
3. Scale and shift (learnable parameters):
$$
y_i = \gamma \hat{x}_i + \beta
$$

**Why It Helps**:
1. **Reduces internal covariate shift**: Stabilizes distribution of activations
2. **Allows higher learning rates**: Less sensitive to initialization
3. **Regularization effect**: Adds noise through batch statistics
4. **Faster convergence**: Smoother optimization landscape

**Implementation**:
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),  # Batch norm after linear layer
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

**During Inference**: Use running statistics (exponential moving average) instead of batch statistics.

---

## Q5: Explain the vanishing/exploding gradient problem.

**Answer**:

### Vanishing Gradients

**Problem**: Gradients become extremely small in early layers, preventing learning.

**Why It Happens**:
- Deep networks multiply many gradients via chain rule
- If gradients < 1, repeated multiplication → very small values
- Common with sigmoid/tanh activations (gradients max at 0.25)

**Example**:
$$
\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial a^{(L-1)}} \cdot ... \cdot \frac{\partial a^{(2)}}{\partial W^{(1)}}
$$

If each $\frac{\partial a^{(l)}}{\partial a^{(l-1)}} < 1$, product vanishes.

**Solutions**:
1. **ReLU activation**: Gradient is 1 for positive inputs
2. **Batch normalization**: Stabilizes gradients
3. **Residual connections** (ResNet): Skip connections allow gradient flow
4. **Better initialization**: Xavier/He initialization
5. **LSTM/GRU**: For RNNs, use gating mechanisms

### Exploding Gradients

**Problem**: Gradients become extremely large, causing unstable training.

**Why It Happens**: Repeated multiplication of gradients > 1

**Solutions**:
1. **Gradient clipping**: Cap gradient magnitude
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
2. **Lower learning rate**
3. **Batch normalization**

---

## Q6: Implement a confusion matrix and calculate metrics.

**Answer**:

**How It Works**: 2x2 matrix for binary classification showing true/false positives/negatives.

**Implementation**:
```python
import numpy as np

def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[tn, fp], [fn, tp]])

def calculate_metrics(cm):
    """Calculate all metrics from confusion matrix"""
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity
    }

# Example
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nMetrics:")
print(calculate_metrics(cm))
```

**Output**:
```
Confusion Matrix:
[[4 1]
 [1 4]]

Metrics:
{'accuracy': 0.8, 'precision': 0.8, 'recall': 0.8, 'f1_score': 0.8, 'specificity': 0.8}
```

---

## Q7: What is transfer learning and when to use it?

**Answer**:

**How It Works**: Use pre-trained model as starting point, fine-tune for your task.

**Typical Approach**:
1. Take model trained on large dataset (e.g., ImageNet)
2. Remove final layer(s)
3. Add new layers for your task
4. Fine-tune:
   - Option A: Freeze early layers, train only new layers
   - Option B: Train all layers with small learning rate

**Implementation**:
```python
import torch
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)

# Only final layer will be trained
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

**When to Use**:
✅ Small dataset (< 10k samples)  
✅ Similar domain (e.g., both are images)  
✅ Limited compute resources  
✅ Want faster convergence

**When NOT to Use**:
❌ Very different domains (text → images)  
❌ Huge dataset available  
❌ Very specific task with no similar pre-trained models

---

## Q8: Explain dropout and how it prevents overfitting.

**Answer**:

**How It Works**: Randomly "drop" (set to 0) neurons during training with probability $p$.

**Algorithm**:
```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    
    # Create mask: 1 with probability (1-p), 0 with probability p
    mask = (np.random.rand(*x.shape) > p).astype(float)
    
    # Scale to maintain expected value
    return x * mask / (1 - p)
```

**Why It Works**:
1. **Prevents co-adaptation**: Neurons can't rely on specific other neurons
2. **Ensemble effect**: Training many "thinned" networks, averaging at test time
3. **Regularization**: Adds noise, prevents overfitting

**During Training**: Randomly drop neurons  
**During Inference**: Use all neurons (no dropout)

**Implementation**:
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # Drop 50% of neurons
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),  # Drop 30% of neurons
    nn.Linear(256, 10)
)
```

**Typical values**: 0.2-0.5 for hidden layers, 0.5 for input layer

---

## Q9: What is the difference between L1 and L2 regularization?

**Answer**:

### L2 Regularization (Ridge)

**Formula**: Add squared magnitude of weights to loss
$$
L_{\text{total}} = L_{\text{data}} + \lambda \sum_{i} w_i^2
$$

**Gradient**: $\frac{\partial}{\partial w} = \frac{\partial L}{\partial w} + 2\lambda w$

**Effect**: Weights decay towards zero but rarely become exactly zero

**Implementation**:
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # alpha is λ
model.fit(X_train, y_train)
```

### L1 Regularization (Lasso)

**Formula**: Add absolute magnitude of weights
$$
L_{\text{total}} = L_{\text{data}} + \lambda \sum_{i} |w_i|
$$

**Gradient**: $\frac{\partial}{\partial w} = \frac{\partial L}{\partial w} + \lambda \cdot \text{sign}(w)$

**Effect**: Drives some weights to exactly zero (feature selection)

**Implementation**:
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)
```

### Comparison

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| Penalty | $\sum \|w_i\|$ | $\sum w_i^2$ |
| Feature Selection | Yes (sparse) | No |
| Solution | Non-differentiable at 0 | Differentiable everywhere |
| Use When | Many irrelevant features | All features contribute |

### Elastic Net

Combines both:
$$
L = L_{\text{data}} + \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2
$$

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=1.0, l1_ratio=0.5)  # 50% L1, 50% L2
model.fit(X_train, y_train)
```

---

## Q10: How do you handle missing data in ML?

**Answer**:

### 1. Deletion

**Listwise deletion**: Remove entire row if any value missing
```python
df_clean = df.dropna()
```
✅ Simple  
❌ Loses data, biased if not MCAR (Missing Completely At Random)

**Pairwise deletion**: Use available data for each calculation
✅ Retains more data  
❌ Different sample sizes for different calculations

### 2. Imputation

**Mean/Median/Mode**:
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)
```
✅ Simple, fast  
❌ Reduces variance, ignores relationships

**K-NN Imputation**:
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```
✅ Uses feature relationships  
❌ Computationally expensive

**Iterative Imputation** (MICE):
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=0)
X_imputed = imputer.fit_transform(X)
```
✅ Models relationships between features  
❌ Slow, can be unstable

### 3. Model-Based

**Use models that handle missing values**:
- XGBoost, LightGBM (built-in handling)
- Decision trees (can split on "missing" as category)

### 4. Create Indicator

Add binary feature indicating missingness:
```python
X['feature_missing'] = X['feature'].isnull().astype(int)
X['feature'].fillna(X['feature'].mean(), inplace=True)
```
✅ Preserves information about missingness  
❌ Increases dimensionality

**Choose based on**:
- Amount of missing data
- Missing mechanism (MCAR, MAR, MNAR)
- Computational resources
- Domain knowledge

