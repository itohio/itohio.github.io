---
title: "Tensor Mathematics & Backpropagation"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "tensors", "backpropagation", "mathematics", "deep-learning"]
---


Tensor mathematics fundamentals and backpropagation theory with detailed mathematical derivations.

---

## Tensor Basics

### What is a Tensor?

A tensor is a generalization of scalars, vectors, and matrices to higher dimensions:

- **Scalar** (0D tensor): $x \in \mathbb{R}$
- **Vector** (1D tensor): $\mathbf{x} \in \mathbb{R}^n$
- **Matrix** (2D tensor): $\mathbf{X} \in \mathbb{R}^{m \times n}$
- **3D Tensor**: $\mathcal{X} \in \mathbb{R}^{m \times n \times p}$
- **nD Tensor**: $\mathcal{X} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_n}$

### Tensor Operations

#### Element-wise Operations

**Addition:**
$$
(\mathbf{A} + \mathbf{B})_{ij} = A_{ij} + B_{ij}
$$

**Multiplication (Hadamard product):**
$$
(\mathbf{A} \odot \mathbf{B})_{ij} = A_{ij} \cdot B_{ij}
$$

#### Matrix Multiplication

$$
(\mathbf{AB})_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
$$

#### Tensor Contraction

Generalization of matrix multiplication to higher dimensions:

$$
\mathcal{C}_{i_1...i_m,k_1...k_p} = \sum_{j_1,...,j_n} \mathcal{A}_{i_1...i_m,j_1...j_n} \mathcal{B}_{j_1...j_n,k_1...k_p}
$$

---

## Gradients and Derivatives

### Scalar Derivative

$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

### Vector Gradient

For $f: \mathbb{R}^n \to \mathbb{R}$:

$$
\nabla f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

### Jacobian Matrix

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$
\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

### Hessian Matrix

Second-order partial derivatives:

$$
\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

---

## Chain Rule for Backpropagation

### Univariate Chain Rule

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

### Multivariate Chain Rule

For $z = f(y_1, ..., y_m)$ where $y_i = g_i(x_1, ..., x_n)$:

$$
\frac{\partial z}{\partial x_j} = \sum_{i=1}^{m} \frac{\partial z}{\partial y_i} \frac{\partial y_i}{\partial x_j}
$$

### Vector Chain Rule

$$
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}}
$$

---

## Backpropagation Algorithm

### Forward Pass

Given input $\mathbf{x}$, compute activations layer by layer:

**Layer $l$:**
$$
\begin{aligned}
\mathbf{z}^{[l]} &= \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \\
\mathbf{a}^{[l]} &= \sigma(\mathbf{z}^{[l]})
\end{aligned}
$$

Where:
- $\mathbf{W}^{[l]}$: Weight matrix for layer $l$
- $\mathbf{b}^{[l]}$: Bias vector for layer $l$
- $\sigma$: Activation function
- $\mathbf{a}^{[0]} = \mathbf{x}$: Input

### Backward Pass

Compute gradients layer by layer (from output to input):

**Output Layer $L$:**
$$
\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot \sigma'(\mathbf{z}^{[L]})
$$

**Hidden Layer $l$:**
$$
\delta^{[l]} = (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \odot \sigma'(\mathbf{z}^{[l]})
$$

**Gradients:**
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} &= \delta^{[l]} (\mathbf{a}^{[l-1]})^T \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} &= \delta^{[l]}
\end{aligned}
$$

---

## Activation Functions and Derivatives

### Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Derivative:**
$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

### Tanh

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**Derivative:**
$$
\tanh'(x) = 1 - \tanh^2(x)
$$

### ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$

**Derivative:**
$$
\text{ReLU}'(x) = \begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$

### Leaky ReLU

$$
\text{LeakyReLU}(x) = \max(\alpha x, x), \quad \alpha \in (0, 1)
$$

**Derivative:**
$$
\text{LeakyReLU}'(x) = \begin{cases}
1 & \text{if } x > 0 \\
\alpha & \text{if } x \leq 0
\end{cases}
$$

### Softmax

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Jacobian:**
$$
\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \text{softmax}(\mathbf{z})_i (\delta_{ij} - \text{softmax}(\mathbf{z})_j)
$$

---

## Cost Functions

### Mean Squared Error (MSE)

$$
\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Gradient:**
$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)
$$

### Binary Cross-Entropy

$$
\mathcal{L}(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

**Gradient:**
$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}
$$

### Categorical Cross-Entropy

$$
\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)
$$

**Gradient:**
$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}
$$

### Softmax + Cross-Entropy (Combined)

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i
$$

This simplification is why softmax and cross-entropy are often combined!

---

## Optimization Algorithms

### Gradient Descent

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

Where $\eta$ is the learning rate.

### Stochastic Gradient Descent (SGD)

Update using single sample or mini-batch:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_i(\theta_t)
$$

### Momentum

$$
\begin{aligned}
\mathbf{v}_t &= \beta \mathbf{v}_{t-1} + (1-\beta) \nabla_\theta \mathcal{L}(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta \mathbf{v}_t
\end{aligned}
$$

Typical: $\beta = 0.9$

### RMSprop

$$
\begin{aligned}
\mathbf{s}_t &= \beta \mathbf{s}_{t-1} + (1-\beta) (\nabla_\theta \mathcal{L}(\theta_t))^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \nabla_\theta \mathcal{L}(\theta_t)
\end{aligned}
$$

### Adam (Adaptive Moment Estimation)

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}(\theta_t) \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}(\theta_t))^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1-\beta_1^t} \\
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t
\end{aligned}
$$

Typical: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

## Batch Normalization

### Forward Pass

$$
\begin{aligned}
\mu_B &= \frac{1}{m} \sum_{i=1}^{m} x_i \\
\sigma_B^2 &= \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2 \\
\hat{x}_i &= \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\
y_i &= \gamma \hat{x}_i + \beta
\end{aligned}
$$

### Backward Pass

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \gamma} &= \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i \\
\frac{\partial \mathcal{L}}{\partial \beta} &= \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \\
\frac{\partial \mathcal{L}}{\partial \hat{x}_i} &= \frac{\partial \mathcal{L}}{\partial y_i} \gamma \\
\frac{\partial \mathcal{L}}{\partial \sigma_B^2} &= \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} (x_i - \mu_B) \cdot \frac{-1}{2}(\sigma_B^2 + \epsilon)^{-3/2} \\
\frac{\partial \mathcal{L}}{\partial \mu_B} &= \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \frac{-1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial \mathcal{L}}{\partial \sigma_B^2} \frac{-2}{m} \sum_{i=1}^{m} (x_i - \mu_B) \\
\frac{\partial \mathcal{L}}{\partial x_i} &= \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial \mathcal{L}}{\partial \sigma_B^2} \frac{2(x_i - \mu_B)}{m} + \frac{\partial \mathcal{L}}{\partial \mu_B} \frac{1}{m}
\end{aligned}
$$

---

## Dropout

### Forward Pass (Training)

$$
\mathbf{r} \sim \text{Bernoulli}(p), \quad \tilde{\mathbf{h}} = \mathbf{r} \odot \mathbf{h}
$$

### Forward Pass (Inference)

$$
\tilde{\mathbf{h}} = p \mathbf{h}
$$

### Backward Pass

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}} = \mathbf{r} \odot \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{h}}}
$$

---

## Convolutional Layer Backpropagation

### Forward Pass

$$
y_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} w_{m,n} \cdot x_{i+m,j+n} + b
$$

### Backward Pass

**Gradient w.r.t. weights:**
$$
\frac{\partial \mathcal{L}}{\partial w_{m,n}} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial y_{i,j}} \cdot x_{i+m,j+n}
$$

**Gradient w.r.t. input:**
$$
\frac{\partial \mathcal{L}}{\partial x_{i,j}} = \sum_{m,n} \frac{\partial \mathcal{L}}{\partial y_{i-m,j-n}} \cdot w_{m,n}
$$

---

## Example: Full Backprop Through 2-Layer Network

### Network Architecture

$$
\begin{aligned}
\mathbf{z}^{[1]} &= \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} \\
\mathbf{a}^{[1]} &= \text{ReLU}(\mathbf{z}^{[1]}) \\
\mathbf{z}^{[2]} &= \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]} \\
\hat{\mathbf{y}} &= \text{softmax}(\mathbf{z}^{[2]}) \\
\mathcal{L} &= -\sum_i y_i \log(\hat{y}_i)
\end{aligned}
$$

### Backpropagation Steps

**Step 1: Output gradient**
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}} = \hat{\mathbf{y}} - \mathbf{y}
$$

**Step 2: Layer 2 gradients**
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}} &= \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}} (\mathbf{a}^{[1]})^T \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[2]}} &= \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}}
\end{aligned}
$$

**Step 3: Backprop to layer 1**
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} = (\mathbf{W}^{[2]})^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}}
$$

**Step 4: Through ReLU**
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} \odot \mathbb{1}[\mathbf{z}^{[1]} > 0]
$$

**Step 5: Layer 1 gradients**
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} &= \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}} \mathbf{x}^T \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}} &= \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}}
\end{aligned}
$$

---