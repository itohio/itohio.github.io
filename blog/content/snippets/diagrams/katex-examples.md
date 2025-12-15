---
title: "KaTeX Math Examples and Tips"
date: 2024-12-12T20:30:00Z
draft: false
description: "Comprehensive guide to KaTeX math notation with examples and tips"
tags: ["katex", "math", "latex", "mathematics", "notation", "equations"]
category: "diagrams"
---

KaTeX renders mathematical notation beautifully in web pages. This guide covers common patterns, syntax, and tips for writing mathematical expressions.

## Use Case

Use KaTeX when you need to:
- Display mathematical formulas and equations
- Show scientific notation
- Write technical documentation with math
- Create educational content with equations
- Document algorithms with mathematical foundations

## Basic Syntax

### Inline Math

Use single dollar signs for inline math: `$x = y + z$` renders as $x = y + z$

### Display Math

Use double dollar signs for display math (centered, on its own line):

```markdown
$$
E = mc^2
$$
```

Renders as:

$$
E = mc^2
$$

## Common Mathematical Expressions

### Fractions

```markdown
$$
\frac{a}{b} \quad \frac{numerator}{denominator} \quad \frac{x^2 + 1}{x - 1}
$$
```

$$
\frac{a}{b} \quad \frac{numerator}{denominator} \quad \frac{x^2 + 1}{x - 1}
$$

### Superscripts and Subscripts

```markdown
$$
x^2 \quad x^{n+1} \quad x_i \quad x_{i+1} \quad x^{y^z}
$$
```

$$
x^2 \quad x^{n+1} \quad x_i \quad x_{i+1} \quad x^{y^z}
$$

### Roots

```markdown
$$
\sqrt{x} \quad \sqrt[n]{x} \quad \sqrt{x^2 + y^2}
$$
```

$$
\sqrt{x} \quad \sqrt[n]{x} \quad \sqrt{x^2 + y^2}
$$

### Sums and Products

```markdown
$$
\sum_{i=1}^{n} x_i \quad \prod_{i=1}^{n} x_i \quad \int_{a}^{b} f(x) dx
$$
```

$$
\sum_{i=1}^{n} x_i \quad \prod_{i=1}^{n} x_i \quad \int_{a}^{b} f(x) dx
$$

### Greek Letters

```markdown
$$
\alpha \quad \beta \quad \gamma \quad \delta \quad \epsilon \quad \theta \quad \lambda \quad \mu \quad \pi \quad \sigma \quad \phi \quad \omega
$$
```

$$
\alpha \quad \beta \quad \gamma \quad \delta \quad \epsilon \quad \theta \quad \lambda \quad \mu \quad \pi \quad \sigma \quad \phi \quad \omega
$$

**Capital Greek letters:**

```markdown
$$
\Alpha \quad \Beta \quad \Gamma \quad \Delta \quad \Theta \quad \Lambda \quad \Pi \quad \Sigma \quad \Phi \quad \Omega
$$
```

$$
\Alpha \quad \Beta \quad \Gamma \quad \Delta \quad \Theta \quad \Lambda \quad \Pi \quad \Sigma \quad \Phi \quad \Omega
$$

## Operators and Relations

### Comparison Operators

```markdown
$$
< \quad > \quad \leq \quad \geq \quad \neq \quad \approx \quad \equiv
$$
```

$$
< \quad > \quad \leq \quad \geq \quad \neq \quad \approx \quad \equiv
$$

### Set Operations

```markdown
$$
\in \quad \notin \quad \subset \quad \subseteq \quad \cup \quad \cap \quad \setminus
$$
```

$$
\in \quad \notin \quad \subset \quad \subseteq \quad \cup \quad \cap \quad \setminus
$$

### Logical Operators

```markdown
$$
\land \quad \lor \quad \neg \quad \implies \quad \iff \quad \forall \quad \exists
$$
```

$$
\land \quad \lor \quad \neg \quad \implies \quad \iff \quad \forall \quad \exists
$$

## Matrices and Vectors

### Matrices

```markdown
$$
\begin{pmatrix}
a & b \\\\
c & d
\end{pmatrix}
\quad
\begin{bmatrix}
1 & 2 & 3 \\\\
4 & 5 & 6 \\\\
7 & 8 & 9
\end{bmatrix}
$$
```

$$
\begin{pmatrix}
a & b \\\\
c & d
\end{pmatrix}
\quad
\begin{bmatrix}
1 & 2 & 3 \\\\
4 & 5 & 6 \\\\
7 & 8 & 9
\end{bmatrix}
$$

### More Matrix Examples

#### Determinant

```markdown
$$
\det(\mathbf{A}) = \begin{vmatrix}
a & b \\\\
c & d
\end{vmatrix} = ad - bc
$$
```

$$
\det(\mathbf{A}) = \begin{vmatrix}
a & b \\\\
c & d
\end{vmatrix} = ad - bc
$$

#### Matrix Types

```markdown
$$
\begin{pmatrix} a & b \\\\ c & d \end{pmatrix} \quad
\begin{bmatrix} a & b \\\\ c & d \end{bmatrix} \quad
\begin{Bmatrix} a & b \\\\ c & d \end{Bmatrix} \quad
\begin{vmatrix} a & b \\\\ c & d \end{vmatrix} \quad
\begin{Vmatrix} a & b \\\\ c & d \end{Vmatrix}
$$
```

$$
\begin{pmatrix} a & b \\\\ c & d \end{pmatrix} \quad
\begin{bmatrix} a & b \\\\ c & d \end{bmatrix} \quad
\begin{Bmatrix} a & b \\\\ c & d \end{Bmatrix} \quad
\begin{vmatrix} a & b \\\\ c & d \end{vmatrix} \quad
\begin{Vmatrix} a & b \\\\ c & d \end{Vmatrix}
$$

#### Large Matrices

```markdown
$$
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\\\
a_{21} & a_{22} & \cdots & a_{2n} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$
```

$$
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\\\
a_{21} & a_{22} & \cdots & a_{2n} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

#### Block Matrices

```markdown
$$
\begin{bmatrix}
\mathbf{A} & \mathbf{B} \\\\
\mathbf{C} & \mathbf{D}
\end{bmatrix} = \begin{bmatrix}
a_{11} & a_{12} & b_{11} & b_{12} \\\\
a_{21} & a_{22} & b_{21} & b_{22} \\\\
c_{11} & c_{12} & d_{11} & d_{12} \\\\
c_{21} & c_{22} & d_{21} & d_{22}
\end{bmatrix}
$$
```

$$
\begin{bmatrix}
\mathbf{A} & \mathbf{B} \\\\
\mathbf{C} & \mathbf{D}
\end{bmatrix} = \begin{bmatrix}
a_{11} & a_{12} & b_{11} & b_{12} \\\\
a_{21} & a_{22} & b_{21} & b_{22} \\\\
c_{11} & c_{12} & d_{11} & d_{12} \\\\
c_{21} & c_{22} & d_{21} & d_{22}
\end{bmatrix}
$$

### Vectors

```markdown
$$
\vec{v} = \begin{pmatrix} x \\\\ y \\\\ z \end{pmatrix} \quad \mathbf{v} = \begin{bmatrix} v_1 \\\\ v_2 \\\\ v_3 \end{bmatrix}
$$
```

$$
\vec{v} = \begin{pmatrix} x \\\\ y \\\\ z \end{pmatrix} \quad \mathbf{v} = \begin{bmatrix} v_1 \\\\ v_2 \\\\ v_3 \end{bmatrix}
$$

## Functions and Special Notation

### Common Functions

```markdown
$$
\sin(x) \quad \cos(x) \quad \tan(x) \quad \log(x) \quad \ln(x) \quad \exp(x)
$$
```

$$
\sin(x) \quad \cos(x) \quad \tan(x) \quad \log(x) \quad \ln(x) \quad \exp(x)
$$

### Limits

```markdown
$$
\lim_{x \to \infty} f(x) \quad \lim_{n \to 0} \frac{\sin(n)}{n} = 1
$$
```

$$
\lim_{x \to \infty} f(x) \quad \lim_{n \to 0} \frac{\sin(n)}{n} = 1
$$

### Derivatives

```markdown
$$
\frac{d}{dx}f(x) \quad f'(x) \quad \frac{\partial f}{\partial x} \quad \nabla f
$$
```

$$
\frac{d}{dx}f(x) \quad f'(x) \quad \frac{\partial f}{\partial x} \quad \nabla f
$$

## Aligned Equations

### Single Alignment

```markdown
$$
\begin{aligned}
x &= a + b \\\\
y &= c + d \\\\
z &= x + y
\end{aligned}
$$
```

$$
\begin{aligned}
x &= a + b \\\\
y &= c + d \\\\
z &= x + y
\end{aligned}
$$

### Multi-line Equations

```markdown
$$
\begin{align}
f(x) &= x^2 + 2x + 1 \\\\
     &= (x + 1)^2 \\\\
     &= x^2 + 2x + 1
\end{align}
$$
```

$$
\begin{align}
f(x) &= x^2 + 2x + 1 \\\\
     &= (x + 1)^2 \\\\
     &= x^2 + 2x + 1
\end{align}
$$

## Cases and Piecewise Functions

### Basic Cases

```markdown
$$
f(x) = \begin{cases}
x^2 & \text{if } x \geq 0 \\\\
-x^2 & \text{if } x < 0
\end{cases}
$$
```

$$
f(x) = \begin{cases}
x^2 & \text{if } x \geq 0 \\\\
-x^2 & \text{if } x < 0
\end{cases}
$$

### Conditional Expressions

```markdown
$$
a = \begin{cases}
2 & \text{if } b = 2 \\\\
4 & \text{if } b = 4 \\\\
6 & \text{if } b = 6 \\\\
0 & \text{otherwise}
\end{cases}
$$
```

$$
a = \begin{cases}
2 & \text{if } b = 2 \\\\
4 & \text{if } b = 4 \\\\
6 & \text{if } b = 6 \\\\
0 & \text{otherwise}
\end{cases}
$$

### Multiple Conditions

```markdown
$$
f(x) = \begin{cases}
x + 1 & \text{if } x < 0 \\\\
x^2 & \text{if } 0 \leq x < 1 \\\\
2x - 1 & \text{if } x \geq 1
\end{cases}
$$
```

$$
f(x) = \begin{cases}
x + 1 & \text{if } x < 0 \\\\
x^2 & \text{if } 0 \leq x < 1 \\\\
2x - 1 & \text{if } x \geq 1
\end{cases}
$$

### Conditional Matrix

```markdown
$$
\mathbf{A} = \begin{cases}
\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix} & \text{if } \det(\mathbf{B}) \neq 0 \\\\
\begin{bmatrix} 0 & 0 \\\\ 0 & 0 \end{bmatrix} & \text{otherwise}
\end{cases}
$$
```

$$
\mathbf{A} = \begin{cases}
\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix} & \text{if } \det(\mathbf{B}) \neq 0 \\\\
\begin{bmatrix} 0 & 0 \\\\ 0 & 0 \end{bmatrix} & \text{otherwise}
\end{cases}
$$

### Conditional with Logical Operators

```markdown
$$
y = \begin{cases}
x^2 & \text{if } x > 0 \text{ and } x \neq 1 \\\\
\log(x) & \text{if } x > 0 \text{ and } x = 1 \\\\
0 & \text{if } x \leq 0
\end{cases}
$$
```

$$
y = \begin{cases}
x^2 & \text{if } x > 0 \text{ and } x \neq 1 \\\\
\log(x) & \text{if } x > 0 \text{ and } x = 1 \\\\
0 & \text{if } x \leq 0
\end{cases}
$$

## Spacing and Formatting

### Manual Spacing

```markdown
$$
a\,b \quad a\;b \quad a\:b \quad a\!b \quad a\ b \quad a\quad b \quad a\qquad b
$$
```

$$
a\,b \quad a\;b \quad a\:b \quad a\!b \quad a\ b \quad a\quad b \quad a\qquad b
$$

### Text in Math

```markdown
$$
\text{for all } x \in \mathbb{R} \quad \text{where } n > 0
$$
```

$$
\text{for all } x \in \mathbb{R} \quad \text{where } n > 0
$$

## Number Sets

```markdown
$$
\mathbb{N} \quad \mathbb{Z} \quad \mathbb{Q} \quad \mathbb{R} \quad \mathbb{C}
$$
```

$$
\mathbb{N} \quad \mathbb{Z} \quad \mathbb{Q} \quad \mathbb{R} \quad \mathbb{C}
$$

## Practical Examples

### Example 1: Quadratic Formula

```markdown
$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$
```

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

### Example 2: Euler's Identity

```markdown
$$
e^{i\pi} + 1 = 0
$$
```

$$
e^{i\pi} + 1 = 0
$$

### Example 3: Bayes' Theorem

```markdown
$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$
```

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

### Example 4: Matrix Multiplication

```markdown
$$
\mathbf{C} = \mathbf{A} \mathbf{B} = \begin{bmatrix}
a_{11} & a_{12} \\\\
a_{21} & a_{22}
\end{bmatrix}
\begin{bmatrix}
b_{11} & b_{12} \\\\
b_{21} & b_{22}
\end{bmatrix}
$$
```

$$
\mathbf{C} = \mathbf{A} \mathbf{B} = \begin{bmatrix}
a_{11} & a_{12} \\\\
a_{21} & a_{22}
\end{bmatrix}
\begin{bmatrix}
b_{11} & b_{12} \\\\
b_{21} & b_{22}
\end{bmatrix}
$$

### Example 5: Gradient Descent Update

```markdown
$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)
$$
```

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)
$$

### Example 6: Neural Network Forward Pass

```markdown
$$
\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})
$$
$$

where $\sigma$ is the activation function, $\mathbf{W}$ is the weight matrix, $\mathbf{x}$ is the input vector, and $\mathbf{b}$ is the bias vector.

### Example 7: Loss Function

```markdown
$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]
$$
```

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]
$$

## Tips and Best Practices

### 1. Use Display Math for Important Equations

Display math (double `$$`) is better for:
- Standalone equations
- Multi-line expressions
- Complex formulas

Inline math (single `$`) is better for:
- Short expressions in text
- Variables mentioned in sentences
- Simple formulas

### 2. Escape Special Characters

In markdown, you may need to escape:
- `$` → `\$` (if not used for math)
- `_` → `\_` (in non-math contexts)
- `*` → `\*` (in non-math contexts)

### 3. Use Aligned Environments for Multi-line

```markdown
$$
\begin{aligned}
f(x) &= x^2 + 2x + 1 \\\\
     &= (x+1)^2
\end{aligned}
$$
```

Better than:

```markdown
$$
f(x) = x^2 + 2x + 1 = (x+1)^2
$$
```

### 4. Label Important Equations

While KaTeX doesn't support `\label` like LaTeX, you can add text labels:

```markdown
$$
\text{(1)} \quad E = mc^2
$$
```

### 5. Use Text Mode for Words

Always use `\text{}` for words in math mode:

```markdown
$$
\text{for } x \in \mathbb{R} \text{ and } y > 0
$$
```

Not:

```markdown
$$
for x \in \mathbb{R} and y > 0  \quad \text{(incorrect)}
$$
```

## Common Mistakes to Avoid

### Mistake 1: Mixing Math and Text

❌ **Wrong:**
```markdown
The value is $x$ where $x > 0$ and positive.
```

✅ **Correct:**
```markdown
The value is $x$ where $x > 0$ and is positive.
```

### Mistake 2: Forgetting Curly Braces

❌ **Wrong:**
```markdown
$x^10$  (renders as $x^10$)
```

✅ **Correct:**
```markdown
$x^{10}$  (renders as $x^{10}$)
```

### Mistake 3: Incorrect Spacing

❌ **Wrong:**
```markdown
$f(x)=x^2+1$  (no spacing)
```

✅ **Correct:**
```markdown
$f(x) = x^2 + 1$  (proper spacing)
```

## Advanced Examples

### Example: Fourier Transform

```markdown
$$
F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt
$$
```

$$
F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt
$$

### Example: Schrödinger Equation

```markdown
$$
i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat{H}\Psi(\mathbf{r},t)
$$
```

$$
i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat{H}\Psi(\mathbf{r},t)
$$

### Example: Einstein Field Equations

```markdown
$$
R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}
$$
```

$$
R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}
$$

## Resources

- [KaTeX Documentation](https://katex.org/docs/supported.html)
- [LaTeX Math Symbols](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols)
- [Detexify](http://detexify.kirelabs.org/) - Draw symbol to find LaTeX command

## Gotchas/Warnings

- ⚠️ **Dollar Signs**: In markdown, `$` starts math mode - escape with `\$` if needed
- ⚠️ **Curly Braces**: Always use `{}` for multi-character subscripts/superscripts
- ⚠️ **Text Mode**: Use `\text{}` for words, not raw text in math mode
- ⚠️ **Spacing**: Math mode removes spaces - use `\,`, `\;`, `\quad`, etc. for spacing
- ⚠️ **Alignment**: Use `aligned` or `align` environments for multi-line equations
- ⚠️ **Backslashes**: In markdown, you may need `\\` for line breaks in matrices

