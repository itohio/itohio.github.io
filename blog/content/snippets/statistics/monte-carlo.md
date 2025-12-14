---
title: "Monte Carlo Methods"
date: 2024-12-12
draft: false
description: "Simulation and numerical integration"
tags: ["probability", "monte-carlo", "simulation", "numerical-methods"]
---

Interactive visualization of Monte Carlo methods for solving complex problems through random sampling.

---

## Principle

Use random sampling to solve deterministic or stochastic problems.

---

## Estimating π - Interactive Animation

```p5js
let points = [];
let inside = 0;
let total = 0;
let piEstimate = 0;
let running = true;

sketch.setup = function() {
  sketch.createCanvas(600, 650);
  
  // Control buttons
  let startBtn = sketch.createButton('Start/Resume');
  startBtn.position(20, 670);
  startBtn.mousePressed(() => { running = true; sketch.loop(); });
  
  let pauseBtn = sketch.createButton('Pause');
  pauseBtn.position(130, 670);
  pauseBtn.mousePressed(() => { running = false; sketch.noLoop(); });
  
  let resetBtn = sketch.createButton('Reset');
  resetBtn.position(210, 670);
  resetBtn.mousePressed(reset);
  
  sketch.frameRate(60);
}

function reset() {
  points = [];
  inside = 0;
  total = 0;
  piEstimate = 0;
  running = true;
  sketch.loop();
}

sketch.draw = function() {
  sketch.background(255);
  
  if (running) {
    // Add new points each frame
    for (let i = 0; i < 10; i++) {
      let x = sketch.random();
      let y = sketch.random();
      let isInside = (x * x + y * y) <= 1;
      
      points.push({ x, y, isInside });
      total++;
      if (isInside) inside++;
    }
    
    piEstimate = 4 * (inside / total);
    
    // Limit points array size for performance
    if (points.length > 5000) {
      points = points.slice(-5000);
    }
  }
  
  // Draw square and circle
  const size = 500;
  const margin = 50;
  
  // Square
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.noFill();
  sketch.rect(margin, margin, size, size);
  
  // Quarter circle
  sketch.stroke(0, 0, 255);
  sketch.strokeWeight(2);
  sketch.noFill();
  sketch.arc(margin, margin, size * 2, size * 2, 0, sketch.HALF_PI);
  
  // Draw points
  sketch.noStroke();
  for (let p of points) {
    if (p.isInside) {
      sketch.fill(0, 255, 0, 150);
    } else {
      sketch.fill(255, 0, 0, 150);
    }
    let px = margin + p.x * size;
    let py = margin + p.y * size;
    sketch.circle(px, py, 3);
  }
  
  // Statistics
  sketch.fill(0);
  sketch.textSize(16);
  sketch.textAlign(sketch.LEFT);
  sketch.text(`Total points: ${total}`, margin, margin + size + 30);
  sketch.text(`Inside circle: ${inside}`, margin, margin + size + 50);
  sketch.text(`Outside circle: ${total - inside}`, margin, margin + size + 70);
  
  // Pi estimate
  sketch.fill(200, 0, 0);
  sketch.textSize(20);
  sketch.text(`π estimate: ${piEstimate.toFixed(6)}`, margin, margin + size + 100);
  sketch.text(`π actual:   ${sketch.PI.toFixed(6)}`, margin, margin + size + 125);
  sketch.text(`Error:      ${sketch.abs(piEstimate - sketch.PI).toFixed(6)}`, margin, margin + size + 150);
  
  // Formula explanation
  sketch.fill(100);
  sketch.textSize(14);
  sketch.text('π ≈ 4 × (points inside / total points)', margin + 250, margin + size + 50);
  
  // Accuracy info
  sketch.fill(0, 0, 200);
  sketch.textSize(12);
  sketch.text('More points = better accuracy!', margin + 250, margin + size + 80);
}
```

---

## Integration Concept

Monte Carlo integration works by randomly sampling points and determining the ratio that fall under the curve.

```mermaid
graph TD
    A[Start: Define function f(x) and bounds a, b] --> B[Generate N random points in rectangle]
    B --> C{Is point under curve?}
    C -->|Yes| D[Count as inside]
    C -->|No| E[Count as outside]
    D --> F{More samples?}
    E --> F
    F -->|Yes| B
    F -->|No| G[Calculate: Area × inside/total]
    G --> H[Result: Integral estimate]
    
    style A fill:#e1f5ff
    style H fill:#c8e6c9
    style C fill:#fff9c4
```

**Algorithm:**

1. Define integration bounds $[a, b]$ and maximum function value $M$
2. Generate random points $(x, y)$ where $x \in [a, b]$, $y \in [0, M]$
3. Check if $y \leq f(x)$ (point is under curve)
4. Estimate: $\int_a^b f(x)dx \approx (b-a) \times M \times \frac{\text{points under curve}}{\text{total points}}$

**Convergence:** Error decreases as $O(1/\sqrt{N})$ where $N$ is the number of samples.

## Estimating π

```python
import numpy as np

def estimate_pi(n_samples=1000000):
    """Estimate π using Monte Carlo"""
    # Random points in [0,1] x [0,1]
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    
    # Count points inside quarter circle
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.mean(inside)
    
    return pi_estimate

print(f"π ≈ {estimate_pi():.6f}")
```

## Numerical Integration

$$
\int_a^b f(x) dx \approx \frac{b-a}{N} \sum_{i=1}^N f(x_i)
$$

```python
def monte_carlo_integrate(f, a, b, n_samples=100000):
    """Integrate f from a to b using Monte Carlo"""
    x = np.random.uniform(a, b, n_samples)
    return (b - a) * np.mean(f(x))

# Example: integrate x^2 from 0 to 1
result = monte_carlo_integrate(lambda x: x**2, 0, 1)
print(f"Integral ≈ {result:.6f} (exact: 0.333333)")
```

## Applications

- Option pricing (finance)
- Risk analysis
- Bayesian inference (MCMC)
- Physics simulations

## Further Reading

- [Monte Carlo Method - Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_method)

