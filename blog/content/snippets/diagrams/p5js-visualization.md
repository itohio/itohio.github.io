---
title: "P5.js Interactive Visualizations"
date: 2024-12-12T19:20:00Z
draft: false
description: "Create interactive visualizations and animations with P5.js"
tags: ["p5js", "visualization", "diagram", "interactive", "animation", "canvas"]
category: "diagrams"
---



P5.js is a JavaScript library for creative coding and interactive visualizations. Perfect for creating custom animations, data visualizations, and interactive diagrams that go beyond static charts.

## Use Case

Use P5.js when you need to:
- Create custom interactive visualizations
- Animate data or concepts
- Build interactive demos
- Visualize algorithms in action
- Create generative art or patterns

## Code

````markdown
```p5js
sketch.setup = function() {
  sketch.createCanvas(400, 300);
};

sketch.draw = function() {
  sketch.background(220);
  sketch.ellipse(sketch.mouseX, sketch.mouseY, 50, 50);
};
```
````

## Explanation

- `setup()` - Runs once at start, initialize canvas
- `draw()` - Runs continuously (60fps by default)
- `createCanvas(w, h)` - Create drawing canvas
- `background(color)` - Clear canvas with color
- `mouseX, mouseY` - Current mouse position
- Drawing functions: `ellipse()`, `rect()`, `line()`, etc.

## Examples

### Example 1: Simple Animation

````markdown
```p5js
let x = 0;

sketch.setup = function() {
  sketch.createCanvas(400, 200);
};

sketch.draw = function() {
  sketch.background(240);
  
  // Moving circle
  sketch.fill(100, 150, 250);
  sketch.ellipse(x, 100, 40, 40);
  
  x = x + 2;
  if (x > sketch.width) {
    x = 0;
  }
};
```
````

**Result:**

```p5js
let x = 0;

sketch.setup = function() {
  sketch.createCanvas(400, 200);
};

sketch.draw = function() {
  sketch.background(240);
  
  // Moving circle
  sketch.fill(100, 150, 250);
  sketch.ellipse(x, 100, 40, 40);
  
  x = x + 2;
  if (x > sketch.width) {
    x = 0;
  }
};
```

### Example 2: Data Visualization

````markdown
```p5js
let data = [45, 67, 89, 34, 78, 56, 90, 23];

sketch.setup = function() {
  sketch.createCanvas(400, 300);
  sketch.noLoop(); // Draw once
};

sketch.draw = function() {
  sketch.background(255);
  
  let barWidth = sketch.width / data.length;
  
  for (let i = 0; i < data.length; i++) {
    let barHeight = sketch.map(data[i], 0, 100, 0, sketch.height - 40);
    
    sketch.fill(100, 150, 250);
    sketch.rect(i * barWidth, sketch.height - barHeight, barWidth - 2, barHeight);
    
    // Labels
    sketch.fill(0);
    sketch.textAlign(sketch.CENTER);
    sketch.text(data[i], i * barWidth + barWidth/2, sketch.height - barHeight - 5);
  }
};
```
````

**Result:**

```p5js
let data = [45, 67, 89, 34, 78, 56, 90, 23];

sketch.setup = function() {
  sketch.createCanvas(400, 300);
  sketch.noLoop(); // Draw once
};

sketch.draw = function() {
  sketch.background(255);
  
  let barWidth = sketch.width / data.length;
  
  for (let i = 0; i < data.length; i++) {
    let barHeight = sketch.map(data[i], 0, 100, 0, sketch.height - 40);
    
    sketch.fill(100, 150, 250);
    sketch.rect(i * barWidth, sketch.height - barHeight, barWidth - 2, barHeight);
    
    // Labels
    sketch.fill(0);
    sketch.textAlign(sketch.CENTER);
    sketch.text(data[i], i * barWidth + barWidth/2, sketch.height - barHeight - 5);
  }
};
```

### Example 3: Interactive Particle System

````markdown
```p5js
let particles = [];

sketch.setup = function() {
  sketch.createCanvas(500, 400);
};

sketch.draw = function() {
  sketch.background(20, 20, 40, 25); // Trailing effect
  
  // Create new particle on mouse press
  if (sketch.mouseIsPressed) {
    particles.push({
      x: sketch.mouseX,
      y: sketch.mouseY,
      vx: sketch.random(-2, 2),
      vy: sketch.random(-2, 2),
      life: 255
    });
  }
  
  // Update and draw particles
  for (let i = particles.length - 1; i >= 0; i--) {
    let p = particles[i];
    
    p.x += p.vx;
    p.y += p.vy;
    p.life -= 2;
    
    sketch.fill(100, 200, 255, p.life);
    sketch.noStroke();
    sketch.ellipse(p.x, p.y, 8, 8);
    
    // Remove dead particles
    if (p.life < 0) {
      particles.splice(i, 1);
    }
  }
  
  // Instructions
  sketch.fill(255);
  sketch.text('Click and drag to create particles', 10, 20);
};
```
````

**Result:** (Click and drag to create particles)

```p5js
let particles = [];

sketch.setup = function() {
  sketch.createCanvas(500, 400);
};

sketch.draw = function() {
  sketch.background(20, 20, 40, 25); // Trailing effect
  
  // Create new particle on mouse press
  if (sketch.mouseIsPressed) {
    particles.push({
      x: sketch.mouseX,
      y: sketch.mouseY,
      vx: sketch.random(-2, 2),
      vy: sketch.random(-2, 2),
      life: 255
    });
  }
  
  // Update and draw particles
  for (let i = particles.length - 1; i >= 0; i--) {
    let p = particles[i];
    
    p.x += p.vx;
    p.y += p.vy;
    p.life -= 2;
    
    sketch.fill(100, 200, 255, p.life);
    sketch.noStroke();
    sketch.ellipse(p.x, p.y, 8, 8);
    
    // Remove dead particles
    if (p.life < 0) {
      particles.splice(i, 1);
    }
  }
  
  // Instructions
  sketch.fill(255);
  sketch.text('Click and drag to create particles', 10, 20);
};
```

### Example 4: Algorithm Visualization (Sorting)

````markdown
```p5js
let values = [];
let i = 0;
let j = 0;

sketch.setup = function() {
  sketch.createCanvas(600, 300);
  
  // Initialize random array
  for (let k = 0; k < 50; k++) {
    values[k] = sketch.random(sketch.height - 40);
  }
  
  sketch.frameRate(10); // Slow down to see sorting
};

sketch.draw = function() {
  sketch.background(240);
  
  // Bubble sort step
  if (i < values.length) {
    if (j < values.length - i - 1) {
      if (values[j] > values[j + 1]) {
        let temp = values[j];
        values[j] = values[j + 1];
        values[j + 1] = temp;
      }
      j++;
    } else {
      j = 0;
      i++;
    }
  }
  
  // Draw bars
  let barWidth = sketch.width / values.length;
  for (let k = 0; k < values.length; k++) {
    if (k === j) {
      sketch.fill(255, 100, 100); // Highlight current comparison
    } else if (k >= values.length - i) {
      sketch.fill(100, 255, 100); // Sorted portion
    } else {
      sketch.fill(150, 150, 250);
    }
    sketch.rect(k * barWidth, sketch.height - values[k], barWidth - 2, values[k]);
  }
  
  // Status
  sketch.fill(0);
  sketch.text('Bubble Sort Visualization', 10, 20);
};
```
````

**Result:** (Watch the sorting algorithm in action)

```p5js
let values = [];
let i = 0;
let j = 0;

sketch.setup = function() {
  sketch.createCanvas(600, 300);
  
  // Initialize random array
  for (let k = 0; k < 50; k++) {
    values[k] = sketch.random(sketch.height - 40);
  }
  
  sketch.frameRate(10); // Slow down to see sorting
};

sketch.draw = function() {
  sketch.background(240);
  
  // Bubble sort step
  if (i < values.length) {
    if (j < values.length - i - 1) {
      if (values[j] > values[j + 1]) {
        let temp = values[j];
        values[j] = values[j + 1];
        values[j + 1] = temp;
      }
      j++;
    } else {
      j = 0;
      i++;
    }
  }
  
  // Draw bars
  let barWidth = sketch.width / values.length;
  for (let k = 0; k < values.length; k++) {
    if (k === j) {
      sketch.fill(255, 100, 100); // Highlight current comparison
    } else if (k >= values.length - i) {
      sketch.fill(100, 255, 100); // Sorted portion
    } else {
      sketch.fill(150, 150, 250);
    }
    sketch.rect(k * barWidth, sketch.height - values[k], barWidth - 2, values[k]);
  }
  
  // Status
  sketch.fill(0);
  sketch.text('Bubble Sort Visualization', 10, 20);
};
```

### Example 5: Network Graph

````markdown
```p5js
let nodes = [];
let connections = [];

sketch.setup = function() {
  sketch.createCanvas(500, 500);
  
  // Create nodes
  for (let i = 0; i < 8; i++) {
    nodes.push({
      x: sketch.random(50, sketch.width - 50),
      y: sketch.random(50, sketch.height - 50),
      vx: sketch.random(-1, 1),
      vy: sketch.random(-1, 1)
    });
  }
  
  // Create random connections
  for (let i = 0; i < 12; i++) {
    let a = sketch.floor(sketch.random(nodes.length));
    let b = sketch.floor(sketch.random(nodes.length));
    if (a !== b) {
      connections.push([a, b]);
    }
  }
};

sketch.draw = function() {
  sketch.background(255);
  
  // Update node positions
  for (let node of nodes) {
    node.x += node.vx;
    node.y += node.vy;
    
    // Bounce off edges
    if (node.x < 30 || node.x > sketch.width - 30) node.vx *= -1;
    if (node.y < 30 || node.y > sketch.height - 30) node.vy *= -1;
  }
  
  // Draw connections
  sketch.stroke(200);
  sketch.strokeWeight(1);
  for (let conn of connections) {
    let nodeA = nodes[conn[0]];
    let nodeB = nodes[conn[1]];
    sketch.line(nodeA.x, nodeA.y, nodeB.x, nodeB.y);
  }
  
  // Draw nodes
  sketch.noStroke();
  sketch.fill(100, 150, 250);
  for (let node of nodes) {
    sketch.ellipse(node.x, node.y, 20, 20);
  }
};
```
````

**Result:** (Animated network graph)

```p5js
let nodes = [];
let connections = [];

sketch.setup = function() {
  sketch.createCanvas(500, 500);
  
  // Create nodes
  for (let i = 0; i < 8; i++) {
    nodes.push({
      x: sketch.random(50, sketch.width - 50),
      y: sketch.random(50, sketch.height - 50),
      vx: sketch.random(-1, 1),
      vy: sketch.random(-1, 1)
    });
  }
  
  // Create random connections
  for (let i = 0; i < 12; i++) {
    let a = sketch.floor(sketch.random(nodes.length));
    let b = sketch.floor(sketch.random(nodes.length));
    if (a !== b) {
      connections.push([a, b]);
    }
  }
};

sketch.draw = function() {
  sketch.background(255);
  
  // Update node positions
  for (let node of nodes) {
    node.x += node.vx;
    node.y += node.vy;
    
    // Bounce off edges
    if (node.x < 30 || node.x > sketch.width - 30) node.vx *= -1;
    if (node.y < 30 || node.y > sketch.height - 30) node.vy *= -1;
  }
  
  // Draw connections
  sketch.stroke(200);
  sketch.strokeWeight(1);
  for (let conn of connections) {
    let nodeA = nodes[conn[0]];
    let nodeB = nodes[conn[1]];
    sketch.line(nodeA.x, nodeA.y, nodeB.x, nodeB.y);
  }
  
  // Draw nodes
  sketch.noStroke();
  sketch.fill(100, 150, 250);
  for (let node of nodes) {
    sketch.ellipse(node.x, node.y, 20, 20);
  }
};
```

## Notes

- P5.js sketches are interactive by default
- Use `sketch.` prefix for all P5.js functions
- `frameRate()` controls animation speed
- `noLoop()` for static drawings
- Mouse/keyboard events: `mousePressed()`, `keyPressed()`, etc.

## Gotchas/Warnings

- ⚠️ **Performance**: Complex animations can be slow - optimize draw loop
- ⚠️ **Canvas size**: Large canvases impact performance
- ⚠️ **Memory**: Clean up arrays and objects to prevent memory leaks
- ⚠️ **Syntax**: Must use `sketch.` prefix for all P5.js functions in Hugo