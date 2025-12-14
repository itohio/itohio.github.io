---
title: "Denavit-Hartenberg (DH) Parameters"
date: 2024-12-13
draft: false
category: "kinematics"
tags: ["robotics", "kinematics", "dh-parameters", "forward-kinematics"]
---

Denavit-Hartenberg convention for systematically describing robot manipulator kinematics using 4 parameters per joint.

## Overview

**Denavit-Hartenberg (DH) parameters** provide a standardized way to describe the kinematic chain of a robot manipulator using only 4 parameters per joint.

**Purpose:**
- Systematic representation of serial manipulators
- Standardized coordinate frame assignment
- Compact kinematic description
- Foundation for forward and inverse kinematics

---

## The Four DH Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| **Link Length** | $a_i$ | Distance along $x_i$ from $z_{i-1}$ to $z_i$ |
| **Link Twist** | $\alpha_i$ | Angle about $x_i$ from $z_{i-1}$ to $z_i$ |
| **Link Offset** | $d_i$ | Distance along $z_{i-1}$ from $x_{i-1}$ to $x_i$ |
| **Joint Angle** | $\theta_i$ | Angle about $z_{i-1}$ from $x_{i-1}$ to $x_i$ |

### Joint Variables

- **Revolute joint**: $\theta_i$ varies (joint variable), $d_i$ is constant
- **Prismatic joint**: $d_i$ varies (joint variable), $\theta_i$ is constant

---

## DH Transformation Matrix

### Interactive DH Parameter Visualization

```p5js
let thetaSlider, dSlider, aSlider, alphaSlider;

sketch.setup = function() {
  sketch.createCanvas(800, 700);
  
  // Create sliders
  thetaSlider = sketch.createSlider(0, sketch.TWO_PI, sketch.PI/4, 0.01);
  thetaSlider.position(150, 720);
  thetaSlider.style('width', '150px');
  
  dSlider = sketch.createSlider(0, 100, 50, 1);
  dSlider.position(150, 750);
  dSlider.style('width', '150px');
  
  aSlider = sketch.createSlider(0, 150, 100, 1);
  aSlider.position(500, 720);
  aSlider.style('width', '150px');
  
  alphaSlider = sketch.createSlider(0, sketch.TWO_PI, sketch.PI/2, 0.01);
  alphaSlider.position(500, 750);
  alphaSlider.style('width', '150px');
}

sketch.draw = function() {
  sketch.background(255);
  
  let theta = thetaSlider.value();
  let d = dSlider.value();
  let a = aSlider.value();
  let alpha = alphaSlider.value();
  
  // Title
  sketch.fill(0);
  sketch.textSize(18);
  sketch.textAlign(sketch.CENTER);
  sketch.text('DH Parameters Interactive Demo', sketch.width/2, 25);
  
  sketch.push();
  sketch.translate(sketch.width/2, sketch.height/2 - 50);
  sketch.scale(1.5);
  
  // Draw previous frame (i-1)
  drawFrame(0, 0, 0, 'Frame i-1', [200, 0, 0]);
  
  // Step 1: Rotate about z by theta
  sketch.push();
  sketch.rotate(theta);
  sketch.stroke(100, 100, 255, 100);
  sketch.strokeWeight(1);
  sketch.noFill();
  sketch.arc(0, 0, 40, 40, 0, theta);
  sketch.pop();
  
  // Step 2: Translate along z by d
  sketch.stroke(0, 200, 0);
  sketch.strokeWeight(2);
  sketch.line(0, 0, 0, -d);
  
  // Apply transformations
  sketch.rotate(theta);
  sketch.translate(0, -d);
  
  // Step 3: Translate along x by a
  sketch.stroke(255, 0, 0);
  sketch.strokeWeight(2);
  sketch.line(0, 0, a, 0);
  
  sketch.translate(a, 0);
  
  // Step 4: Rotate about x by alpha (shown as rotation in plane)
  sketch.push();
  sketch.stroke(100, 255, 100, 100);
  sketch.strokeWeight(1);
  sketch.noFill();
  sketch.arc(0, 0, 30, 30, -alpha, 0);
  sketch.pop();
  
  sketch.rotate(-alpha);
  
  // Draw new frame (i)
  drawFrame(0, 0, 0, 'Frame i', [0, 0, 200]);
  
  sketch.pop();
  
  // Parameter labels
  sketch.fill(0);
  sketch.textSize(14);
  sketch.textAlign(sketch.LEFT);
  sketch.text(`θ (theta) = ${(theta * 180 / sketch.PI).toFixed(1)}°`, 20, 735);
  sketch.text(`d (offset) = ${d.toFixed(0)}`, 20, 765);
  sketch.text(`a (length) = ${a.toFixed(0)}`, 370, 735);
  sketch.text(`α (alpha) = ${(alpha * 180 / sketch.PI).toFixed(1)}°`, 370, 765);
  
  // Legend
  sketch.textSize(12);
  sketch.fill(200, 0, 0);
  sketch.text('Red: Previous frame (i-1)', 20, 650);
  sketch.fill(0, 0, 200);
  sketch.text('Blue: Current frame (i)', 20, 670);
  sketch.fill(0);
  sketch.text('1. Rotate θ about z  2. Translate d along z  3. Translate a along x  4. Rotate α about x', 20, 690);
}

function drawFrame(x, y, angle, label, color) {
  sketch.push();
  sketch.translate(x, y);
  sketch.rotate(angle);
  
  // X axis (red component)
  sketch.stroke(color[0], 0, 0);
  sketch.strokeWeight(3);
  sketch.line(0, 0, 50, 0);
  sketch.fill(color[0], 0, 0);
  sketch.noStroke();
  sketch.triangle(50, 0, 45, -3, 45, 3);
  sketch.textSize(10);
  sketch.text('x', 55, 5);
  
  // Y axis (green component)
  sketch.stroke(0, color[1], 0);
  sketch.strokeWeight(3);
  sketch.line(0, 0, 0, -50);
  sketch.fill(0, color[1], 0);
  sketch.noStroke();
  sketch.triangle(0, -50, -3, -45, 3, -45);
  sketch.textSize(10);
  sketch.text('y', 5, -55);
  
  // Z axis (pointing out - shown as circle)
  sketch.stroke(0, 0, color[2]);
  sketch.strokeWeight(2);
  sketch.noFill();
  sketch.circle(0, 0, 10);
  sketch.fill(0, 0, color[2]);
  sketch.noStroke();
  sketch.textSize(10);
  sketch.text('z⊙', -15, -15);
  
  // Label
  sketch.fill(color[0], color[1], color[2]);
  sketch.textSize(11);
  sketch.text(label, -30, 20);
  
  sketch.pop();
}
```

---

### Classic DH Convention (Craig)

Transformation from frame $i-1$ to frame $i$:

$$
T_i^{i-1} = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\\\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\\\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\\\
0 & 0 & 0 & 1
\end{bmatrix}
$$

**Transformation Sequence:**

1. Rotate about $z_{i-1}$ by $\theta_i$
2. Translate along $z_{i-1}$ by $d_i$
3. Translate along $x_i$ by $a_i$
4. Rotate about $x_i$ by $\alpha_i$

### Python Implementation

```python
import numpy as np

def dh_matrix(theta, d, a, alpha):
    """
    Compute DH transformation matrix (Classic convention)
    
    Args:
        theta: Joint angle (radians)
        d: Link offset
        a: Link length
        alpha: Link twist (radians)
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    T = np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])
    
    return T

# Example: Single joint
theta = np.pi/4  # 45 degrees
d = 0.1
a = 0.5
alpha = np.pi/2  # 90 degrees

T = dh_matrix(theta, d, a, alpha)
print("Transformation matrix:")
print(T)
```

---

## Frame Assignment Rules

### Classic DH Rules

1. **$z_i$ axis**: Along the axis of joint $i+1$
2. **$x_i$ axis**: Common normal between $z_{i-1}$ and $z_i$
3. **$y_i$ axis**: Complete right-hand coordinate system
4. **Origin**: Intersection of $x_i$ and $z_i$

### Step-by-Step Procedure

```text
Step 1: Identify joint axes
   - Mark the axis of rotation (revolute) or translation (prismatic)
   - These become the z-axes

Step 2: Establish base frame
   - z₀ along joint 1 axis
   - x₀ and y₀ chosen for convenience (usually align with world frame)

Step 3: For each joint i (i = 1 to n):
   a. Draw z_i along joint i+1 axis
   b. Find common normal between z_{i-1} and z_i
   c. Place x_i along this common normal
      - If z_{i-1} and z_i intersect: x_i perpendicular to both
      - If z_{i-1} and z_i are parallel: x_i arbitrary (choose convenient)
   d. Place origin at intersection of x_i and z_i
   e. Complete frame with y_i (right-hand rule)

Step 4: Measure DH parameters
   a. a_i: distance along x_i from z_{i-1} to z_i
   b. α_i: angle about x_i from z_{i-1} to z_i (right-hand rule)
   c. d_i: distance along z_{i-1} from x_{i-1} to x_i
   d. θ_i: angle about z_{i-1} from x_{i-1} to x_i (right-hand rule)
```

---

## Modified DH Convention (Khalil)

Alternative convention with frame attached to current joint instead of next joint.

### Modified DH Matrix

```python
def modified_dh_matrix(theta, d, a, alpha):
    """
    Modified DH transformation (Khalil convention)
    
    Differences from classic:
    - Frame i attached to joint i (not i+1)
    - Different parameter interpretation
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    T = np.array([
        [ct,    -st,    0,   a    ],
        [st*ca, ct*ca, -sa, -d*sa],
        [st*sa, ct*sa,  ca,  d*ca],
        [0,     0,      0,   1    ]
    ])
    
    return T
```

### Classic vs Modified DH

| Aspect | Classic DH | Modified DH |
|--------|-----------|-------------|
| **Frame attachment** | Frame $i$ at joint $i+1$ | Frame $i$ at joint $i$ |
| **Parameter order** | Rot($z$) → Trans($z$) → Trans($x$) → Rot($x$) | Trans($x$) → Rot($x$) → Trans($z$) → Rot($z$) |
| **Common normal** | From $z_{i-1}$ to $z_i$ | From $z_i$ to $z_{i+1}$ |
| **Usage** | More common in textbooks | Used in some software (Khalil) |

---

## Forward Kinematics

### Computing End-Effector Pose

```python
def forward_kinematics(dh_params):
    """
    Compute forward kinematics from DH parameters
    
    Args:
        dh_params: List of dicts with keys 'theta', 'd', 'a', 'alpha'
    
    Returns:
        4x4 transformation matrix from base to end-effector
    """
    T = np.eye(4)
    
    for params in dh_params:
        Ti = dh_matrix(**params)
        T = T @ Ti
    
    return T

# Example: 2-link planar arm
dh_params = [
    {'theta': np.pi/4, 'd': 0, 'a': 1.0, 'alpha': 0},
    {'theta': np.pi/3, 'd': 0, 'a': 0.8, 'alpha': 0}
]

T = forward_kinematics(dh_params)
print("End-effector position:", T[:3, 3])
print("End-effector orientation:")
print(T[:3, :3])
```

### Extract Position and Orientation

```python
def extract_pose(T):
    """Extract position and orientation from transformation matrix"""
    # Position
    position = T[:3, 3]
    
    # Orientation (rotation matrix)
    rotation = T[:3, :3]
    
    # Convert to Euler angles (ZYX convention)
    import scipy.spatial.transform as tf
    r = tf.Rotation.from_matrix(rotation)
    euler_angles = r.as_euler('zyx', degrees=False)
    
    return {
        'position': position,
        'rotation_matrix': rotation,
        'euler_angles': euler_angles
    }

pose = extract_pose(T)
print("Position:", pose['position'])
print("Euler angles (ZYX):", np.degrees(pose['euler_angles']), "degrees")
```

---

## Common Mistakes

### 1. Incorrect Frame Assignment

```text
❌ BAD: z-axis not along joint axis
   - Arbitrary frame placement
   - Doesn't follow DH convention

✅ GOOD: z-axis strictly along joint rotation/translation
   - Follow DH rules systematically
   - Verify each frame placement
```

### 2. Wrong Transformation Order

```python
# ❌ BAD: Incorrect order
T = Trans(x, a) @ Rot(z, theta) @ Trans(z, d) @ Rot(x, alpha)

# ✅ GOOD: Correct DH order (Classic)
T = Rot(z, theta) @ Trans(z, d) @ Trans(x, a) @ Rot(x, alpha)
```

### 3. Angle Units

```python
# ❌ BAD: Mixing degrees and radians
theta_deg = 45
T = dh_matrix(theta_deg, d, a, alpha)  # Wrong!

# ✅ GOOD: Always use radians
theta_rad = np.radians(45)
T = dh_matrix(theta_rad, d, a, alpha)
```

### 4. Sign Conventions

```text
❌ BAD: Inconsistent angle measurement
   - Mixing left-hand and right-hand rules
   - Arbitrary positive directions

✅ GOOD: Consistent right-hand rule
   - All rotations follow right-hand rule
   - Document positive directions clearly
```

---

## Validation

### Test with Known Configurations

```python
def validate_dh(robot, test_cases):
    """Validate DH parameters with known configurations"""
    
    for test in test_cases:
        joints = test['joints']
        expected = test['expected_position']
        
        # Compute FK
        T = robot.forward_kinematics(*joints)
        actual = T[:3, 3]
        
        # Check error
        error = np.linalg.norm(actual - expected)
        
        print(f"Joints: {joints}")
        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Error: {error:.6f} m")
        
        assert error < 1e-6, f"FK error too large: {error}"
    
    print("\n✓ All validation tests passed!")

# Example test cases
test_cases = [
    {
        'joints': [0, 0],
        'expected_position': np.array([1.8, 0, 0])  # L1 + L2
    },
    {
        'joints': [np.pi/2, 0],
        'expected_position': np.array([0, 1.8, 0])
    }
]
```

---

## Best Practices

```text
✅ DO:
- Follow DH convention strictly
- Document frame assignments clearly
- Use consistent units (radians, meters)
- Validate with known configurations
- Draw diagrams before implementing
- Check for parallel/intersecting axes cases

❌ DON'T:
- Mix DH conventions (classic vs modified)
- Skip frame assignment verification
- Use degrees in calculations
- Forget to handle special cases (parallel axes)
- Assume arbitrary frame placement is valid
```

---

## Further Reading

- [Craig, J.J. "Introduction to Robotics: Mechanics and Control"](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-robotics-mechanics-and-control/P200000003484)
- [Spong, M.W. "Robot Modeling and Control"](https://www.wiley.com/en-us/Robot+Modeling+and+Control-p-9780471649908)
- [DH Parameters Tutorial (Duke)](https://www.cs.duke.edu/brd/Teaching/Bio/asmb/current/Papers/chap3-forward-kinematics.pdf)
- [Modified DH Parameters (Khalil)](https://hal.science/hal-01478537/document)
