---
title: "DH Parameter Examples and Common Configurations"
date: 2024-12-13
draft: false
category: "kinematics"
tags: ["robotics", "kinematics", "dh-parameters", "examples"]
---

Practical examples of DH parameter configurations for common robot manipulators with complete Python implementations.

## 2-Link Planar Arm

Simplest manipulator for learning DH parameters.

### Interactive 2-Link Arm

```p5js
let theta1Slider, theta2Slider;
let L1 = 150;
let L2 = 120;

sketch.setup = function() {
  sketch.createCanvas(800, 650);
  
  theta1Slider = sketch.createSlider(-sketch.PI, sketch.PI, sketch.PI/4, 0.01);
  theta1Slider.position(150, 670);
  theta1Slider.style('width', '200px');
  
  theta2Slider = sketch.createSlider(-sketch.PI, sketch.PI, sketch.PI/3, 0.01);
  theta2Slider.position(450, 670);
  theta2Slider.style('width', '200px');
}

sketch.draw = function() {
  sketch.background(255);
  
  let theta1 = theta1Slider.value();
  let theta2 = theta2Slider.value();
  
  // Title
  sketch.fill(0);
  sketch.textSize(18);
  sketch.textAlign(sketch.CENTER);
  sketch.text('2-Link Planar Arm - Forward Kinematics', sketch.width/2, 25);
  
  // Draw workspace circle
  sketch.push();
  sketch.translate(sketch.width/2, sketch.height/2);
  
  sketch.stroke(200);
  sketch.strokeWeight(1);
  sketch.noFill();
  sketch.circle(0, 0, 2 * (L1 + L2));
  sketch.circle(0, 0, 2 * Math.abs(L1 - L2));
  
  // Base
  sketch.fill(100);
  sketch.noStroke();
  sketch.circle(0, 0, 20);
  
  // Link 1
  let x1 = L1 * sketch.cos(theta1);
  let y1 = L1 * sketch.sin(theta1);
  
  sketch.stroke(200, 0, 0);
  sketch.strokeWeight(8);
  sketch.line(0, 0, x1, y1);
  
  // Joint 1
  sketch.fill(255, 100, 100);
  sketch.noStroke();
  sketch.circle(x1, y1, 15);
  
  // Link 2
  let x2 = x1 + L2 * sketch.cos(theta1 + theta2);
  let y2 = y1 + L2 * sketch.sin(theta1 + theta2);
  
  sketch.stroke(0, 0, 200);
  sketch.strokeWeight(8);
  sketch.line(x1, y1, x2, y2);
  
  // End effector
  sketch.fill(0, 200, 0);
  sketch.noStroke();
  sketch.circle(x2, y2, 20);
  
  // Draw trajectory trace
  if (sketch.frameCount % 3 === 0) {
    sketch.stroke(0, 200, 0, 50);
    sketch.strokeWeight(2);
    sketch.point(x2, y2);
  }
  
  // Angle arcs
  sketch.noFill();
  sketch.stroke(255, 0, 0, 150);
  sketch.strokeWeight(2);
  sketch.arc(0, 0, 60, 60, 0, theta1);
  
  sketch.stroke(0, 0, 255, 150);
  sketch.arc(x1, y1, 50, 50, theta1, theta1 + theta2);
  
  sketch.pop();
  
  // Calculate end-effector position
  let endX = L1 * sketch.cos(theta1) + L2 * sketch.cos(theta1 + theta2);
  let endY = L1 * sketch.sin(theta1) + L2 * sketch.sin(theta1 + theta2);
  let endAngle = theta1 + theta2;
  
  // Display info
  sketch.fill(0);
  sketch.textSize(14);
  sketch.textAlign(sketch.LEFT);
  sketch.text(`θ₁ = ${(theta1 * 180 / sketch.PI).toFixed(1)}°`, 20, 685);
  sketch.text(`θ₂ = ${(theta2 * 180 / sketch.PI).toFixed(1)}°`, 370, 685);
  
  sketch.textSize(13);
  sketch.text(`End-effector: x=${endX.toFixed(1)}, y=${endY.toFixed(1)}`, 20, 50);
  sketch.text(`Orientation: φ=${(endAngle * 180 / sketch.PI).toFixed(1)}°`, 20, 70);
  sketch.text(`Reach: ${sketch.sqrt(endX*endX + endY*endY).toFixed(1)}`, 20, 90);
  
  sketch.fill(150);
  sketch.textSize(11);
  sketch.text('Workspace: Inner circle = min reach, Outer circle = max reach', 20, sketch.height - 20);
}
```

---

### DH Table

| Joint | $\theta$ | $d$ | $a$ | $\alpha$ |
|-------|----------|-----|-----|----------|
| 1     | $\theta_1$ | 0 | $L_1$ | 0 |
| 2     | $\theta_2$ | 0 | $L_2$ | 0 |

### Implementation

```python
import numpy as np

def dh_matrix(theta, d, a, alpha):
    """DH transformation matrix"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])

class PlanarArm2DOF:
    def __init__(self, L1=1.0, L2=0.8):
        self.L1 = L1
        self.L2 = L2
    
    def forward_kinematics(self, theta1, theta2):
        """Compute end-effector pose"""
        T1 = dh_matrix(theta1, 0, self.L1, 0)
        T2 = dh_matrix(theta2, 0, self.L2, 0)
        
        T = T1 @ T2
        
        x = T[0, 3]
        y = T[1, 3]
        phi = np.arctan2(T[1, 0], T[0, 0])
        
        return x, y, phi
    
    def plot_arm(self, theta1, theta2):
        """Visualize arm configuration"""
        import matplotlib.pyplot as plt
        
        # Joint positions
        p0 = np.array([0, 0])
        p1 = np.array([self.L1 * np.cos(theta1),
                       self.L1 * np.sin(theta1)])
        p2 = np.array([p1[0] + self.L2 * np.cos(theta1 + theta2),
                       p1[1] + self.L2 * np.sin(theta1 + theta2)])
        
        plt.figure(figsize=(8, 8))
        plt.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[2]], 'o-', linewidth=2)
        plt.plot(p0[0], p0[1], 'ro', markersize=10, label='Base')
        plt.plot(p2[0], p2[1], 'go', markersize=10, label='End-effector')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.title(f'θ₁={np.degrees(theta1):.1f}°, θ₂={np.degrees(theta2):.1f}°')
        plt.show()

# Example
arm = PlanarArm2DOF(L1=1.0, L2=0.8)
x, y, phi = arm.forward_kinematics(np.pi/4, np.pi/3)
print(f"End-effector: x={x:.3f}, y={y:.3f}, φ={np.degrees(phi):.1f}°")
```

---

## 3-DOF RRR Manipulator

Spatial manipulator with 3 revolute joints.

### DH Table

| Joint | $\theta$ | $d$ | $a$ | $\alpha$ |
|-------|----------|-----|-----|----------|
| 1     | $\theta_1$ | $L_1$ | 0 | $\pi/2$ |
| 2     | $\theta_2$ | 0 | $L_2$ | 0 |
| 3     | $\theta_3$ | 0 | $L_3$ | 0 |

### Implementation

```python
class RRRManipulator:
    def __init__(self, L1=0.5, L2=0.4, L3=0.3):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
    
    def forward_kinematics(self, theta1, theta2, theta3):
        """Compute end-effector pose"""
        T1 = dh_matrix(theta1, self.L1, 0, np.pi/2)
        T2 = dh_matrix(theta2, 0, self.L2, 0)
        T3 = dh_matrix(theta3, 0, self.L3, 0)
        
        T = T1 @ T2 @ T3
        return T
    
    def get_joint_positions(self, theta1, theta2, theta3):
        """Get position of each joint"""
        T0 = np.eye(4)
        T1 = dh_matrix(theta1, self.L1, 0, np.pi/2)
        T2 = dh_matrix(theta2, 0, self.L2, 0)
        T3 = dh_matrix(theta3, 0, self.L3, 0)
        
        positions = [
            T0[:3, 3],
            T1[:3, 3],
            (T1 @ T2)[:3, 3],
            (T1 @ T2 @ T3)[:3, 3]
        ]
        
        return positions

# Example
robot = RRRManipulator()
T = robot.forward_kinematics(0, np.pi/4, np.pi/6)
print("End-effector position:", T[:3, 3])
```

---

## SCARA Robot

Selective Compliance Assembly Robot Arm - 2 revolute + 1 prismatic + 1 revolute.

### DH Table

| Joint | $\theta$ | $d$ | $a$ | $\alpha$ |
|-------|----------|-----|-----|----------|
| 1     | $\theta_1$ | 0 | $L_1$ | 0 |
| 2     | $\theta_2$ | 0 | $L_2$ | $\pi$ |
| 3     | 0 | $d_3$ | 0 | 0 |
| 4     | $\theta_4$ | 0 | 0 | 0 |

### Implementation

```python
class SCARArobot:
    def __init__(self, L1=0.3, L2=0.25, d3_min=0.0, d3_max=0.2):
        self.L1 = L1
        self.L2 = L2
        self.d3_min = d3_min
        self.d3_max = d3_max
    
    def forward_kinematics(self, theta1, theta2, d3, theta4):
        """Compute end-effector pose"""
        # Clamp prismatic joint
        d3 = np.clip(d3, self.d3_min, self.d3_max)
        
        T1 = dh_matrix(theta1, 0, self.L1, 0)
        T2 = dh_matrix(theta2, 0, self.L2, np.pi)
        T3 = dh_matrix(0, d3, 0, 0)
        T4 = dh_matrix(theta4, 0, 0, 0)
        
        T = T1 @ T2 @ T3 @ T4
        return T
    
    def workspace_limits(self):
        """Calculate workspace dimensions"""
        r_max = self.L1 + self.L2
        r_min = abs(self.L1 - self.L2)
        z_min = self.d3_min
        z_max = self.d3_max
        
        return {
            'r_min': r_min,
            'r_max': r_max,
            'z_min': z_min,
            'z_max': z_max
        }

# Example
scara = SCARArobot()
T = scara.forward_kinematics(np.pi/6, np.pi/4, 0.1, 0)
print(f"SCARA end-effector: [{T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f}]")
print("Workspace:", scara.workspace_limits())
```

---

## Stanford Arm

5-DOF manipulator: 2 revolute + 1 prismatic + 2 revolute.

### DH Table

| Joint | $\theta$ | $d$ | $a$ | $\alpha$ |
|-------|----------|-----|-----|----------|
| 1     | $\theta_1$ | 0 | 0 | $-\pi/2$ |
| 2     | $\theta_2$ | $d_2$ | 0 | $\pi/2$ |
| 3     | 0 | $d_3$ | 0 | 0 |
| 4     | $\theta_4$ | 0 | 0 | $-\pi/2$ |
| 5     | $\theta_5$ | 0 | 0 | 0 |

### Implementation

```python
class StanfordArm:
    def __init__(self, d2=0.154, d3_min=0.0, d3_max=0.5):
        self.d2 = d2
        self.d3_min = d3_min
        self.d3_max = d3_max
    
    def forward_kinematics(self, theta1, theta2, d3, theta4, theta5):
        """Compute end-effector pose"""
        d3 = np.clip(d3, self.d3_min, self.d3_max)
        
        T1 = dh_matrix(theta1, 0, 0, -np.pi/2)
        T2 = dh_matrix(theta2, self.d2, 0, np.pi/2)
        T3 = dh_matrix(0, d3, 0, 0)
        T4 = dh_matrix(theta4, 0, 0, -np.pi/2)
        T5 = dh_matrix(theta5, 0, 0, 0)
        
        T = T1 @ T2 @ T3 @ T4 @ T5
        return T

# Example
stanford = StanfordArm()
T = stanford.forward_kinematics(0, np.pi/4, 0.3, 0, 0)
print("Stanford arm end-effector:", T[:3, 3])
```

---

## PUMA 560

Classic 6-DOF industrial robot.

### DH Table

| Joint | $\theta$ | $d$ | $a$ | $\alpha$ |
|-------|----------|-----|-----|----------|
| 1     | $\theta_1$ | 0 | 0 | $\pi/2$ |
| 2     | $\theta_2$ | 0 | $a_2$ | 0 |
| 3     | $\theta_3$ | $d_3$ | $a_3$ | $-\pi/2$ |
| 4     | $\theta_4$ | $d_4$ | 0 | $\pi/2$ |
| 5     | $\theta_5$ | 0 | 0 | $-\pi/2$ |
| 6     | $\theta_6$ | 0 | 0 | 0 |

### Implementation

```python
class PUMA560:
    def __init__(self):
        # PUMA 560 dimensions (meters)
        self.a2 = 0.4318
        self.a3 = 0.0203
        self.d3 = 0.15005
        self.d4 = 0.4318
    
    def forward_kinematics(self, joints):
        """
        Compute forward kinematics
        
        Args:
            joints: List of 6 joint angles [θ1, θ2, θ3, θ4, θ5, θ6]
        """
        theta1, theta2, theta3, theta4, theta5, theta6 = joints
        
        T1 = dh_matrix(theta1, 0, 0, np.pi/2)
        T2 = dh_matrix(theta2, 0, self.a2, 0)
        T3 = dh_matrix(theta3, self.d3, self.a3, -np.pi/2)
        T4 = dh_matrix(theta4, self.d4, 0, np.pi/2)
        T5 = dh_matrix(theta5, 0, 0, -np.pi/2)
        T6 = dh_matrix(theta6, 0, 0, 0)
        
        T = T1 @ T2 @ T3 @ T4 @ T5 @ T6
        return T
    
    def get_all_transforms(self, joints):
        """Get transformation to each joint frame"""
        theta1, theta2, theta3, theta4, theta5, theta6 = joints
        
        T1 = dh_matrix(theta1, 0, 0, np.pi/2)
        T2 = dh_matrix(theta2, 0, self.a2, 0)
        T3 = dh_matrix(theta3, self.d3, self.a3, -np.pi/2)
        T4 = dh_matrix(theta4, self.d4, 0, np.pi/2)
        T5 = dh_matrix(theta5, 0, 0, -np.pi/2)
        T6 = dh_matrix(theta6, 0, 0, 0)
        
        transforms = [
            np.eye(4),
            T1,
            T1 @ T2,
            T1 @ T2 @ T3,
            T1 @ T2 @ T3 @ T4,
            T1 @ T2 @ T3 @ T4 @ T5,
            T1 @ T2 @ T3 @ T4 @ T5 @ T6
        ]
        
        return transforms

# Example
puma = PUMA560()
joints = [0, np.pi/4, 0, 0, 0, 0]
T = puma.forward_kinematics(joints)
print("PUMA 560 end-effector:")
print(T)
```

---

## Generic N-DOF Manipulator

Flexible class for any serial manipulator.

```python
class GenericManipulator:
    def __init__(self, dh_table):
        """
        Args:
            dh_table: List of dicts with keys:
                - 'a': link length
                - 'alpha': link twist
                - 'd_const': constant offset (for revolute)
                - 'theta_const': constant angle (for prismatic)
                - 'joint_type': 'revolute' or 'prismatic'
        """
        self.dh_table = dh_table
        self.n_joints = len(dh_table)
    
    def forward_kinematics(self, joint_values):
        """
        Compute forward kinematics
        
        Args:
            joint_values: List of joint values (angles or distances)
        """
        T = np.eye(4)
        
        for i, (value, dh) in enumerate(zip(joint_values, self.dh_table)):
            if dh['joint_type'] == 'prismatic':
                d = value
                theta = dh.get('theta_const', 0)
            else:  # revolute
                d = dh.get('d_const', 0)
                theta = value
            
            Ti = dh_matrix(theta, d, dh['a'], dh['alpha'])
            T = T @ Ti
        
        return T
    
    def check_joint_limits(self, joint_values):
        """Check if joints are within limits"""
        for i, (value, dh) in enumerate(zip(joint_values, self.dh_table)):
            limits = dh.get('limits', {'min': -np.inf, 'max': np.inf})
            if not (limits['min'] <= value <= limits['max']):
                return False, f"Joint {i+1} out of range"
        return True, "OK"

# Example: Custom 4-DOF robot
custom_dh = [
    {'a': 0, 'alpha': np.pi/2, 'd_const': 0.5, 'joint_type': 'revolute',
     'limits': {'min': -np.pi, 'max': np.pi}},
    {'a': 0.4, 'alpha': 0, 'd_const': 0, 'joint_type': 'revolute',
     'limits': {'min': -np.pi/2, 'max': np.pi/2}},
    {'a': 0, 'alpha': 0, 'theta_const': 0, 'joint_type': 'prismatic',
     'limits': {'min': 0, 'max': 0.3}},
    {'a': 0, 'alpha': 0, 'd_const': 0.2, 'joint_type': 'revolute',
     'limits': {'min': -np.pi, 'max': np.pi}}
]

robot = GenericManipulator(custom_dh)
joints = [np.pi/6, np.pi/4, 0.15, 0]
valid, msg = robot.check_joint_limits(joints)
if valid:
    T = robot.forward_kinematics(joints)
    print("End-effector:", T[:3, 3])
```

---

## Workspace Visualization

```python
def plot_workspace_3d(robot, n_samples=5000):
    """Plot 3D workspace by sampling joint space"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    points = []
    
    for _ in range(n_samples):
        # Random valid joint configuration
        joints = []
        for dh in robot.dh_table:
            limits = dh.get('limits', {'min': -np.pi, 'max': np.pi})
            value = np.random.uniform(limits['min'], limits['max'])
            joints.append(value)
        
        # Forward kinematics
        T = robot.forward_kinematics(joints)
        points.append(T[:3, 3])
    
    points = np.array(points)
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               alpha=0.1, s=1, c=points[:, 2], cmap='viridis')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Robot Workspace')
    plt.colorbar(ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                            c=points[:, 2], cmap='viridis'), ax=ax)
    plt.show()

# Example
plot_workspace_3d(robot, n_samples=3000)
```

---

## Joint Limit Handling

```python
class RobotWithLimits:
    def __init__(self, dh_table, joint_limits):
        self.dh_table = dh_table
        self.joint_limits = joint_limits
    
    def clamp_joints(self, joints):
        """Clamp joints to valid range"""
        clamped = []
        for joint, limits in zip(joints, self.joint_limits):
            clamped.append(np.clip(joint, limits['min'], limits['max']))
        return np.array(clamped)
    
    def safe_forward_kinematics(self, joints):
        """FK with automatic joint clamping"""
        joints_safe = self.clamp_joints(joints)
        
        # Warn if clamping occurred
        if not np.allclose(joints, joints_safe):
            print("Warning: Joints clamped to limits")
            for i, (orig, safe) in enumerate(zip(joints, joints_safe)):
                if not np.isclose(orig, safe):
                    print(f"  Joint {i+1}: {orig:.3f} → {safe:.3f}")
        
        # Compute FK
        T = np.eye(4)
        for value, dh in zip(joints_safe, self.dh_table):
            if dh['joint_type'] == 'prismatic':
                Ti = dh_matrix(dh.get('theta_const', 0), value, 
                              dh['a'], dh['alpha'])
            else:
                Ti = dh_matrix(value, dh.get('d_const', 0), 
                              dh['a'], dh['alpha'])
            T = T @ Ti
        
        return T, joints_safe
```

---

## Further Reading

- [Common Robot Configurations](https://www.robotics.org/joseph-engelberger/robot-configurations.cfm)
- [PUMA 560 Specifications](http://www.mech.sharif.ir/c/document_library/get_file?uuid=5a4bb247-1430-4e46-942c-d692dead831f&groupId=14040)
- [Industrial Robot Kinematics](https://link.springer.com/book/10.1007/978-1-4471-4664-3)

