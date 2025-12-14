---
title: "Rear-Wheel Drive (RWD) Kinematics"
date: 2024-12-13
draft: false
category: "kinematics"
tags: ["robotics", "mobile-robots", "rear-wheel-drive", "kinematics"]
---

Forward and inverse kinematics for rear-wheel drive robots with front-wheel steering.

## Overview

**Rear-Wheel Drive (RWD)** has driven wheels at the rear and steered wheels at the front, like many trucks and sports cars.

**Advantages:**
- Better weight distribution
- Good for pushing/towing
- Better acceleration (weight transfer)
- Simpler drivetrain

**Disadvantages:**
- Can oversteer (tail slides out)
- Less intuitive than FWD
- Potential for fishtailing
- Cannot rotate in place

---

## Robot Configuration

```text
        Front (Steered only)
              ↑
         ┌─────────┐
         │ ╱     ╲ │  ← Steered wheels (passive)
         │         │
         │    •    │  ← Center of rotation
         │         │
         │ ■     ■ │  ← Rear wheels (driven)
         └─────────┘
```

**Parameters:**
- $L$: Wheelbase (distance between front and rear axles)
- $\delta$: Front steering angle
- $v$: Linear velocity at rear axle
- $r$: Wheel radius

---

## Interactive RWD Simulator

```p5js
let speedSlider, steerSlider;
let robotX = 0;
let robotY = 0;
let robotTheta = 0;
let trail = [];
let wheelbase = 100;

sketch.setup = function() {
  sketch.createCanvas(800, 750);
  
  speedSlider = sketch.createSlider(-5, 5, 2, 0.5);
  speedSlider.position(150, 770);
  speedSlider.style('width', '200px');
  
  steerSlider = sketch.createSlider(-sketch.PI/3, sketch.PI/3, 0, 0.01);
  steerSlider.position(450, 770);
  steerSlider.style('width', '200px');
  
  let resetBtn = sketch.createButton('Reset');
  resetBtn.position(700, 770);
  resetBtn.mousePressed(resetRobot);
  
  robotX = sketch.width/2;
  robotY = sketch.height/2;
  robotTheta = 0;
}

function resetRobot() {
  robotX = sketch.width/2;
  robotY = sketch.height/2;
  robotTheta = 0;
  trail = [];
}

sketch.draw = function() {
  sketch.background(255);
  
  let v = speedSlider.value();
  let delta = steerSlider.value();
  
  // Title
  sketch.fill(0);
  sketch.textSize(18);
  sketch.textAlign(sketch.CENTER);
  sketch.text('Rear-Wheel Drive Robot (Front Steering)', sketch.width/2, 25);
  
  // Calculate angular velocity (RWD uses rear axle as reference)
  let omega = 0;
  let R = 0;
  if (Math.abs(delta) > 0.01) {
    // For RWD, turning radius is measured from rear axle
    R = wheelbase / Math.sin(delta);
    omega = v * Math.sin(delta) / wheelbase;
  }
  
  // Update robot pose (velocity at rear axle)
  let dt = 0.1;
  robotX += v * sketch.cos(robotTheta) * dt;
  robotY += v * sketch.sin(robotTheta) * dt;
  robotTheta += omega * dt;
  
  // Keep robot on screen
  if (robotX < 80) robotX = 80;
  if (robotX > sketch.width - 80) robotX = sketch.width - 80;
  if (robotY < 80) robotY = 80;
  if (robotY > sketch.height - 200) robotY = sketch.height - 200;
  
  // Add to trail (rear axle position)
  if (sketch.frameCount % 3 === 0) {
    trail.push({x: robotX, y: robotY});
    if (trail.length > 200) trail.shift();
  }
  
  // Draw trail
  sketch.stroke(255, 150, 100, 100);
  sketch.strokeWeight(2);
  sketch.noFill();
  sketch.beginShape();
  for (let p of trail) {
    sketch.vertex(p.x, p.y);
  }
  sketch.endShape();
  
  // Draw turning radius indicator
  if (Math.abs(delta) > 0.01 && Math.abs(R) < 500) {
    // ICC is perpendicular to rear axle
    let iccX = robotX - R * sketch.sin(robotTheta);
    let iccY = robotY + R * sketch.cos(robotTheta);
    
    sketch.stroke(255, 0, 0, 100);
    sketch.strokeWeight(1);
    sketch.noFill();
    sketch.circle(iccX, iccY, Math.abs(R) * 2);
    
    // ICC marker
    sketch.fill(255, 0, 0, 150);
    sketch.noStroke();
    sketch.circle(iccX, iccY, 8);
    sketch.textSize(10);
    sketch.text('ICC', iccX + 15, iccY);
    
    // Line to ICC
    sketch.stroke(255, 0, 0, 50);
    sketch.strokeWeight(1);
    sketch.line(robotX, robotY, iccX, iccY);
  }
  
  // Draw robot
  sketch.push();
  sketch.translate(robotX, robotY);
  sketch.rotate(robotTheta);
  
  // Robot body
  sketch.fill(255, 230, 220);
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.rect(-40, -30, 80, 60, 5);
  
  // Front wheels (steered only, passive)
  sketch.push();
  sketch.translate(wheelbase/2, -25);
  sketch.rotate(delta);
  
  // Left front wheel (passive)
  sketch.fill(180);
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.rect(-8, -12, 16, 24, 2);
  
  // Steering linkage
  sketch.stroke(100);
  sketch.strokeWeight(1);
  sketch.line(0, 0, 0, -15);
  
  sketch.pop();
  
  sketch.push();
  sketch.translate(wheelbase/2, 25);
  sketch.rotate(delta);
  
  // Right front wheel (passive)
  sketch.fill(180);
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.rect(-8, -12, 16, 24, 2);
  
  // Steering linkage
  sketch.stroke(100);
  sketch.strokeWeight(1);
  sketch.line(0, 0, 0, 15);
  
  sketch.pop();
  
  // Rear wheels (driven)
  sketch.fill(v > 0 ? [0, 255, 0] : v < 0 ? [255, 0, 0] : [150]);
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.rect(-wheelbase/2 - 8, -25 - 12, 16, 24, 2);
  sketch.rect(-wheelbase/2 - 8, 25 - 12, 16, 24, 2);
  
  // Drive indicators on rear wheels
  if (Math.abs(v) > 0.1) {
    sketch.stroke(0);
    sketch.strokeWeight(2);
    sketch.noFill();
    let arcAngle = v > 0 ? sketch.HALF_PI : -sketch.HALF_PI;
    sketch.arc(-wheelbase/2, -25, 20, 20, 0, arcAngle);
    sketch.arc(-wheelbase/2, 25, 20, 20, 0, arcAngle);
  }
  
  // Direction arrow
  sketch.fill(255, 0, 0);
  sketch.noStroke();
  sketch.triangle(wheelbase/2 + 20, 0, wheelbase/2 + 5, -8, wheelbase/2 + 5, 8);
  
  // Steering angle indicator
  if (Math.abs(delta) > 0.01) {
    sketch.stroke(0, 0, 255);
    sketch.strokeWeight(2);
    sketch.noFill();
    sketch.arc(wheelbase/2, 0, 40, 40, -delta, 0);
  }
  
  // Rear axle marker (reference point)
  sketch.stroke(255, 100, 0);
  sketch.strokeWeight(3);
  sketch.line(-wheelbase/2, -35, -wheelbase/2, 35);
  
  // Center point
  sketch.fill(0);
  sketch.noStroke();
  sketch.circle(0, 0, 6);
  
  sketch.pop();
  
  // Display info
  sketch.fill(0);
  sketch.textSize(14);
  sketch.textAlign(sketch.LEFT);
  sketch.text(`Speed: v = ${v.toFixed(1)} m/s`, 20, 785);
  sketch.text(`Steering: δ = ${(delta * 180 / sketch.PI).toFixed(1)}°`, 400, 785);
  
  sketch.textSize(13);
  sketch.text(`Linear velocity (rear axle): ${v.toFixed(2)} m/s`, 20, 50);
  sketch.text(`Angular velocity: ${omega.toFixed(3)} rad/s`, 20, 70);
  sketch.text(`Heading: θ = ${(robotTheta * 180 / sketch.PI % 360).toFixed(1)}°`, 20, 90);
  
  if (Math.abs(delta) > 0.01) {
    sketch.text(`Turning radius: R = ${Math.abs(R).toFixed(1)} units`, 20, 110);
  }
  
  // Behavior note
  sketch.fill(200, 0, 0);
  sketch.textSize(12);
  if (v < 0 && Math.abs(delta) > 0.1) {
    sketch.text('⚠ Reversing with steering: Watch for oversteer!', 20, 130);
  }
  
  sketch.fill(150);
  sketch.textSize(11);
  sketch.text('Green = forward, Red = backward, Gray = passive', 20, sketch.height - 20);
  sketch.text('Orange line = rear axle (reference point)', 20, sketch.height - 5);
}
```

---

## Forward Kinematics

### RWD Equations

For RWD, the velocity reference is at the **rear axle**:

$$
\omega = \frac{v \sin\delta}{L}
$$

$$
R = \frac{L}{\sin\delta}
$$

Where:
- $v$: Velocity at rear axle
- $\delta$: Front steering angle
- $L$: Wheelbase

### Pose Update

$$
\dot{x} = v \cos\theta
$$

$$
\dot{y} = v \sin\theta
$$

$$
\dot{\theta} = \omega = \frac{v \sin\delta}{L}
$$

**Note:** Position $(x, y)$ is at the rear axle, not the center!

### Python Implementation

```python
import numpy as np

class RearWheelDrive:
    def __init__(self, wheelbase=0.5, wheel_radius=0.1):
        """
        Args:
            wheelbase: Distance between front and rear axles (m)
            wheel_radius: Radius of wheels (m)
        """
        self.L = wheelbase
        self.r = wheel_radius
        
        # State: [x, y, theta] at rear axle
        self.state = np.array([0.0, 0.0, 0.0])
    
    def forward_kinematics(self, v, delta, dt):
        """
        Update robot pose from velocity and steering angle
        
        Args:
            v: Linear velocity at rear axle (m/s)
            delta: Front steering angle (rad)
            dt: Time step (s)
        
        Returns:
            Updated pose [x, y, theta] at rear axle
        """
        x, y, theta = self.state
        
        # Calculate angular velocity
        if abs(delta) < 1e-6:
            omega = 0
        else:
            omega = v * np.sin(delta) / self.L
        
        # Update pose (rear axle position)
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt
        
        # Normalize angle
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        self.state = np.array([x, y, theta])
        return self.state.copy()
    
    def get_front_position(self):
        """Get front axle position"""
        x, y, theta = self.state
        x_front = x + self.L * np.cos(theta)
        y_front = y + self.L * np.sin(theta)
        return x_front, y_front
    
    def get_turning_radius(self, delta):
        """Calculate turning radius for given steering angle"""
        if abs(delta) < 1e-6:
            return float('inf')
        return self.L / np.sin(delta)

# Example
robot = RearWheelDrive(wheelbase=0.5, wheel_radius=0.1)

# Straight line
for _ in range(100):
    robot.forward_kinematics(v=1.0, delta=0, dt=0.01)

print(f"After straight motion (rear axle): {robot.state}")
print(f"Front axle position: {robot.get_front_position()}")
```

---

## Inverse Kinematics

### From Desired Path to Control Inputs

Given desired turning radius $R$ and velocity $v$:

$$
\delta = \arcsin\left(\frac{L}{R}\right)
$$

### Implementation

```python
def inverse_kinematics(self, v_desired, R_desired):
    """
    Compute steering angle for desired turning radius
    
    Args:
        v_desired: Desired linear velocity at rear axle (m/s)
        R_desired: Desired turning radius (m)
    
    Returns:
        (v, delta): velocity and steering angle
    """
    if abs(R_desired) > 1000:  # Effectively straight
        delta = 0
    else:
        # Ensure R is achievable
        if abs(R_desired) < self.L:
            R_desired = self.L * np.sign(R_desired)
        
        delta = np.arcsin(self.L / R_desired)
    
    return v_desired, delta

# Add to RearWheelDrive class
RearWheelDrive.inverse_kinematics = inverse_kinematics

# Example
robot = RearWheelDrive(wheelbase=0.5)
v, delta = robot.inverse_kinematics(v_desired=1.0, R_desired=2.0)
print(f"For R=2m: v={v:.2f} m/s, δ={np.degrees(delta):.1f}°")
```

---

## RWD vs FWD Comparison

| Aspect | RWD | FWD |
|--------|-----|-----|
| **Reference Point** | Rear axle | Center/Front axle |
| **Kinematics** | $\omega = \frac{v\sin\delta}{L}$ | $\omega = \frac{v\tan\delta}{L}$ |
| **Turning Radius** | $R = \frac{L}{\sin\delta}$ | $R = \frac{L}{\tan\delta}$ |
| **Stability** | Can oversteer | More stable |
| **Reversing** | More challenging | Easier |
| **Towing** | Better | Worse |

---

## Oversteer Behavior

### Understanding Oversteer

RWD vehicles can **oversteer** when:
- Reversing with steering input
- High speed turns
- Slippery surfaces

```python
def check_oversteer_risk(self, v, delta, friction_coef=0.7):
    """
    Check for potential oversteer conditions
    
    Args:
        v: Velocity
        delta: Steering angle
        friction_coef: Surface friction coefficient
    
    Returns:
        (risk_level, warning)
    """
    # Calculate lateral acceleration
    if abs(delta) < 1e-6:
        return 'low', None
    
    R = self.get_turning_radius(delta)
    lateral_accel = v**2 / abs(R)
    max_accel = friction_coef * 9.81
    
    risk_ratio = lateral_accel / max_accel
    
    if risk_ratio > 0.9:
        return 'high', 'Severe oversteer risk!'
    elif risk_ratio > 0.7:
        return 'medium', 'Moderate oversteer risk'
    elif risk_ratio > 0.5:
        return 'low', 'Slight oversteer possible'
    else:
        return 'minimal', None

# Add to class
RearWheelDrive.check_oversteer_risk = check_oversteer_risk
```

---

## Reversing Control

### Special Considerations for Reverse

When reversing, RWD behaves differently:

```python
def reverse_control(self, v_reverse, target_x, target_y):
    """
    Control for reversing (more challenging than forward)
    
    Args:
        v_reverse: Reverse velocity (negative)
        target_x, target_y: Target position
    
    Returns:
        Steering angle
    """
    x, y, theta = self.state
    
    # Vector to target
    dx = target_x - x
    dy = target_y - y
    
    # Desired heading (opposite for reverse)
    target_theta = np.arctan2(dy, dx) + np.pi
    
    # Heading error
    theta_error = np.arctan2(np.sin(target_theta - theta),
                             np.cos(target_theta - theta))
    
    # Proportional control (reversed sign for reverse motion)
    delta = -np.clip(theta_error, -np.pi/4, np.pi/4)
    
    return delta

# Add to class
RearWheelDrive.reverse_control = reverse_control
```

---

## Trailer Towing

RWD is better for towing because the driven wheels push the load:

```python
def towing_kinematics(self, v, delta, trailer_length, dt):
    """
    Kinematics with trailer attached
    
    Args:
        trailer_length: Length from hitch to trailer axle
        
    Returns:
        (robot_pose, trailer_angle)
    """
    # Update robot
    self.forward_kinematics(v, delta, dt)
    
    # Trailer angle (simplified model)
    x, y, theta = self.state
    
    # Hitch is at rear axle
    # Trailer follows with its own angle
    
    return self.state, None  # Simplified
```

---

## Advantages and Applications

### Advantages
- **Better towing**: Push force more effective
- **Weight distribution**: Better for heavy loads
- **Acceleration**: Weight transfer to driven wheels
- **Durability**: Simpler drivetrain

### Applications
- **Towing robots**: Pulling carts/trailers
- **Heavy-duty platforms**: Construction/industrial
- **Forklift AGVs**: Pushing/pulling loads
- **Outdoor robots**: Rough terrain with loads

---

## Further Reading

- [Vehicle Dynamics](https://www.springer.com/gp/book/9783319534411)
- [Oversteer vs Understeer](https://en.wikipedia.org/wiki/Oversteer)
- [Mobile Robot Kinematics](https://www.sciencedirect.com/topics/engineering/mobile-robot)

