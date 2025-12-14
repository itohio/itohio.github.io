---
title: "Front-Wheel Drive (FWD) Kinematics"
date: 2024-12-13
draft: false
category: "kinematics"
tags: ["robotics", "mobile-robots", "front-wheel-drive", "ackermann", "kinematics"]
---

Forward and inverse kinematics for front-wheel drive robots with Ackermann steering geometry.

## Overview

**Front-Wheel Drive (FWD)** combines steering and driving at the front wheels, similar to most cars.

**Advantages:**
- Car-like steering (intuitive)
- Good forward traction
- Simpler than four-wheel steering
- Predictable handling

**Disadvantages:**
- Cannot rotate in place
- Larger turning radius
- Limited maneuverability in tight spaces
- Front wheel wear

---

## Robot Configuration

```text
        Front (Steered + Driven)
              ↑
         ┌─────────┐
         │ ╱     ╲ │  ← Steered wheels
         │         │
         │    •    │  ← Center of rotation
         │         │
         │ │     │ │  ← Rear wheels (passive)
         └─────────┘
```

**Parameters:**
- $L$: Wheelbase (distance between front and rear axles)
- $\delta$: Steering angle
- $v$: Linear velocity
- $r$: Wheel radius

---

## Interactive FWD Simulator

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
  sketch.text('Front-Wheel Drive Robot (Ackermann Steering)', sketch.width/2, 25);
  
  // Calculate angular velocity
  let omega = 0;
  let R = 0;
  if (Math.abs(delta) > 0.01) {
    R = wheelbase / Math.tan(delta);
    omega = v / R;
  }
  
  // Update robot pose
  let dt = 0.1;
  if (Math.abs(delta) < 0.01) {
    // Straight line
    robotX += v * sketch.cos(robotTheta) * dt;
    robotY += v * sketch.sin(robotTheta) * dt;
  } else {
    // Arc motion
    robotX += v * sketch.cos(robotTheta) * dt;
    robotY += v * sketch.sin(robotTheta) * dt;
    robotTheta += omega * dt;
  }
  
  // Keep robot on screen
  if (robotX < 80) robotX = 80;
  if (robotX > sketch.width - 80) robotX = sketch.width - 80;
  if (robotY < 80) robotY = 80;
  if (robotY > sketch.height - 200) robotY = sketch.height - 200;
  
  // Add to trail
  if (sketch.frameCount % 3 === 0) {
    trail.push({x: robotX, y: robotY});
    if (trail.length > 200) trail.shift();
  }
  
  // Draw trail
  sketch.stroke(100, 150, 255, 100);
  sketch.strokeWeight(2);
  sketch.noFill();
  sketch.beginShape();
  for (let p of trail) {
    sketch.vertex(p.x, p.y);
  }
  sketch.endShape();
  
  // Draw turning radius indicator
  if (Math.abs(delta) > 0.01 && Math.abs(R) < 500) {
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
  }
  
  // Draw robot
  sketch.push();
  sketch.translate(robotX, robotY);
  sketch.rotate(robotTheta);
  
  // Robot body
  sketch.fill(220, 230, 255);
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.rect(-40, -30, 80, 60, 5);
  
  // Front wheels (steered and driven)
  sketch.push();
  sketch.translate(wheelbase/2, -25);
  sketch.rotate(delta);
  
  // Left front wheel
  sketch.fill(v > 0 ? [0, 255, 0] : v < 0 ? [255, 0, 0] : [150]);
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
  
  // Right front wheel
  sketch.fill(v > 0 ? [0, 255, 0] : v < 0 ? [255, 0, 0] : [150]);
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.rect(-8, -12, 16, 24, 2);
  
  // Steering linkage
  sketch.stroke(100);
  sketch.strokeWeight(1);
  sketch.line(0, 0, 0, 15);
  
  sketch.pop();
  
  // Rear wheels (passive)
  sketch.fill(180);
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.rect(-wheelbase/2 - 8, -25 - 12, 16, 24, 2);
  sketch.rect(-wheelbase/2 - 8, 25 - 12, 16, 24, 2);
  
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
  sketch.text(`Linear velocity: ${v.toFixed(2)} m/s`, 20, 50);
  sketch.text(`Angular velocity: ${omega.toFixed(3)} rad/s`, 20, 70);
  sketch.text(`Heading: θ = ${(robotTheta * 180 / sketch.PI % 360).toFixed(1)}°`, 20, 90);
  
  if (Math.abs(delta) > 0.01) {
    sketch.text(`Turning radius: R = ${Math.abs(R).toFixed(1)} units`, 20, 110);
    sketch.text(`Min turning radius: ${wheelbase.toFixed(1)} units`, 20, 130);
  }
  
  sketch.fill(150);
  sketch.textSize(11);
  sketch.text('Green = forward, Red = backward, Gray = passive', 20, sketch.height - 20);
  sketch.text('Red circle = Instantaneous Center of Curvature (ICC)', 20, sketch.height - 5);
}
```

---

## Forward Kinematics

### Ackermann Steering Equations

Given steering angle $\delta$ and velocity $v$:

$$
\omega = \frac{v \tan\delta}{L}
$$

$$
R = \frac{L}{\tan\delta}
$$

Where:
- $\omega$: Angular velocity
- $R$: Turning radius
- $L$: Wheelbase

### Pose Update

$$
\dot{x} = v \cos\theta
$$

$$
\dot{y} = v \sin\theta
$$

$$
\dot{\theta} = \omega = \frac{v \tan\delta}{L}
$$

### Python Implementation

```python
import numpy as np

class FrontWheelDrive:
    def __init__(self, wheelbase=0.5, wheel_radius=0.1):
        """
        Args:
            wheelbase: Distance between front and rear axles (m)
            wheel_radius: Radius of wheels (m)
        """
        self.L = wheelbase
        self.r = wheel_radius
        
        # State: [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])
    
    def forward_kinematics(self, v, delta, dt):
        """
        Update robot pose from velocity and steering angle
        
        Args:
            v: Linear velocity (m/s)
            delta: Steering angle (rad)
            dt: Time step (s)
        
        Returns:
            Updated pose [x, y, theta]
        """
        x, y, theta = self.state
        
        # Calculate angular velocity
        if abs(delta) < 1e-6:
            # Straight line motion
            omega = 0
        else:
            omega = v * np.tan(delta) / self.L
        
        # Update pose
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt
        
        # Normalize angle
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        self.state = np.array([x, y, theta])
        return self.state.copy()
    
    def get_turning_radius(self, delta):
        """Calculate turning radius for given steering angle"""
        if abs(delta) < 1e-6:
            return float('inf')
        return self.L / np.tan(delta)
    
    def get_min_turning_radius(self, max_steering_angle=np.pi/3):
        """Get minimum turning radius"""
        return self.L / np.tan(max_steering_angle)

# Example
robot = FrontWheelDrive(wheelbase=0.5, wheel_radius=0.1)

# Straight line
for _ in range(100):
    robot.forward_kinematics(v=1.0, delta=0, dt=0.01)

print(f"After straight motion: {robot.state}")

# Turn
robot.state = np.array([0.0, 0.0, 0.0])
for _ in range(100):
    robot.forward_kinematics(v=1.0, delta=np.pi/6, dt=0.01)

print(f"After turning: {robot.state}")
```

---

## Inverse Kinematics

### From Desired Path to Control Inputs

Given desired turning radius $R$ and velocity $v$:

$$
\delta = \arctan\left(\frac{L}{R}\right)
$$

### Implementation

```python
def inverse_kinematics(self, v_desired, R_desired):
    """
    Compute steering angle for desired turning radius
    
    Args:
        v_desired: Desired linear velocity (m/s)
        R_desired: Desired turning radius (m)
            - Positive: turn left
            - Negative: turn right
            - Infinity: straight line
    
    Returns:
        (v, delta): velocity and steering angle
    """
    if abs(R_desired) > 1000:  # Effectively straight
        delta = 0
    else:
        delta = np.arctan(self.L / R_desired)
    
    return v_desired, delta

# Add to FrontWheelDrive class
FrontWheelDrive.inverse_kinematics = inverse_kinematics

# Example: Turn with 2m radius
robot = FrontWheelDrive(wheelbase=0.5)
v, delta = robot.inverse_kinematics(v_desired=1.0, R_desired=2.0)
print(f"For R=2m: v={v:.2f} m/s, δ={np.degrees(delta):.1f}°")
```

---

## Ackermann Geometry

### Proper Ackermann Steering

For proper turning without wheel slip, inner and outer wheels need different steering angles:

$$
\cot\delta_o - \cot\delta_i = \frac{W}{L}
$$

Where:
- $\delta_o$: Outer wheel steering angle
- $\delta_i$: Inner wheel steering angle
- $W$: Track width

### Implementation

```python
def ackermann_angles(self, delta_center, track_width=0.4):
    """
    Calculate individual wheel steering angles
    
    Args:
        delta_center: Center steering angle (rad)
        track_width: Distance between left and right wheels (m)
    
    Returns:
        (delta_inner, delta_outer): Steering angles for inner and outer wheels
    """
    if abs(delta_center) < 1e-6:
        return 0, 0
    
    # Turning radius at center
    R = self.L / np.tan(delta_center)
    
    # Inner wheel (sharper angle)
    R_inner = R - track_width / 2
    delta_inner = np.arctan(self.L / R_inner)
    
    # Outer wheel (gentler angle)
    R_outer = R + track_width / 2
    delta_outer = np.arctan(self.L / R_outer)
    
    return delta_inner, delta_outer

# Add to class
FrontWheelDrive.ackermann_angles = ackermann_angles

# Example
robot = FrontWheelDrive(wheelbase=0.5)
delta_inner, delta_outer = robot.ackermann_angles(delta_center=np.pi/6, track_width=0.4)
print(f"Inner wheel: {np.degrees(delta_inner):.1f}°")
print(f"Outer wheel: {np.degrees(delta_outer):.1f}°")
```

---

## Path Following

### Pure Pursuit Controller

```python
def pure_pursuit(self, target_x, target_y, lookahead_distance=1.0):
    """
    Pure pursuit path following algorithm
    
    Args:
        target_x, target_y: Target point coordinates
        lookahead_distance: Lookahead distance (m)
    
    Returns:
        Steering angle to reach target
    """
    x, y, theta = self.state
    
    # Transform target to robot frame
    dx = target_x - x
    dy = target_y - y
    
    # Rotate to robot frame
    dx_robot = dx * np.cos(theta) + dy * np.sin(theta)
    dy_robot = -dx * np.sin(theta) + dy * np.cos(theta)
    
    # Calculate curvature
    curvature = 2 * dy_robot / (lookahead_distance ** 2)
    
    # Calculate steering angle
    delta = np.arctan(curvature * self.L)
    
    # Limit steering angle
    max_delta = np.pi / 3
    delta = np.clip(delta, -max_delta, max_delta)
    
    return delta

# Add to class
FrontWheelDrive.pure_pursuit = pure_pursuit
```

---

## Advantages and Limitations

### Advantages
- **Intuitive**: Similar to car driving
- **Good traction**: Driven wheels pull the robot
- **Predictable**: Well-understood dynamics
- **Simple**: Fewer actuators than 4WD

### Limitations
- **Turning radius**: Cannot turn as sharply as differential drive
- **No zero-radius turns**: Cannot rotate in place
- **Parking**: Difficult in tight spaces
- **Wheel wear**: Front wheels wear faster

---

## Applications

- **Autonomous cars**: Most self-driving cars use FWD or AWD
- **Delivery robots**: Street navigation
- **Agricultural robots**: Row following
- **Warehouse AGVs**: Aisle navigation
- **Outdoor robots**: Natural terrain

---

## Further Reading

- [Ackermann Steering Geometry](https://en.wikipedia.org/wiki/Ackermann_steering_geometry)
- [Pure Pursuit Algorithm](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
- [Autonomous Vehicle Control](https://www.springer.com/gp/book/9783319556642)

