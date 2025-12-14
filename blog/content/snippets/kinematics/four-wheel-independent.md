---
title: "Four-Wheel Independent Drive Kinematics"
date: 2024-12-13
draft: false
category: "kinematics"
tags: ["robotics", "mobile-robots", "four-wheel-drive", "kinematics"]
---

Forward and inverse kinematics for four-wheel independent drive robots with individual motor control.

## Overview

**Four-Wheel Independent Drive** uses four independently driven wheels, providing high maneuverability and redundancy.

**Advantages:**
- High traction and power
- Redundancy (can operate with one wheel failure)
- Good for rough terrain
- Simple mechanical design

**Disadvantages:**
- Cannot move sideways (nonholonomic)
- Wheel coordination required
- Potential for wheel slip/skidding

---

## Robot Configuration

```text
        Front
          ↑
    1 ─────── 2
    │    •    │  ← Center of rotation
    │         │
    4 ─────── 3
    
    Wheels: 1=FL, 2=FR, 3=RR, 4=RL
```

**Parameters:**
- $L$: Wheelbase (front-back distance)
- $W$: Track width (left-right distance)
- $r$: Wheel radius
- $(x, y, \theta)$: Robot pose

---

## Interactive Four-Wheel Drive Simulator

```p5js
let w1Slider, w2Slider, w3Slider, w4Slider;
let robotX = 0;
let robotY = 0;
let robotTheta = 0;
let trail = [];
let wheelRadius = 10;
let wheelbase = 80;
let trackWidth = 60;

sketch.setup = function() {
  sketch.createCanvas(800, 800);
  
  w1Slider = sketch.createSlider(-10, 10, 5, 0.5);
  w1Slider.position(50, 820);
  w1Slider.style('width', '120px');
  
  w2Slider = sketch.createSlider(-10, 10, 5, 0.5);
  w2Slider.position(230, 820);
  w2Slider.style('width', '120px');
  
  w3Slider = sketch.createSlider(-10, 10, 5, 0.5);
  w3Slider.position(410, 820);
  w3Slider.style('width', '120px');
  
  w4Slider = sketch.createSlider(-10, 10, 5, 0.5);
  w4Slider.position(590, 820);
  w4Slider.style('width', '120px');
  
  let resetBtn = sketch.createButton('Reset');
  resetBtn.position(720, 820);
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
  
  let omega1 = w1Slider.value();
  let omega2 = w2Slider.value();
  let omega3 = w3Slider.value();
  let omega4 = w4Slider.value();
  
  // Title
  sketch.fill(0);
  sketch.textSize(18);
  sketch.textAlign(sketch.CENTER);
  sketch.text('Four-Wheel Independent Drive Robot', sketch.width/2, 25);
  
  // Calculate average velocities (simplified model)
  let r = wheelRadius;
  let omegaLeft = (omega1 + omega4) / 2;
  let omegaRight = (omega2 + omega3) / 2;
  
  let v = r * (omegaRight + omegaLeft) / 2;
  let omega = r * (omegaRight - omegaLeft) / trackWidth;
  
  // Update robot pose
  let dt = 0.1;
  robotX += v * sketch.cos(robotTheta) * dt;
  robotY += v * sketch.sin(robotTheta) * dt;
  robotTheta += omega * dt;
  
  // Keep robot on screen
  if (robotX < 80) robotX = 80;
  if (robotX > sketch.width - 80) robotX = sketch.width - 80;
  if (robotY < 80) robotY = 80;
  if (robotY > sketch.height - 250) robotY = sketch.height - 250;
  
  // Add to trail
  if (sketch.frameCount % 3 === 0) {
    trail.push({x: robotX, y: robotY});
    if (trail.length > 200) trail.shift();
  }
  
  // Draw trail
  sketch.stroke(0, 200, 150, 100);
  sketch.strokeWeight(2);
  sketch.noFill();
  sketch.beginShape();
  for (let p of trail) {
    sketch.vertex(p.x, p.y);
  }
  sketch.endShape();
  
  // Draw robot
  sketch.push();
  sketch.translate(robotX, robotY);
  sketch.rotate(robotTheta);
  
  // Robot body
  sketch.fill(200, 220, 255);
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.rect(-wheelbase/2, -trackWidth/2, wheelbase, trackWidth, 5);
  
  // Draw four wheels
  let wheels = [
    {x: wheelbase/2, y: -trackWidth/2, omega: omega1, label: '1 (FL)'},
    {x: wheelbase/2, y: trackWidth/2, omega: omega2, label: '2 (FR)'},
    {x: -wheelbase/2, y: trackWidth/2, omega: omega3, label: '3 (RR)'},
    {x: -wheelbase/2, y: -trackWidth/2, omega: omega4, label: '4 (RL)'}
  ];
  
  for (let wheel of wheels) {
    sketch.push();
    sketch.translate(wheel.x, wheel.y);
    
    // Wheel color based on direction
    if (wheel.omega > 0.1) {
      sketch.fill(0, 255, 0);
    } else if (wheel.omega < -0.1) {
      sketch.fill(255, 0, 0);
    } else {
      sketch.fill(150);
    }
    
    sketch.stroke(0);
    sketch.strokeWeight(2);
    sketch.rect(-8, -15, 16, 30, 2);
    
    // Rotation indicator
    if (Math.abs(wheel.omega) > 0.1) {
      sketch.stroke(0);
      sketch.strokeWeight(2);
      sketch.noFill();
      let arcAngle = wheel.omega > 0 ? sketch.HALF_PI : -sketch.HALF_PI;
      sketch.arc(0, 0, 20, 20, 0, arcAngle);
    }
    
    // Label
    sketch.fill(255);
    sketch.noStroke();
    sketch.textSize(8);
    sketch.textAlign(sketch.CENTER, sketch.CENTER);
    sketch.text(wheel.label.charAt(0), 0, 0);
    
    sketch.pop();
  }
  
  // Direction arrow
  sketch.fill(255, 0, 0);
  sketch.noStroke();
  sketch.triangle(wheelbase/2 + 15, 0, wheelbase/2, -8, wheelbase/2, 8);
  
  // Center point
  sketch.fill(0);
  sketch.noStroke();
  sketch.circle(0, 0, 8);
  
  sketch.pop();
  
  // Display info
  sketch.fill(0);
  sketch.textSize(12);
  sketch.textAlign(sketch.LEFT);
  sketch.text(`ω₁=${omega1.toFixed(1)}`, 20, 835);
  sketch.text(`ω₂=${omega2.toFixed(1)}`, 200, 835);
  sketch.text(`ω₃=${omega3.toFixed(1)}`, 380, 835);
  sketch.text(`ω₄=${omega4.toFixed(1)}`, 560, 835);
  
  sketch.textSize(13);
  sketch.text(`Linear velocity: v = ${v.toFixed(2)} m/s`, 20, 50);
  sketch.text(`Angular velocity: ω = ${omega.toFixed(2)} rad/s`, 20, 70);
  sketch.text(`Heading: θ = ${(robotTheta * 180 / sketch.PI % 360).toFixed(1)}°`, 20, 90);
  
  // Wheel slip warning
  let frontDiff = Math.abs(omega1 - omega2);
  let rearDiff = Math.abs(omega4 - omega3);
  if (frontDiff > 2 || rearDiff > 2) {
    sketch.fill(255, 0, 0);
    sketch.text('⚠ Warning: Large wheel speed difference may cause slip!', 20, 110);
  }
  
  sketch.fill(150);
  sketch.textSize(11);
  sketch.text('Green = forward, Red = backward, Gray = stopped', 20, sketch.height - 20);
}
```

---

## Forward Kinematics

### Simplified Model (Ackermann Approximation)

For four-wheel drive, we can approximate using average left and right velocities:

$$
v = \frac{r(\omega_R + \omega_L)}{2}
$$

$$
\omega = \frac{r(\omega_R - \omega_L)}{W}
$$

Where:
- $\omega_L = \frac{\omega_1 + \omega_4}{2}$ (average left wheels)
- $\omega_R = \frac{\omega_2 + \omega_3}{2}$ (average right wheels)

### Python Implementation

```python
import numpy as np

class FourWheelDrive:
    def __init__(self, wheelbase=0.5, track_width=0.4, wheel_radius=0.1):
        """
        Args:
            wheelbase: Distance between front and rear axles (m)
            track_width: Distance between left and right wheels (m)
            wheel_radius: Radius of wheels (m)
        """
        self.L = wheelbase
        self.W = track_width
        self.r = wheel_radius
        
        # State: [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])
    
    def forward_kinematics(self, omega1, omega2, omega3, omega4, dt):
        """
        Update robot pose from wheel velocities
        
        Args:
            omega1, omega2, omega3, omega4: Wheel angular velocities (rad/s)
                1=FL, 2=FR, 3=RR, 4=RL
            dt: Time step (s)
        
        Returns:
            Updated pose [x, y, theta]
        """
        x, y, theta = self.state
        
        # Average left and right velocities
        omega_left = (omega1 + omega4) / 2
        omega_right = (omega2 + omega3) / 2
        
        # Robot velocities
        v = self.r * (omega_right + omega_left) / 2
        omega = self.r * (omega_right - omega_left) / self.W
        
        # Update pose
        if abs(omega) < 1e-6:
            # Straight line motion
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
        else:
            # Arc motion
            R = v / omega
            x += R * (np.sin(theta + omega * dt) - np.sin(theta))
            y += R * (-np.cos(theta + omega * dt) + np.cos(theta))
            theta += omega * dt
        
        # Normalize angle
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        self.state = np.array([x, y, theta])
        return self.state.copy()
    
    def get_velocity(self, omega1, omega2, omega3, omega4):
        """Get robot linear and angular velocity"""
        omega_left = (omega1 + omega4) / 2
        omega_right = (omega2 + omega3) / 2
        
        v = self.r * (omega_right + omega_left) / 2
        omega = self.r * (omega_right - omega_left) / self.W
        
        return v, omega

# Example
robot = FourWheelDrive(wheelbase=0.5, track_width=0.4, wheel_radius=0.1)

# All wheels same speed → straight line
for _ in range(100):
    robot.forward_kinematics(5.0, 5.0, 5.0, 5.0, dt=0.01)

print(f"After straight motion: {robot.state}")
```

---

## Inverse Kinematics

### From Robot Velocity to Wheel Velocities

Given desired $v$ and $\omega$:

$$
\omega_L = \frac{v - \omega W/2}{r}
$$

$$
\omega_R = \frac{v + \omega W/2}{r}
$$

Then distribute to individual wheels:
- $\omega_1 = \omega_4 = \omega_L$ (left wheels)
- $\omega_2 = \omega_3 = \omega_R$ (right wheels)

### Implementation

```python
def inverse_kinematics(self, v_desired, omega_desired):
    """
    Compute wheel velocities for desired motion
    
    Args:
        v_desired: Desired linear velocity (m/s)
        omega_desired: Desired angular velocity (rad/s)
    
    Returns:
        (omega1, omega2, omega3, omega4) wheel velocities
    """
    omega_left = (v_desired - omega_desired * self.W / 2) / self.r
    omega_right = (v_desired + omega_desired * self.W / 2) / self.r
    
    # Distribute to all four wheels
    omega1 = omega_left   # FL
    omega2 = omega_right  # FR
    omega3 = omega_right  # RR
    omega4 = omega_left   # RL
    
    return omega1, omega2, omega3, omega4

# Add to FourWheelDrive class
FourWheelDrive.inverse_kinematics = inverse_kinematics

# Example: Move forward at 0.5 m/s
robot = FourWheelDrive()
w1, w2, w3, w4 = robot.inverse_kinematics(v_desired=0.5, omega_desired=0)
print(f"Straight: ω₁={w1:.2f}, ω₂={w2:.2f}, ω₃={w3:.2f}, ω₄={w4:.2f}")

# Turn in place
w1, w2, w3, w4 = robot.inverse_kinematics(v_desired=0, omega_desired=1.0)
print(f"Rotation: ω₁={w1:.2f}, ω₂={w2:.2f}, ω₃={w3:.2f}, ω₄={w4:.2f}")
```

---

## Wheel Slip Considerations

### Slip Detection

```python
def check_wheel_slip(self, omega1, omega2, omega3, omega4, threshold=2.0):
    """
    Check for potential wheel slip
    
    Returns:
        (has_slip, slip_info)
    """
    # Check front wheels
    front_diff = abs(omega1 - omega2)
    
    # Check rear wheels
    rear_diff = abs(omega4 - omega3)
    
    # Check left wheels
    left_diff = abs(omega1 - omega4)
    
    # Check right wheels
    right_diff = abs(omega2 - omega3)
    
    slip_info = {
        'front_diff': front_diff,
        'rear_diff': rear_diff,
        'left_diff': left_diff,
        'right_diff': right_diff
    }
    
    has_slip = (front_diff > threshold or rear_diff > threshold or
                left_diff > threshold or right_diff > threshold)
    
    return has_slip, slip_info

# Add to class
FourWheelDrive.check_wheel_slip = check_wheel_slip
```

---

## Advantages Over Two-Wheel Drive

1. **Higher Traction**: Four contact points with ground
2. **Better Climbing**: More power for obstacles
3. **Redundancy**: Can operate with one wheel failure
4. **Stability**: Better weight distribution

---

## Practical Considerations

### 1. Wheel Synchronization

```python
def synchronized_control(self, v_desired, omega_desired, max_accel=1.0, dt=0.01):
    """
    Smooth wheel velocity changes to prevent slip
    
    Args:
        max_accel: Maximum acceleration (rad/s²)
    """
    target_w1, target_w2, target_w3, target_w4 = self.inverse_kinematics(
        v_desired, omega_desired
    )
    
    # Gradually adjust to target (simple rate limiting)
    # In practice, use PID controllers for each wheel
    
    return target_w1, target_w2, target_w3, target_w4
```

### 2. Terrain Adaptation

```python
def terrain_adaptive_control(self, terrain_type='flat'):
    """
    Adjust control parameters based on terrain
    
    Args:
        terrain_type: 'flat', 'rough', 'slippery', 'incline'
    """
    if terrain_type == 'rough':
        # Increase individual wheel control
        return {'independent': True, 'slip_threshold': 3.0}
    elif terrain_type == 'slippery':
        # Reduce acceleration, increase monitoring
        return {'max_accel': 0.5, 'slip_threshold': 1.0}
    elif terrain_type == 'incline':
        # Increase rear wheel power
        return {'rear_bias': 1.2}
    else:
        return {'independent': False, 'slip_threshold': 2.0}
```

---

## Applications

- **Off-road robots**: High traction for rough terrain
- **Heavy-duty platforms**: Carrying large payloads
- **Mars rovers**: Redundancy and reliability
- **Agricultural robots**: Uneven terrain navigation
- **Construction robots**: High power requirements

---

## Further Reading

- [Four-Wheel Drive Kinematics](https://www.sciencedirect.com/topics/engineering/four-wheel-drive)
- [Mars Rover Mobility](https://mars.nasa.gov/mars2020/spacecraft/rover/wheels/)
- [Mobile Robot Control](https://www.springer.com/gp/book/9783319015446)

