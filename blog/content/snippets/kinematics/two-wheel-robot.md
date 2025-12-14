---
title: "Two-Wheel Differential Drive Robot Kinematics"
date: 2024-12-13
draft: false
category: "kinematics"
tags: ["robotics", "mobile-robots", "differential-drive", "kinematics"]
---

Forward and inverse kinematics for two-wheel differential drive robots, the most common mobile robot configuration.

## Overview

**Differential drive** uses two independently driven wheels on a common axis with a passive caster or ball for balance.

**Advantages:**
- Simple mechanical design
- Easy to control
- Can rotate in place (zero turning radius)
- Cost-effective

**Disadvantages:**
- Cannot move sideways
- Nonholonomic constraints

---

## Robot Configuration

```text
        Front
          ↑
    ┌─────┴─────┐
    │     •     │  ← Center of mass/rotation
    │           │
    ├───┐   ┌───┤
    │ L │   │ R │  ← Left and Right wheels
    └───┘   └───┘
         ↕
         L (wheelbase)
```

**Parameters:**
- $L$: Wheelbase (distance between wheels)
- $r$: Wheel radius
- $(x, y, \theta)$: Robot pose (position and heading)
- $\omega_L, \omega_R$: Left and right wheel angular velocities

---

## Forward Kinematics

### Interactive Differential Drive Simulator

```p5js
let leftSlider, rightSlider;
let robotX = 0;
let robotY = 0;
let robotTheta = 0;
let trail = [];
let wheelbase = 60;
let wheelRadius = 10;

sketch.setup = function() {
  sketch.createCanvas(800, 700);
  
  leftSlider = sketch.createSlider(-10, 10, 5, 0.5);
  leftSlider.position(150, 720);
  leftSlider.style('width', '150px');
  
  rightSlider = sketch.createSlider(-10, 10, 5, 0.5);
  rightSlider.position(450, 720);
  rightSlider.style('width', '150px');
  
  // Reset button
  let resetBtn = sketch.createButton('Reset');
  resetBtn.position(650, 720);
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
  
  let omegaL = leftSlider.value();
  let omegaR = rightSlider.value();
  
  // Title
  sketch.fill(0);
  sketch.textSize(18);
  sketch.textAlign(sketch.CENTER);
  sketch.text('Differential Drive Robot Simulator', sketch.width/2, 25);
  
  // Calculate velocities
  let r = wheelRadius;
  let L = wheelbase;
  let v = r * (omegaR + omegaL) / 2;
  let omega = r * (omegaR - omegaL) / L;
  
  // Update robot pose
  let dt = 0.1;
  robotX += v * sketch.cos(robotTheta) * dt;
  robotY += v * sketch.sin(robotTheta) * dt;
  robotTheta += omega * dt;
  
  // Keep robot on screen
  if (robotX < 50) robotX = 50;
  if (robotX > sketch.width - 50) robotX = sketch.width - 50;
  if (robotY < 50) robotY = 50;
  if (robotY > sketch.height - 150) robotY = sketch.height - 150;
  
  // Add to trail
  if (sketch.frameCount % 3 === 0) {
    trail.push({x: robotX, y: robotY});
    if (trail.length > 200) trail.shift();
  }
  
  // Draw trail
  sketch.stroke(0, 150, 255, 100);
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
  sketch.fill(200, 200, 255);
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.rect(-30, -wheelbase/2 - 5, 60, wheelbase + 10, 5);
  
  // Left wheel
  sketch.fill(omegaL > 0 ? [0, 255, 0] : omegaL < 0 ? [255, 0, 0] : [150]);
  sketch.rect(-35, -wheelbase/2, 10, 20, 2);
  
  // Right wheel
  sketch.fill(omegaR > 0 ? [0, 255, 0] : omegaR < 0 ? [255, 0, 0] : [150]);
  sketch.rect(-35, wheelbase/2 - 20, 10, 20, 2);
  
  // Direction arrow
  sketch.fill(255, 0, 0);
  sketch.noStroke();
  sketch.triangle(30, 0, 15, -8, 15, 8);
  
  // Wheel rotation indicators
  if (Math.abs(omegaL) > 0.1) {
    sketch.stroke(0);
    sketch.strokeWeight(2);
    sketch.noFill();
    let arcAngle = omegaL > 0 ? sketch.HALF_PI : -sketch.HALF_PI;
    sketch.arc(-30, -wheelbase/2 + 10, 15, 15, 0, arcAngle);
  }
  
  if (Math.abs(omegaR) > 0.1) {
    sketch.stroke(0);
    sketch.strokeWeight(2);
    sketch.noFill();
    let arcAngle = omegaR > 0 ? sketch.HALF_PI : -sketch.HALF_PI;
    sketch.arc(-30, wheelbase/2 - 10, 15, 15, 0, arcAngle);
  }
  
  sketch.pop();
  
  // Display info
  sketch.fill(0);
  sketch.textSize(14);
  sketch.textAlign(sketch.LEFT);
  sketch.text(`ωL = ${omegaL.toFixed(1)} rad/s`, 20, 735);
  sketch.text(`ωR = ${omegaR.toFixed(1)} rad/s`, 370, 735);
  
  sketch.textSize(13);
  sketch.text(`Linear velocity: v = ${v.toFixed(2)} m/s`, 20, 50);
  sketch.text(`Angular velocity: ω = ${omega.toFixed(2)} rad/s`, 20, 70);
  sketch.text(`Heading: θ = ${(robotTheta * 180 / sketch.PI % 360).toFixed(1)}°`, 20, 90);
  
  // Motion type indicator
  sketch.textSize(12);
  sketch.fill(0, 0, 200);
  if (Math.abs(omegaL - omegaR) < 0.1) {
    sketch.text('Motion: Straight', 20, 110);
  } else if (Math.abs(omegaL + omegaR) < 0.1) {
    sketch.text('Motion: Rotation in place', 20, 110);
  } else {
    sketch.text('Motion: Arc/Curve', 20, 110);
  }
  
  sketch.fill(150);
  sketch.textSize(11);
  sketch.text('Green wheel = forward, Red wheel = backward', 20, sketch.height - 20);
}
```

---

### Differential Drive Equations

Given wheel velocities $\omega_L$ and $\omega_R$, compute robot velocity:

$$
v = \frac{r(\omega_R + \omega_L)}{2}
$$

$$
\omega = \frac{r(\omega_R - \omega_L)}{L}
$$

Where:
- $v$: Linear velocity (forward)
- $\omega$: Angular velocity (rotation rate)

### Pose Update

$$
\dot{x} = v \cos\theta
$$

$$
\dot{y} = v \sin\theta
$$

$$
\dot{\theta} = \omega
$$

### Python Implementation

```python
import numpy as np

class DifferentialDriveRobot:
    def __init__(self, wheelbase=0.3, wheel_radius=0.05):
        """
        Args:
            wheelbase: Distance between wheels (m)
            wheel_radius: Radius of wheels (m)
        """
        self.L = wheelbase
        self.r = wheel_radius
        
        # State: [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])
    
    def forward_kinematics(self, omega_left, omega_right, dt):
        """
        Update robot pose given wheel velocities
        
        Args:
            omega_left: Left wheel angular velocity (rad/s)
            omega_right: Right wheel angular velocity (rad/s)
            dt: Time step (s)
        
        Returns:
            Updated pose [x, y, theta]
        """
        x, y, theta = self.state
        
        # Robot velocities
        v = self.r * (omega_right + omega_left) / 2
        omega = self.r * (omega_right - omega_left) / self.L
        
        # Update pose
        if abs(omega) < 1e-6:
            # Straight line motion
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
        else:
            # Arc motion
            R = v / omega  # Turning radius
            
            # Instantaneous center of curvature (ICC)
            icc_x = x - R * np.sin(theta)
            icc_y = y + R * np.cos(theta)
            
            # Rotate around ICC
            dtheta = omega * dt
            x = np.cos(dtheta) * (x - icc_x) - np.sin(dtheta) * (y - icc_y) + icc_x
            y = np.sin(dtheta) * (x - icc_x) + np.cos(dtheta) * (y - icc_y) + icc_y
            theta += dtheta
        
        # Normalize angle
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        self.state = np.array([x, y, theta])
        return self.state.copy()
    
    def get_velocity(self, omega_left, omega_right):
        """Get robot linear and angular velocity"""
        v = self.r * (omega_right + omega_left) / 2
        omega = self.r * (omega_right - omega_left) / self.L
        return v, omega

# Example: Straight line
robot = DifferentialDriveRobot(wheelbase=0.3, wheel_radius=0.05)

# Both wheels same speed → straight line
for _ in range(100):
    robot.forward_kinematics(omega_left=5.0, omega_right=5.0, dt=0.01)

print(f"After straight motion: {robot.state}")

# Reset and turn in place
robot.state = np.array([0.0, 0.0, 0.0])

# Opposite wheel speeds → rotation in place
for _ in range(100):
    robot.forward_kinematics(omega_left=-5.0, omega_right=5.0, dt=0.01)

print(f"After rotation: {robot.state}")
```

---

## Inverse Kinematics

### From Robot Velocity to Wheel Velocities

Given desired $v$ and $\omega$, compute wheel velocities:

$$
\omega_L = \frac{v - \omega L/2}{r}
$$

$$
\omega_R = \frac{v + \omega L/2}{r}
$$

### Implementation

```python
def inverse_kinematics(self, v_desired, omega_desired):
    """
    Compute wheel velocities for desired robot motion
    
    Args:
        v_desired: Desired linear velocity (m/s)
        omega_desired: Desired angular velocity (rad/s)
    
    Returns:
        (omega_left, omega_right) in rad/s
    """
    omega_left = (v_desired - omega_desired * self.L / 2) / self.r
    omega_right = (v_desired + omega_desired * self.L / 2) / self.r
    
    return omega_left, omega_right

# Add to DifferentialDriveRobot class
DifferentialDriveRobot.inverse_kinematics = inverse_kinematics

# Example: Move forward at 0.5 m/s
robot = DifferentialDriveRobot()
omega_l, omega_r = robot.inverse_kinematics(v_desired=0.5, omega_desired=0)
print(f"Straight motion: ωL={omega_l:.2f}, ωR={omega_r:.2f} rad/s")

# Turn in place at 1 rad/s
omega_l, omega_r = robot.inverse_kinematics(v_desired=0, omega_desired=1.0)
print(f"Rotation: ωL={omega_l:.2f}, ωR={omega_r:.2f} rad/s")

# Arc motion
omega_l, omega_r = robot.inverse_kinematics(v_desired=0.5, omega_desired=0.5)
print(f"Arc motion: ωL={omega_l:.2f}, ωR={omega_r:.2f} rad/s")
```

---

## Trajectory Following

### Point-to-Point Navigation

```python
def navigate_to_point(self, target_x, target_y, kp_linear=1.0, kp_angular=2.0):
    """
    Simple proportional controller to reach target point
    
    Args:
        target_x, target_y: Target position
        kp_linear: Linear velocity gain
        kp_angular: Angular velocity gain
    
    Returns:
        (omega_left, omega_right) wheel velocities
    """
    x, y, theta = self.state
    
    # Error in position
    dx = target_x - x
    dy = target_y - y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Desired heading
    target_theta = np.arctan2(dy, dx)
    
    # Heading error
    theta_error = np.arctan2(np.sin(target_theta - theta),
                             np.cos(target_theta - theta))
    
    # Control law
    v_desired = kp_linear * distance
    omega_desired = kp_angular * theta_error
    
    # Limit velocities
    v_max = 1.0  # m/s
    omega_max = 2.0  # rad/s
    v_desired = np.clip(v_desired, -v_max, v_max)
    omega_desired = np.clip(omega_desired, -omega_max, omega_max)
    
    # Inverse kinematics
    omega_left, omega_right = self.inverse_kinematics(v_desired, omega_desired)
    
    return omega_left, omega_right, distance

# Add to class
DifferentialDriveRobot.navigate_to_point = navigate_to_point

# Example: Navigate to (2, 3)
robot = DifferentialDriveRobot()
target = (2.0, 3.0)

dt = 0.05
trajectory = [robot.state.copy()]

for _ in range(500):
    omega_l, omega_r, dist = robot.navigate_to_point(*target)
    
    if dist < 0.05:  # Reached target
        break
    
    robot.forward_kinematics(omega_l, omega_r, dt)
    trajectory.append(robot.state.copy())

print(f"Final position: {robot.state}")
print(f"Distance to target: {dist:.4f} m")

# Plot trajectory
import matplotlib.pyplot as plt
trajectory = np.array(trajectory)
plt.figure(figsize=(8, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Path')
plt.plot(0, 0, 'go', markersize=10, label='Start')
plt.plot(target[0], target[1], 'ro', markersize=10, label='Target')
plt.grid(True)
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.title('Differential Drive Navigation')
plt.show()
```

---

## Velocity Constraints

### Maximum Wheel Velocities

```python
def apply_velocity_limits(self, omega_left, omega_right, omega_max=10.0):
    """
    Apply wheel velocity limits while preserving motion direction
    
    Args:
        omega_left, omega_right: Desired wheel velocities
        omega_max: Maximum wheel velocity (rad/s)
    
    Returns:
        Limited wheel velocities
    """
    # Find scaling factor
    max_omega = max(abs(omega_left), abs(omega_right))
    
    if max_omega > omega_max:
        scale = omega_max / max_omega
        omega_left *= scale
        omega_right *= scale
    
    return omega_left, omega_right

# Add to class
DifferentialDriveRobot.apply_velocity_limits = apply_velocity_limits
```

---

## Odometry

### Dead Reckoning from Wheel Encoders

```python
class DifferentialDriveWithOdometry(DifferentialDriveRobot):
    def __init__(self, wheelbase=0.3, wheel_radius=0.05, 
                 encoder_resolution=2048):
        super().__init__(wheelbase, wheel_radius)
        self.encoder_resolution = encoder_resolution
        self.encoder_left = 0
        self.encoder_right = 0
    
    def update_from_encoders(self, encoder_left, encoder_right, dt):
        """
        Update pose from encoder readings
        
        Args:
            encoder_left, encoder_right: Encoder tick counts
            dt: Time since last update
        """
        # Encoder deltas
        delta_left = encoder_left - self.encoder_left
        delta_right = encoder_right - self.encoder_right
        
        # Update stored values
        self.encoder_left = encoder_left
        self.encoder_right = encoder_right
        
        # Convert to angular displacement
        angle_left = 2 * np.pi * delta_left / self.encoder_resolution
        angle_right = 2 * np.pi * delta_right / self.encoder_resolution
        
        # Wheel velocities
        omega_left = angle_left / dt
        omega_right = angle_right / dt
        
        # Update pose
        return self.forward_kinematics(omega_left, omega_right, dt)

# Example
robot = DifferentialDriveWithOdometry()

# Simulate encoder readings
encoder_left = 0
encoder_right = 0

for i in range(100):
    # Simulate forward motion
    encoder_left += 10  # 10 ticks per iteration
    encoder_right += 10
    
    robot.update_from_encoders(encoder_left, encoder_right, dt=0.01)

print(f"Odometry position: {robot.state}")
```

---

## Motion Primitives

### Common Maneuvers

```python
class MotionPrimitives:
    @staticmethod
    def straight(distance, speed=0.5):
        """Move straight for given distance"""
        duration = distance / speed
        return {
            'v': speed,
            'omega': 0,
            'duration': duration
        }
    
    @staticmethod
    def rotate(angle, angular_speed=1.0):
        """Rotate in place by given angle"""
        duration = abs(angle) / angular_speed
        omega = angular_speed if angle > 0 else -angular_speed
        return {
            'v': 0,
            'omega': omega,
            'duration': duration
        }
    
    @staticmethod
    def arc(radius, angle, speed=0.5):
        """Follow circular arc"""
        arc_length = radius * abs(angle)
        duration = arc_length / speed
        omega = speed / radius
        if angle < 0:
            omega = -omega
        return {
            'v': speed,
            'omega': omega,
            'duration': duration
        }

# Example: Execute motion sequence
robot = DifferentialDriveRobot()

sequence = [
    MotionPrimitives.straight(distance=1.0, speed=0.5),
    MotionPrimitives.rotate(angle=np.pi/2, angular_speed=1.0),
    MotionPrimitives.straight(distance=0.5, speed=0.5),
    MotionPrimitives.arc(radius=0.5, angle=np.pi, speed=0.3)
]

dt = 0.01
for primitive in sequence:
    t = 0
    while t < primitive['duration']:
        omega_l, omega_r = robot.inverse_kinematics(
            primitive['v'], primitive['omega']
        )
        robot.forward_kinematics(omega_l, omega_r, dt)
        t += dt

print(f"Final pose after sequence: {robot.state}")
```

---

## Simulation and Visualization

```python
def simulate_and_plot(robot, control_function, duration=10.0, dt=0.01):
    """
    Simulate robot motion and plot trajectory
    
    Args:
        robot: DifferentialDriveRobot instance
        control_function: Function(robot, t) -> (omega_left, omega_right)
        duration: Simulation duration (s)
        dt: Time step (s)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    trajectory = []
    t = 0
    
    while t < duration:
        # Get control inputs
        omega_l, omega_r = control_function(robot, t)
        
        # Update robot
        robot.forward_kinematics(omega_l, omega_r, dt)
        trajectory.append(robot.state.copy())
        
        t += dt
    
    trajectory = np.array(trajectory)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Trajectory
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    
    # Draw robot orientation at intervals
    for i in range(0, len(trajectory), len(trajectory)//10):
        x, y, theta = trajectory[i]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)
        ax1.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05, fc='r', ec='r')
    
    ax1.grid(True)
    ax1.axis('equal')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.set_title('Robot Trajectory')
    
    # Heading over time
    time = np.arange(len(trajectory)) * dt
    ax2.plot(time, np.degrees(trajectory[:, 2]), 'b-', linewidth=2)
    ax2.grid(True)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Heading (degrees)')
    ax2.set_title('Robot Heading vs Time')
    
    plt.tight_layout()
    plt.show()

# Example: Figure-8 pattern
def figure_eight_control(robot, t):
    """Control for figure-8 trajectory"""
    v = 0.5  # m/s
    omega = 0.5 * np.sin(t)  # rad/s
    return robot.inverse_kinematics(v, omega)

robot = DifferentialDriveRobot()
simulate_and_plot(robot, figure_eight_control, duration=20.0)
```

---

## Further Reading

- [Siegwart, R. "Introduction to Autonomous Mobile Robots"](https://mitpress.mit.edu/9780262015356/introduction-to-autonomous-mobile-robots/)
- [Differential Drive Kinematics](https://www.cs.columbia.edu/~allen/F17/NOTES/icckinematics.pdf)
- [Mobile Robot Control](https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf)

