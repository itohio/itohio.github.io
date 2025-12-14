---
title: "Mecanum Wheel Kinematics"
date: 2024-12-13
draft: false
category: "kinematics"
tags: ["robotics", "mobile-robots", "mecanum-wheels", "omnidirectional", "kinematics"]
---

Forward and inverse kinematics for mecanum wheel robots, enabling omnidirectional motion including sideways movement.

## Overview

**Mecanum wheels** have rollers at 45° to the wheel axis, allowing omnidirectional movement without changing orientation.

**Advantages:**
- Omnidirectional motion (can move in any direction)
- Can translate while rotating
- No need to reorient before moving
- Excellent maneuverability in tight spaces

**Disadvantages:**
- More complex mechanically
- Lower efficiency than standard wheels
- Sensitive to floor conditions
- More expensive

---

## Wheel Configuration

```text
     Front
       ↑
   1       2
    \  •  /     • = Robot center
    /     \
   4       3

Wheel 1 (FL): Rollers at +45°
Wheel 2 (FR): Rollers at -45°
Wheel 3 (RR): Rollers at +45°
Wheel 4 (RL): Rollers at -45°
```

**Parameters:**
- $l_x$: Half of wheelbase (front-back distance / 2)
- $l_y$: Half of track width (left-right distance / 2)
- $r$: Wheel radius
- $(v_x, v_y, \omega)$: Robot velocity (forward, sideways, rotation)

---

## Forward Kinematics

### Interactive Mecanum Drive Simulator

```p5js
let vxSlider, vySlider, omegaSlider;
let robotX = 0;
let robotY = 0;
let robotTheta = 0;
let trail = [];

sketch.setup = function() {
  sketch.createCanvas(800, 750);
  
  vxSlider = sketch.createSlider(-5, 5, 0, 0.5);
  vxSlider.position(150, 770);
  vxSlider.style('width', '120px');
  
  vySlider = sketch.createSlider(-5, 5, 0, 0.5);
  vySlider.position(350, 770);
  vySlider.style('width', '120px');
  
  omegaSlider = sketch.createSlider(-2, 2, 0, 0.1);
  omegaSlider.position(550, 770);
  omegaSlider.style('width', '120px');
  
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
  
  let vx = vxSlider.value();
  let vy = vySlider.value();
  let omega = omegaSlider.value();
  
  // Title
  sketch.fill(0);
  sketch.textSize(18);
  sketch.textAlign(sketch.CENTER);
  sketch.text('Mecanum Wheel Robot - Omnidirectional Motion', sketch.width/2, 25);
  
  // Update robot pose (in world frame)
  let dt = 0.1;
  let vx_world = vx * sketch.cos(robotTheta) - vy * sketch.sin(robotTheta);
  let vy_world = vx * sketch.sin(robotTheta) + vy * sketch.cos(robotTheta);
  
  robotX += vx_world * dt;
  robotY += vy_world * dt;
  robotTheta += omega * dt;
  
  // Keep robot on screen
  if (robotX < 60) robotX = 60;
  if (robotX > sketch.width - 60) robotX = sketch.width - 60;
  if (robotY < 60) robotY = 60;
  if (robotY > sketch.height - 200) robotY = sketch.height - 200;
  
  // Add to trail
  if (sketch.frameCount % 3 === 0) {
    trail.push({x: robotX, y: robotY});
    if (trail.length > 200) trail.shift();
  }
  
  // Draw trail
  sketch.stroke(255, 150, 0, 100);
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
  sketch.fill(220, 220, 255);
  sketch.stroke(0);
  sketch.strokeWeight(2);
  sketch.rect(-40, -40, 80, 80, 5);
  
  // Draw mecanum wheels at corners
  let wheelSize = 15;
  let positions = [
    {x: -40, y: -40, angle: sketch.PI/4, label: '1'},      // FL
    {x: 40, y: -40, angle: -sketch.PI/4, label: '2'},      // FR
    {x: 40, y: 40, angle: sketch.PI/4, label: '3'},        // RR
    {x: -40, y: 40, angle: -sketch.PI/4, label: '4'}       // RL
  ];
  
  for (let wheel of positions) {
    sketch.push();
    sketch.translate(wheel.x, wheel.y);
    
    // Wheel
    sketch.fill(100);
    sketch.rect(-wheelSize/2, -wheelSize/2, wheelSize, wheelSize, 2);
    
    // Roller direction indicator
    sketch.stroke(255, 200, 0);
    sketch.strokeWeight(2);
    sketch.line(
      -wheelSize/2 * sketch.cos(wheel.angle), 
      -wheelSize/2 * sketch.sin(wheel.angle),
      wheelSize/2 * sketch.cos(wheel.angle), 
      wheelSize/2 * sketch.sin(wheel.angle)
    );
    
    // Wheel number
    sketch.fill(255);
    sketch.noStroke();
    sketch.textSize(10);
    sketch.textAlign(sketch.CENTER, sketch.CENTER);
    sketch.text(wheel.label, 0, 0);
    
    sketch.pop();
  }
  
  // Direction arrow (forward)
  sketch.fill(255, 0, 0);
  sketch.noStroke();
  sketch.triangle(40, 0, 25, -8, 25, 8);
  
  // Velocity vector
  if (vx !== 0 || vy !== 0) {
    sketch.stroke(0, 200, 0);
    sketch.strokeWeight(3);
    let scale = 5;
    sketch.line(0, 0, vx * scale, vy * scale);
    sketch.fill(0, 200, 0);
    sketch.noStroke();
    sketch.circle(vx * scale, vy * scale, 8);
  }
  
  // Rotation indicator
  if (Math.abs(omega) > 0.1) {
    sketch.noFill();
    sketch.stroke(0, 0, 255);
    sketch.strokeWeight(2);
    let arcAngle = omega > 0 ? sketch.HALF_PI : -sketch.HALF_PI;
    sketch.arc(0, 0, 50, 50, 0, arcAngle);
  }
  
  sketch.pop();
  
  // Display info
  sketch.fill(0);
  sketch.textSize(14);
  sketch.textAlign(sketch.LEFT);
  sketch.text(`vx = ${vx.toFixed(1)} m/s`, 20, 785);
  sketch.text(`vy = ${vy.toFixed(1)} m/s`, 280, 785);
  sketch.text(`ω = ${omega.toFixed(1)} rad/s`, 500, 785);
  
  sketch.textSize(13);
  sketch.text(`Robot velocity (robot frame): [${vx.toFixed(1)}, ${vy.toFixed(1)}]`, 20, 50);
  sketch.text(`Heading: θ = ${(robotTheta * 180 / sketch.PI % 360).toFixed(1)}°`, 20, 70);
  
  // Motion type
  sketch.textSize(12);
  sketch.fill(0, 0, 200);
  if (Math.abs(vx) < 0.1 && Math.abs(vy) < 0.1 && Math.abs(omega) > 0.1) {
    sketch.text('Motion: Pure rotation', 20, 90);
  } else if (Math.abs(vx) > 0.1 && Math.abs(vy) < 0.1 && Math.abs(omega) < 0.1) {
    sketch.text('Motion: Forward/Backward', 20, 90);
  } else if (Math.abs(vx) < 0.1 && Math.abs(vy) > 0.1 && Math.abs(omega) < 0.1) {
    sketch.text('Motion: Sideways (strafe)', 20, 90);
  } else if (Math.abs(vx) > 0.1 && Math.abs(vy) > 0.1 && Math.abs(omega) < 0.1) {
    sketch.text('Motion: Diagonal', 20, 90);
  } else if ((Math.abs(vx) > 0.1 || Math.abs(vy) > 0.1) && Math.abs(omega) > 0.1) {
    sketch.text('Motion: Combined translation + rotation', 20, 90);
  }
  
  sketch.fill(150);
  sketch.textSize(11);
  sketch.text('Green arrow = velocity direction, Blue arc = rotation', 20, sketch.height - 20);
}
```

---

### From Wheel Velocities to Robot Velocity

Given wheel angular velocities $[\omega_1, \omega_2, \omega_3, \omega_4]$:

$$
\begin{bmatrix} v_x \\\\ v_y \\\\ \omega \end{bmatrix} = 
\frac{r}{4} \begin{bmatrix}
1 & 1 & 1 & 1 \\\\
-1 & 1 & 1 & -1 \\\\
-\frac{1}{l_x + l_y} & \frac{1}{l_x + l_y} & -\frac{1}{l_x + l_y} & \frac{1}{l_x + l_y}
\end{bmatrix}
\begin{bmatrix} \omega_1 \\\\ \omega_2 \\\\ \omega_3 \\\\ \omega_4 \end{bmatrix}
$$

### Python Implementation

```python
import numpy as np

class MecanumRobot:
    def __init__(self, lx=0.2, ly=0.2, wheel_radius=0.05):
        """
        Args:
            lx: Half of wheelbase (m)
            ly: Half of track width (m)
            wheel_radius: Radius of wheels (m)
        """
        self.lx = lx
        self.ly = ly
        self.r = wheel_radius
        
        # State: [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])
        
        # Forward kinematics matrix
        self.FK_matrix = (self.r / 4) * np.array([
            [1,  1,  1,  1],
            [-1, 1,  1, -1],
            [-1/(lx+ly), 1/(lx+ly), -1/(lx+ly), 1/(lx+ly)]
        ])
    
    def forward_kinematics(self, wheel_velocities, dt):
        """
        Compute robot velocity from wheel velocities
        
        Args:
            wheel_velocities: [ω1, ω2, ω3, ω4] (rad/s)
            dt: Time step (s)
        
        Returns:
            Robot velocity [vx, vy, ω] in robot frame
        """
        wheel_velocities = np.array(wheel_velocities)
        
        # Robot velocity in robot frame
        robot_vel = self.FK_matrix @ wheel_velocities
        vx_robot, vy_robot, omega = robot_vel
        
        # Transform to world frame
        x, y, theta = self.state
        
        vx_world = vx_robot * np.cos(theta) - vy_robot * np.sin(theta)
        vy_world = vx_robot * np.sin(theta) + vy_robot * np.cos(theta)
        
        # Update pose
        self.state[0] += vx_world * dt
        self.state[1] += vy_world * dt
        self.state[2] += omega * dt
        
        # Normalize angle
        self.state[2] = np.arctan2(np.sin(self.state[2]), 
                                   np.cos(self.state[2]))
        
        return self.state.copy()
    
    def get_robot_velocity(self, wheel_velocities):
        """Get robot velocity without updating state"""
        wheel_velocities = np.array(wheel_velocities)
        return self.FK_matrix @ wheel_velocities

# Example: Forward motion
robot = MecanumRobot(lx=0.2, ly=0.2, wheel_radius=0.05)

# All wheels same speed → forward motion
for _ in range(100):
    robot.forward_kinematics([10, 10, 10, 10], dt=0.01)

print(f"Forward motion: {robot.state}")

# Reset and move sideways
robot.state = np.array([0.0, 0.0, 0.0])

# Alternating wheel pattern → sideways motion
for _ in range(100):
    robot.forward_kinematics([-10, 10, 10, -10], dt=0.01)

print(f"Sideways motion: {robot.state}")
```

---

## Inverse Kinematics

### From Robot Velocity to Wheel Velocities

Given desired robot velocity $(v_x, v_y, \omega)$ in robot frame:

$$
\begin{bmatrix} \omega_1 \\\\ \omega_2 \\\\ \omega_3 \\\\ \omega_4 \end{bmatrix} = 
\frac{1}{r} \begin{bmatrix}
1 & -1 & -(l_x + l_y) \\\\
1 & 1 & (l_x + l_y) \\\\
1 & 1 & -(l_x + l_y) \\\\
1 & -1 & (l_x + l_y)
\end{bmatrix}
\begin{bmatrix} v_x \\\\ v_y \\\\ \omega \end{bmatrix}
$$

### Implementation

```python
def inverse_kinematics(self, vx, vy, omega):
    """
    Compute wheel velocities for desired robot motion
    
    Args:
        vx: Desired forward velocity (m/s) in robot frame
        vy: Desired sideways velocity (m/s) in robot frame
        omega: Desired angular velocity (rad/s)
    
    Returns:
        [ω1, ω2, ω3, ω4] wheel velocities (rad/s)
    """
    # Inverse kinematics matrix
    IK_matrix = (1 / self.r) * np.array([
        [1, -1, -(self.lx + self.ly)],
        [1,  1,  (self.lx + self.ly)],
        [1,  1, -(self.lx + self.ly)],
        [1, -1,  (self.lx + self.ly)]
    ])
    
    robot_vel = np.array([vx, vy, omega])
    wheel_velocities = IK_matrix @ robot_vel
    
    return wheel_velocities

# Add to MecanumRobot class
MecanumRobot.inverse_kinematics = inverse_kinematics

# Example usage
robot = MecanumRobot()

# Forward motion
wheels = robot.inverse_kinematics(vx=0.5, vy=0, omega=0)
print(f"Forward: {wheels}")

# Sideways motion
wheels = robot.inverse_kinematics(vx=0, vy=0.5, omega=0)
print(f"Sideways: {wheels}")

# Diagonal motion
wheels = robot.inverse_kinematics(vx=0.5, vy=0.5, omega=0)
print(f"Diagonal: {wheels}")

# Rotation in place
wheels = robot.inverse_kinematics(vx=0, vy=0, omega=1.0)
print(f"Rotation: {wheels}")

# Combined motion
wheels = robot.inverse_kinematics(vx=0.3, vy=0.2, omega=0.5)
print(f"Combined: {wheels}")
```

---

## Velocity Constraints

### Apply Wheel Velocity Limits

```python
def apply_velocity_limits(self, wheel_velocities, omega_max=20.0):
    """
    Scale wheel velocities to respect limits while preserving motion direction
    
    Args:
        wheel_velocities: Desired wheel velocities [ω1, ω2, ω3, ω4]
        omega_max: Maximum wheel velocity (rad/s)
    
    Returns:
        Limited wheel velocities
    """
    wheel_velocities = np.array(wheel_velocities)
    
    # Find maximum absolute velocity
    max_vel = np.max(np.abs(wheel_velocities))
    
    if max_vel > omega_max:
        # Scale all velocities proportionally
        scale = omega_max / max_vel
        wheel_velocities *= scale
    
    return wheel_velocities

# Add to class
MecanumRobot.apply_velocity_limits = apply_velocity_limits

# Example
robot = MecanumRobot()
wheels = robot.inverse_kinematics(vx=2.0, vy=2.0, omega=5.0)
print(f"Unlimited: {wheels}")

wheels_limited = robot.apply_velocity_limits(wheels, omega_max=15.0)
print(f"Limited: {wheels_limited}")
```

---

## Motion Control

### Velocity Controller

```python
def velocity_control(self, vx_desired, vy_desired, omega_desired, 
                     world_frame=False):
    """
    Control robot to achieve desired velocity
    
    Args:
        vx_desired: Desired x velocity (m/s)
        vy_desired: Desired y velocity (m/s)
        omega_desired: Desired angular velocity (rad/s)
        world_frame: If True, velocities are in world frame
    
    Returns:
        Wheel velocities [ω1, ω2, ω3, ω4]
    """
    if world_frame:
        # Transform to robot frame
        theta = self.state[2]
        vx_robot = vx_desired * np.cos(theta) + vy_desired * np.sin(theta)
        vy_robot = -vx_desired * np.sin(theta) + vy_desired * np.cos(theta)
    else:
        vx_robot = vx_desired
        vy_robot = vy_desired
    
    # Inverse kinematics
    wheel_velocities = self.inverse_kinematics(vx_robot, vy_robot, 
                                               omega_desired)
    
    # Apply limits
    wheel_velocities = self.apply_velocity_limits(wheel_velocities)
    
    return wheel_velocities

# Add to class
MecanumRobot.velocity_control = velocity_control
```

### Position Controller

```python
def navigate_to_point(self, target_x, target_y, target_theta=None,
                     kp_linear=1.0, kp_angular=2.0):
    """
    Navigate to target position
    
    Args:
        target_x, target_y: Target position (world frame)
        target_theta: Target orientation (None = don't care)
        kp_linear: Linear velocity gain
        kp_angular: Angular velocity gain
    
    Returns:
        Wheel velocities and distance to target
    """
    x, y, theta = self.state
    
    # Position error (world frame)
    dx_world = target_x - x
    dy_world = target_y - y
    distance = np.sqrt(dx_world**2 + dy_world**2)
    
    # Transform to robot frame
    dx_robot = dx_world * np.cos(theta) + dy_world * np.sin(theta)
    dy_robot = -dx_world * np.sin(theta) + dy_world * np.cos(theta)
    
    # Desired velocities (robot frame)
    vx_desired = kp_linear * dx_robot
    vy_desired = kp_linear * dy_robot
    
    # Orientation control
    if target_theta is not None:
        theta_error = np.arctan2(np.sin(target_theta - theta),
                                np.cos(target_theta - theta))
        omega_desired = kp_angular * theta_error
    else:
        omega_desired = 0
    
    # Velocity limits
    v_max = 1.0  # m/s
    omega_max_robot = 2.0  # rad/s
    
    vx_desired = np.clip(vx_desired, -v_max, v_max)
    vy_desired = np.clip(vy_desired, -v_max, v_max)
    omega_desired = np.clip(omega_desired, -omega_max_robot, omega_max_robot)
    
    # Inverse kinematics
    wheel_velocities = self.inverse_kinematics(vx_desired, vy_desired, 
                                               omega_desired)
    wheel_velocities = self.apply_velocity_limits(wheel_velocities)
    
    return wheel_velocities, distance

# Add to class
MecanumRobot.navigate_to_point = navigate_to_point

# Example: Navigate to point
robot = MecanumRobot()
target = (2.0, 1.5, np.pi/4)

dt = 0.05
trajectory = [robot.state.copy()]

for _ in range(500):
    wheels, dist = robot.navigate_to_point(*target)
    
    if dist < 0.05:
        break
    
    robot.forward_kinematics(wheels, dt)
    trajectory.append(robot.state.copy())

print(f"Final position: {robot.state}")
print(f"Distance to target: {dist:.4f} m")
```

---

## Motion Primitives

### Common Maneuvers

```python
class MecanumMotionPrimitives:
    @staticmethod
    def forward(distance, speed=0.5):
        """Move forward"""
        return {
            'vx': speed,
            'vy': 0,
            'omega': 0,
            'duration': distance / speed
        }
    
    @staticmethod
    def sideways(distance, speed=0.5):
        """Move sideways (positive = right)"""
        direction = 1 if distance > 0 else -1
        return {
            'vx': 0,
            'vy': direction * speed,
            'omega': 0,
            'duration': abs(distance) / speed
        }
    
    @staticmethod
    def diagonal(distance_x, distance_y, speed=0.5):
        """Move diagonally"""
        distance = np.sqrt(distance_x**2 + distance_y**2)
        duration = distance / speed
        
        vx = distance_x / duration
        vy = distance_y / duration
        
        return {
            'vx': vx,
            'vy': vy,
            'omega': 0,
            'duration': duration
        }
    
    @staticmethod
    def rotate(angle, angular_speed=1.0):
        """Rotate in place"""
        duration = abs(angle) / angular_speed
        omega = angular_speed if angle > 0 else -angular_speed
        
        return {
            'vx': 0,
            'vy': 0,
            'omega': omega,
            'duration': duration
        }
    
    @staticmethod
    def strafe_arc(radius, angle, speed=0.5):
        """Move in arc while maintaining orientation"""
        arc_length = radius * abs(angle)
        duration = arc_length / speed
        
        # Velocity perpendicular to radius
        vx = speed * np.cos(angle / 2)
        vy = speed * np.sin(angle / 2) * (1 if angle > 0 else -1)
        
        return {
            'vx': vx,
            'vy': vy,
            'omega': 0,
            'duration': duration
        }

# Example: Execute sequence
robot = MecanumRobot()

sequence = [
    MecanumMotionPrimitives.forward(1.0, 0.5),
    MecanumMotionPrimitives.sideways(0.5, 0.3),
    MecanumMotionPrimitives.diagonal(0.5, 0.5, 0.4),
    MecanumMotionPrimitives.rotate(np.pi/2, 1.0)
]

dt = 0.01
for primitive in sequence:
    t = 0
    while t < primitive['duration']:
        wheels = robot.inverse_kinematics(
            primitive['vx'], primitive['vy'], primitive['omega']
        )
        robot.forward_kinematics(wheels, dt)
        t += dt

print(f"Final pose: {robot.state}")
```

---

## Slippage Compensation

Mecanum wheels are prone to slippage, especially on smooth surfaces.

```python
class MecanumRobotWithSlippage(MecanumRobot):
    def __init__(self, lx=0.2, ly=0.2, wheel_radius=0.05,
                 slip_factor_forward=0.95,
                 slip_factor_sideways=0.85):
        """
        Args:
            slip_factor_forward: Efficiency in forward direction (0-1)
            slip_factor_sideways: Efficiency in sideways direction (0-1)
        """
        super().__init__(lx, ly, wheel_radius)
        self.slip_forward = slip_factor_forward
        self.slip_sideways = slip_factor_sideways
    
    def compensate_slippage(self, vx, vy, omega):
        """Compensate for expected slippage"""
        vx_compensated = vx / self.slip_forward
        vy_compensated = vy / self.slip_sideways
        
        return vx_compensated, vy_compensated, omega
    
    def inverse_kinematics_compensated(self, vx, vy, omega):
        """IK with slippage compensation"""
        vx_comp, vy_comp, omega_comp = self.compensate_slippage(vx, vy, omega)
        return self.inverse_kinematics(vx_comp, vy_comp, omega_comp)

# Example
robot = MecanumRobotWithSlippage(slip_factor_sideways=0.8)
wheels = robot.inverse_kinematics_compensated(vx=0.5, vy=0.5, omega=0)
print(f"Compensated wheel velocities: {wheels}")
```

---

## Visualization

```python
def simulate_and_plot(robot, control_sequence, dt=0.01):
    """Simulate and visualize mecanum robot motion"""
    import matplotlib.pyplot as plt
    
    trajectory = [robot.state.copy()]
    
    for cmd in control_sequence:
        t = 0
        while t < cmd['duration']:
            wheels = robot.inverse_kinematics(cmd['vx'], cmd['vy'], cmd['omega'])
            robot.forward_kinematics(wheels, dt)
            trajectory.append(robot.state.copy())
            t += dt
    
    trajectory = np.array(trajectory)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Path')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=12, label='End')
    
    # Robot orientation at intervals
    interval = len(trajectory) // 15
    for i in range(0, len(trajectory), interval):
        x, y, theta = trajectory[i]
        
        # Draw robot body
        size = 0.15
        corners = np.array([
            [size, size],
            [-size, size],
            [-size, -size],
            [size, -size],
            [size, size]
        ])
        
        # Rotate corners
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        corners_rot = (rot @ corners.T).T
        corners_rot[:, 0] += x
        corners_rot[:, 1] += y
        
        ax.plot(corners_rot[:, 0], corners_rot[:, 1], 'k-', linewidth=1)
        
        # Direction arrow
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05, 
                fc='red', ec='red')
    
    ax.grid(True)
    ax.axis('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.set_title('Mecanum Robot Trajectory')
    plt.show()

# Example: Complex maneuver
robot = MecanumRobot()

maneuver = [
    {'vx': 0.5, 'vy': 0, 'omega': 0, 'duration': 2.0},      # Forward
    {'vx': 0, 'vy': 0.5, 'omega': 0, 'duration': 2.0},      # Right
    {'vx': 0.5, 'vy': 0.5, 'omega': 0, 'duration': 2.0},    # Diagonal
    {'vx': 0, 'vy': 0, 'omega': 1.0, 'duration': np.pi},    # Rotate 180°
    {'vx': 0.5, 'vy': -0.5, 'omega': 0.5, 'duration': 3.0}  # Combined
]

simulate_and_plot(robot, maneuver)
```

---

## Further Reading

- [Mecanum Wheel Kinematics](https://research.ijcaonline.org/volume113/number3/pxc3901586.pdf)
- [Omnidirectional Mobile Robots](https://www.intechopen.com/chapters/39493)
- [Mecanum Wheel Control](https://www.hindawi.com/journals/jr/2015/347379/)

