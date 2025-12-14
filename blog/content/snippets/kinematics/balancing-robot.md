---
title: "Two-Wheel Balancing Robot Kinematics and Control"
date: 2024-12-13
draft: false
category: "kinematics"
tags: ["robotics", "balancing-robot", "inverted-pendulum", "control", "kinematics"]
---

Kinematics and control for two-wheel self-balancing robots (inverted pendulum on wheels).

## Overview

A **two-wheel balancing robot** is an underactuated system that must continuously balance while moving, similar to a Segway.

**Key Characteristics:**
- Inverted pendulum dynamics
- Inherently unstable (requires active control)
- 2 actuators (wheels), 3+ degrees of freedom
- Nonlinear dynamics

**Applications:**
- Personal transporters (Segway)
- Service robots
- Educational platforms
- Research in control theory

---

## System Model

```text
        ┌─────┐
        │ Body│  ← Center of mass (height h)
        │  •  │
        └──┬──┘
           │
      ─────┴─────  ← Wheel axis
      ●         ●  ← Wheels (radius r)
```

**State Variables:**
- $x$: Position (m)
- $\dot{x}$: Velocity (m/s)
- $\theta$: Body tilt angle from vertical (rad)
- $\dot{\theta}$: Angular velocity (rad/s)
- $\phi$: Wheel angle (rad)

**Parameters:**
- $m_w$: Mass of wheels
- $m_b$: Mass of body
- $I_w$: Moment of inertia of wheel
- $I_b$: Moment of inertia of body
- $r$: Wheel radius
- $L$: Distance from wheel axis to body center of mass
- $g$: Gravity (9.81 m/s²)

---

## Equations of Motion

### Linearized Model (Small Angles)

For $\theta \approx 0$, the system can be linearized:

$$
\ddot{\theta} = \frac{(m_b + m_w)gL\theta - m_bL\ddot{x}}{I_b + m_bL^2}
$$

$$
\ddot{x} = \frac{\tau - m_bL\ddot{\theta}}{m_b + m_w + I_w/r^2}
$$

Where $\tau$ is the torque applied by the wheels.

### State-Space Representation

$$
\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}
$$

Where:

$$
\mathbf{x} = \begin{bmatrix} x \\\\ \dot{x} \\\\ \theta \\\\ \dot{\theta} \end{bmatrix}, \quad
\mathbf{u} = \begin{bmatrix} \tau_L \\\\ \tau_R \end{bmatrix}
$$

---

## Python Implementation

### System Dynamics

```python
import numpy as np
from scipy.integrate import odeint

class BalancingRobot:
    def __init__(self, m_body=1.0, m_wheel=0.1, L=0.3, r=0.05,
                 I_body=0.05, I_wheel=0.001, wheelbase=0.2):
        """
        Args:
            m_body: Mass of body (kg)
            m_wheel: Mass of each wheel (kg)
            L: Distance to center of mass (m)
            r: Wheel radius (m)
            I_body: Body moment of inertia (kg⋅m²)
            I_wheel: Wheel moment of inertia (kg⋅m²)
            wheelbase: Distance between wheels (m)
        """
        self.m_b = m_body
        self.m_w = m_wheel
        self.L = L
        self.r = r
        self.I_b = I_body
        self.I_w = I_wheel
        self.wheelbase = wheelbase
        self.g = 9.81
        
        # State: [x, x_dot, theta, theta_dot, psi]
        # psi = heading angle
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    def dynamics(self, state, t, tau_left, tau_right):
        """
        Compute state derivatives
        
        Args:
            state: [x, x_dot, theta, theta_dot, psi]
            t: Time
            tau_left, tau_right: Wheel torques (N⋅m)
        
        Returns:
            State derivatives
        """
        x, x_dot, theta, theta_dot, psi = state
        
        # Average torque for forward motion
        tau_avg = (tau_left + tau_right) / 2
        
        # Differential torque for turning
        tau_diff = (tau_right - tau_left) / 2
        
        # Linearized dynamics (valid for small theta)
        # Total mass
        m_total = self.m_b + 2 * self.m_w
        
        # Denominator terms
        denom1 = self.I_b + self.m_b * self.L**2
        denom2 = m_total + 2 * self.I_w / self.r**2
        
        # Acceleration of body tilt
        theta_ddot = (
            (m_total * self.g * self.L * np.sin(theta) 
             - self.m_b * self.L * np.cos(theta) * 
             (tau_avg / self.r - self.m_b * self.L * theta_dot**2 * np.sin(theta)) / denom2)
            / (denom1 - self.m_b**2 * self.L**2 * np.cos(theta)**2 / denom2)
        )
        
        # Acceleration of position
        x_ddot = (
            tau_avg / self.r 
            - self.m_b * self.L * (theta_ddot * np.cos(theta) 
                                   - theta_dot**2 * np.sin(theta))
        ) / denom2
        
        # Heading rate (differential drive)
        psi_dot = tau_diff * self.r / (self.wheelbase * self.I_w)
        
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot, psi_dot])
    
    def step(self, tau_left, tau_right, dt):
        """
        Simulate one time step
        
        Args:
            tau_left, tau_right: Wheel torques
            dt: Time step
        
        Returns:
            Updated state
        """
        t_span = [0, dt]
        solution = odeint(self.dynamics, self.state, t_span, 
                         args=(tau_left, tau_right))
        self.state = solution[-1]
        
        return self.state.copy()

# Example: Free fall (no control)
robot = BalancingRobot()
robot.state[2] = 0.1  # Initial tilt

trajectory = [robot.state.copy()]
dt = 0.01

for _ in range(100):
    robot.step(tau_left=0, tau_right=0, dt=dt)
    trajectory.append(robot.state.copy())

trajectory = np.array(trajectory)

import matplotlib.pyplot as plt
time = np.arange(len(trajectory)) * dt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(time, trajectory[:, 0])
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)

plt.subplot(132)
plt.plot(time, np.degrees(trajectory[:, 2]))
plt.xlabel('Time (s)')
plt.ylabel('Tilt Angle (deg)')
plt.grid(True)

plt.subplot(133)
plt.plot(time, trajectory[:, 1])
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Balance Control

### PID Controller

```python
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        
        self.integral = 0
        self.prev_error = 0
    
    def update(self, measurement, dt):
        """Compute control output"""
        error = self.setpoint - measurement
        
        # Proportional
        P = self.Kp * error
        
        # Integral (with anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)
        I = self.Ki * self.integral
        
        # Derivative
        D = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        return P + I + D
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0
        self.prev_error = 0

class BalancingController:
    def __init__(self, robot):
        self.robot = robot
        
        # Balance controller (tilt angle)
        self.balance_pid = PIDController(
            Kp=50.0,   # Proportional gain
            Ki=5.0,    # Integral gain
            Kd=5.0,    # Derivative gain
            setpoint=0.0
        )
        
        # Position controller
        self.position_pid = PIDController(
            Kp=2.0,
            Ki=0.1,
            Kd=3.0,
            setpoint=0.0
        )
    
    def compute_control(self, target_position=0, target_heading=0):
        """
        Compute wheel torques for balance and position control
        
        Returns:
            (tau_left, tau_right)
        """
        x, x_dot, theta, theta_dot, psi = self.robot.state
        
        # Position error
        position_error = target_position - x
        
        # Desired tilt angle for position control
        theta_desired = self.position_pid.update(x, dt=0.01)
        theta_desired = np.clip(theta_desired, -0.3, 0.3)  # Limit tilt
        
        # Balance control
        self.balance_pid.setpoint = theta_desired
        tau_balance = self.balance_pid.update(theta, dt=0.01)
        
        # Heading control (simple proportional)
        heading_error = target_heading - psi
        heading_error = np.arctan2(np.sin(heading_error), 
                                   np.cos(heading_error))
        tau_heading = 0.5 * heading_error
        
        # Combine
        tau_left = tau_balance - tau_heading
        tau_right = tau_balance + tau_heading
        
        # Limit torques
        tau_max = 2.0  # N⋅m
        tau_left = np.clip(tau_left, -tau_max, tau_max)
        tau_right = np.clip(tau_right, -tau_max, tau_max)
        
        return tau_left, tau_right

# Example: Balance with position control
robot = BalancingRobot()
robot.state[2] = 0.1  # Initial tilt

controller = BalancingController(robot)

trajectory = [robot.state.copy()]
dt = 0.01

for i in range(500):
    # Target position moves
    target_pos = 0.5 * np.sin(i * dt * 0.5)
    
    tau_l, tau_r = controller.compute_control(target_position=target_pos)
    robot.step(tau_l, tau_r, dt)
    trajectory.append(robot.state.copy())

trajectory = np.array(trajectory)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
time = np.arange(len(trajectory)) * dt

axes[0, 0].plot(time, trajectory[:, 0], label='Actual')
axes[0, 0].plot(time, 0.5 * np.sin(time * 0.5), '--', label='Target')
axes[0, 0].set_ylabel('Position (m)')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(time, np.degrees(trajectory[:, 2]))
axes[0, 1].set_ylabel('Tilt Angle (deg)')
axes[0, 1].grid(True)

axes[1, 0].plot(time, trajectory[:, 1])
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Velocity (m/s)')
axes[1, 0].grid(True)

axes[1, 1].plot(time, trajectory[:, 3])
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

---

## LQR Controller

Linear Quadratic Regulator for optimal control.

```python
from scipy.linalg import solve_continuous_are

def compute_lqr_gain(robot):
    """
    Compute LQR gain matrix
    
    Returns:
        K: Feedback gain matrix
    """
    # Linearized system matrices
    m_total = robot.m_b + 2 * robot.m_w
    denom = robot.I_b + robot.m_b * robot.L**2
    
    # State matrix A
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -robot.m_b * robot.g * robot.L / m_total, 0],
        [0, 0, 0, 1],
        [0, 0, m_total * robot.g * robot.L / denom, 0]
    ])
    
    # Input matrix B
    B = np.array([
        [0],
        [1 / (m_total * robot.r)],
        [0],
        [-1 / (denom * robot.r)]
    ])
    
    # Cost matrices
    Q = np.diag([10.0, 1.0, 100.0, 10.0])  # State cost
    R = np.array([[0.1]])  # Control cost
    
    # Solve Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    
    # Compute gain
    K = np.linalg.inv(R) @ B.T @ P
    
    return K

class LQRBalancingController:
    def __init__(self, robot):
        self.robot = robot
        self.K = compute_lqr_gain(robot)
        print(f"LQR Gains: {self.K}")
    
    def compute_control(self, target_state=None):
        """
        Compute control using LQR
        
        Args:
            target_state: Desired [x, x_dot, theta, theta_dot]
        
        Returns:
            (tau_left, tau_right)
        """
        if target_state is None:
            target_state = np.array([0, 0, 0, 0])
        
        # State error
        current_state = self.robot.state[:4]
        error = current_state - target_state
        
        # LQR control law: u = -K * error
        tau = -self.K @ error
        tau = tau[0]  # Extract scalar
        
        # Apply to both wheels (no turning)
        tau_left = tau
        tau_right = tau
        
        # Limit
        tau_max = 2.0
        tau_left = np.clip(tau_left, -tau_max, tau_max)
        tau_right = np.clip(tau_right, -tau_max, tau_max)
        
        return tau_left, tau_right

# Example with LQR
robot = BalancingRobot()
robot.state[2] = 0.15  # Larger initial tilt

controller = LQRBalancingController(robot)

trajectory = [robot.state.copy()]
dt = 0.01

for _ in range(500):
    tau_l, tau_r = controller.compute_control()
    robot.step(tau_l, tau_r, dt)
    trajectory.append(robot.state.copy())

trajectory = np.array(trajectory)
print(f"Final tilt: {np.degrees(trajectory[-1, 2]):.2f} degrees")
```

---

## Forward and Inverse Kinematics

### Forward Kinematics

```python
def forward_kinematics(self, omega_left, omega_right):
    """
    Compute robot velocity from wheel velocities
    
    Args:
        omega_left, omega_right: Wheel angular velocities (rad/s)
    
    Returns:
        (v, omega): Linear and angular velocity
    """
    # Linear velocity
    v = self.r * (omega_left + omega_right) / 2
    
    # Angular velocity (heading rate)
    omega = self.r * (omega_right - omega_left) / self.wheelbase
    
    return v, omega

# Add to BalancingRobot class
BalancingRobot.forward_kinematics = forward_kinematics
```

### Inverse Kinematics

```python
def inverse_kinematics(self, v_desired, omega_desired):
    """
    Compute wheel velocities for desired motion
    
    Args:
        v_desired: Desired linear velocity (m/s)
        omega_desired: Desired angular velocity (rad/s)
    
    Returns:
        (omega_left, omega_right): Wheel angular velocities
    """
    omega_left = (v_desired - omega_desired * self.wheelbase / 2) / self.r
    omega_right = (v_desired + omega_desired * self.wheelbase / 2) / self.r
    
    return omega_left, omega_right

# Add to BalancingRobot class
BalancingRobot.inverse_kinematics = inverse_kinematics
```

---

## Practical Considerations

### Sensor Fusion (IMU)

```python
class IMUSensor:
    def __init__(self, noise_gyro=0.01, noise_accel=0.1):
        self.noise_gyro = noise_gyro
        self.noise_accel = noise_accel
    
    def read(self, true_theta, true_theta_dot):
        """Simulate IMU readings with noise"""
        # Gyroscope (angular velocity)
        gyro = true_theta_dot + np.random.normal(0, self.noise_gyro)
        
        # Accelerometer (tilt angle estimate)
        accel_theta = true_theta + np.random.normal(0, self.noise_accel)
        
        return gyro, accel_theta

class ComplementaryFilter:
    def __init__(self, alpha=0.98):
        """
        Args:
            alpha: Filter coefficient (0.95-0.99 typical)
        """
        self.alpha = alpha
        self.theta_fused = 0
    
    def update(self, gyro, accel_theta, dt):
        """
        Fuse gyro and accelerometer data
        
        Args:
            gyro: Angular velocity from gyroscope
            accel_theta: Angle from accelerometer
            dt: Time step
        """
        # Integrate gyro
        theta_gyro = self.theta_fused + gyro * dt
        
        # Complementary filter
        self.theta_fused = self.alpha * theta_gyro + (1 - self.alpha) * accel_theta
        
        return self.theta_fused

# Example
imu = IMUSensor()
filter = ComplementaryFilter(alpha=0.98)

true_theta = 0.1
true_theta_dot = 0.5

for _ in range(100):
    gyro, accel = imu.read(true_theta, true_theta_dot)
    theta_estimate = filter.update(gyro, accel, dt=0.01)
    
    # Update true values (simple dynamics)
    true_theta += true_theta_dot * 0.01

print(f"True: {true_theta:.3f}, Estimated: {theta_estimate:.3f}")
```

### Motor Saturation

```python
def apply_motor_limits(self, tau_left, tau_right, tau_max=2.0, 
                      omega_max=50.0):
    """
    Apply motor torque and velocity limits
    
    Returns:
        Limited torques
    """
    # Torque limits
    tau_left = np.clip(tau_left, -tau_max, tau_max)
    tau_right = np.clip(tau_right, -tau_max, tau_max)
    
    # Velocity limits (if needed)
    # This would require tracking wheel velocities
    
    return tau_left, tau_right
```

---

## Complete Simulation

```python
def simulate_balancing_robot(duration=10.0, dt=0.01, 
                            controller_type='lqr'):
    """Complete simulation with visualization"""
    robot = BalancingRobot()
    robot.state[2] = 0.1  # Initial tilt
    
    if controller_type == 'lqr':
        controller = LQRBalancingController(robot)
    else:
        controller = BalancingController(robot)
    
    trajectory = []
    controls = []
    
    t = 0
    while t < duration:
        # Compute control
        if controller_type == 'lqr':
            tau_l, tau_r = controller.compute_control()
        else:
            target_pos = 0.5 * np.sin(t * 0.5)
            tau_l, tau_r = controller.compute_control(target_position=target_pos)
        
        # Step simulation
        robot.step(tau_l, tau_r, dt)
        
        trajectory.append(robot.state.copy())
        controls.append([tau_l, tau_r])
        
        t += dt
    
    trajectory = np.array(trajectory)
    controls = np.array(controls)
    time = np.arange(len(trajectory)) * dt
    
    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    axes[0, 0].plot(time, trajectory[:, 0])
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Position')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(time, np.degrees(trajectory[:, 2]))
    axes[0, 1].set_ylabel('Tilt (deg)')
    axes[0, 1].set_title('Tilt Angle')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(time, trajectory[:, 1])
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].set_title('Linear Velocity')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(time, trajectory[:, 3])
    axes[1, 1].set_ylabel('Angular Vel (rad/s)')
    axes[1, 1].set_title('Tilt Rate')
    axes[1, 1].grid(True)
    
    axes[2, 0].plot(time, controls[:, 0], label='Left')
    axes[2, 0].plot(time, controls[:, 1], label='Right')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Torque (N⋅m)')
    axes[2, 0].set_title('Control Torques')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(trajectory[:, 0], np.degrees(trajectory[:, 2]))
    axes[2, 1].set_xlabel('Position (m)')
    axes[2, 1].set_ylabel('Tilt (deg)')
    axes[2, 1].set_title('Phase Plot')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return trajectory, controls

# Run simulation
trajectory, controls = simulate_balancing_robot(duration=10.0, 
                                                controller_type='lqr')
```

---

## Further Reading

- [Inverted Pendulum Control](https://www.cds.caltech.edu/~murray/courses/cds101/fa04/caltech/am04_ch5-3nov04.pdf)
- [Segway Dynamics](https://web.mit.edu/first/segway/papers/Segway_modeling.pdf)
- [LQR Tutorial](https://www.cds.caltech.edu/~murray/courses/cds110/wi06/lqr.pdf)
- [Self-Balancing Robot Design](https://ieeexplore.ieee.org/document/8424051)

