---
title: "Jacobian Matrices for Forward and Inverse Kinematics"
date: 2024-12-13
draft: false
category: "kinematics"
tags: ["robotics", "kinematics", "jacobian", "inverse-kinematics", "velocity"]
---

Jacobian matrices relate joint velocities to end-effector velocities, essential for inverse kinematics and control.

## Overview

The **Jacobian matrix** $J$ maps joint velocities $\dot{q}$ to end-effector velocity $\dot{x}$:

$$
\dot{x} = J(q) \dot{q}
$$

Where:
- $\dot{x} = [\dot{x}, \dot{y}, \dot{z}, \omega_x, \omega_y, \omega_z]^T$ (linear and angular velocity)
- $\dot{q} = [\dot{q}_1, \dot{q}_2, \ldots, \dot{q}_n]^T$ (joint velocities)
- $J(q)$ is a $6 \times n$ matrix (for spatial manipulators)

---

## Geometric Jacobian

### Column-wise Construction

For joint $i$, the $i$-th column of the Jacobian is:

**Revolute Joint:**

$$
J_i = \begin{bmatrix} z_{i-1} \times (o_n - o_{i-1}) \\\\ z_{i-1} \end{bmatrix}
$$

**Prismatic Joint:**

$$
J_i = \begin{bmatrix} z_{i-1} \\\\ 0 \end{bmatrix}
$$

Where:
- $z_{i-1}$: Joint axis direction
- $o_n$: End-effector position
- $o_{i-1}$: Joint $i$ position

### Python Implementation

```python
import numpy as np

def compute_jacobian(transforms, joint_types):
    """
    Compute geometric Jacobian
    
    Args:
        transforms: List of 4x4 transformation matrices to each joint
        joint_types: List of 'revolute' or 'prismatic'
    
    Returns:
        6 x n Jacobian matrix
    """
    n = len(joint_types)
    J = np.zeros((6, n))
    
    # End-effector position
    o_n = transforms[-1][:3, 3]
    
    for i in range(n):
        # Joint i position and z-axis
        o_i = transforms[i][:3, 3]
        z_i = transforms[i][:3, 2]
        
        if joint_types[i] == 'revolute':
            # Linear velocity component
            J[:3, i] = np.cross(z_i, o_n - o_i)
            # Angular velocity component
            J[3:, i] = z_i
        else:  # prismatic
            # Linear velocity component
            J[:3, i] = z_i
            # Angular velocity component (zero)
            J[3:, i] = 0
    
    return J

# Example: 2-link planar arm
def planar_arm_jacobian(theta1, theta2, L1=1.0, L2=0.8):
    """Jacobian for 2-link planar arm"""
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c12 = np.cos(theta1 + theta2)
    s12 = np.sin(theta1 + theta2)
    
    J = np.array([
        [-L1*s1 - L2*s12, -L2*s12],
        [ L1*c1 + L2*c12,  L2*c12],
        [ 1,               1      ]
    ])
    
    return J

# Test
J = planar_arm_jacobian(np.pi/4, np.pi/3)
print("Jacobian:")
print(J)
```

---

## Analytical Jacobian

Maps joint velocities to end-effector position and orientation rates (e.g., Euler angles).

### Relationship to Geometric Jacobian

$$
J_a = T(x) J_g
$$

Where $T(x)$ transforms angular velocity to orientation rate.

### For ZYX Euler Angles

```python
def euler_transformation_matrix(roll, pitch, yaw):
    """
    Transformation from angular velocity to Euler angle rates
    
    ω = T(φ, θ, ψ) * [φ̇, θ̇, ψ̇]ᵀ
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    
    T = np.array([
        [cp, 0, 1],
        [sp*sr, cr, 0],
        [sp*cr, -sr, 0]
    ])
    
    return T

def analytical_jacobian(geometric_jacobian, euler_angles):
    """Convert geometric to analytical Jacobian"""
    roll, pitch, yaw = euler_angles
    
    T = euler_transformation_matrix(roll, pitch, yaw)
    T_inv = np.linalg.inv(T)
    
    # Build transformation matrix
    T_full = np.eye(6)
    T_full[3:, 3:] = T_inv
    
    J_analytical = T_full @ geometric_jacobian
    
    return J_analytical
```

---

## Inverse Kinematics with Jacobian

### Jacobian Inverse Method

For square Jacobian ($n = 6$):

$$
\dot{q} = J^{-1}(q) \dot{x}
$$

```python
def inverse_kinematics_step(current_joints, target_pose, current_pose, 
                            robot, dt=0.01, gain=1.0):
    """
    Single IK iteration using Jacobian inverse
    
    Args:
        current_joints: Current joint angles
        target_pose: Desired end-effector pose (position + orientation)
        current_pose: Current end-effector pose
        robot: Robot object with compute_jacobian method
        dt: Time step
        gain: Convergence gain
    
    Returns:
        Updated joint angles
    """
    # Compute pose error
    position_error = target_pose[:3] - current_pose[:3]
    orientation_error = target_pose[3:] - current_pose[3:]
    
    error = np.concatenate([position_error, orientation_error])
    
    # Desired end-effector velocity
    v_desired = gain * error
    
    # Compute Jacobian
    J = robot.compute_jacobian(current_joints)
    
    # Compute joint velocities
    try:
        q_dot = np.linalg.solve(J, v_desired)
    except np.linalg.LinAlgError:
        print("Warning: Singular Jacobian, using pseudoinverse")
        q_dot = np.linalg.pinv(J) @ v_desired
    
    # Update joints
    new_joints = current_joints + q_dot * dt
    
    return new_joints
```

### Jacobian Pseudoinverse (Redundant/Underconstrained)

For non-square Jacobian:

$$
\dot{q} = J^+ \dot{x}
$$

Where $J^+ = J^T(JJ^T)^{-1}$ is the Moore-Penrose pseudoinverse.

```python
def inverse_kinematics_pseudoinverse(current_joints, target_pose, 
                                     current_pose, robot, 
                                     max_iterations=100, tolerance=1e-3):
    """
    IK using Jacobian pseudoinverse
    
    Works for redundant and underconstrained manipulators
    """
    joints = current_joints.copy()
    
    for iteration in range(max_iterations):
        # Current pose
        T = robot.forward_kinematics(joints)
        current_pos = T[:3, 3]
        
        # Position error
        error = target_pose[:3] - current_pos
        error_norm = np.linalg.norm(error)
        
        if error_norm < tolerance:
            print(f"Converged in {iteration} iterations")
            return joints, True
        
        # Compute Jacobian (position only)
        J = robot.compute_jacobian(joints)[:3, :]
        
        # Pseudoinverse
        J_pinv = np.linalg.pinv(J)
        
        # Update
        delta_q = J_pinv @ error
        joints = joints + 0.1 * delta_q  # Step size 0.1
    
    print(f"Did not converge after {max_iterations} iterations")
    return joints, False
```

---

## Damped Least Squares (Levenberg-Marquardt)

Handles singularities better than pure pseudoinverse.

$$
\dot{q} = J^T(JJ^T + \lambda^2 I)^{-1} \dot{x}
$$

```python
def damped_least_squares_ik(current_joints, target_pose, robot,
                            max_iterations=100, tolerance=1e-3,
                            lambda_damping=0.01):
    """
    IK using damped least squares
    
    Args:
        lambda_damping: Damping factor (larger = more stable, slower convergence)
    """
    joints = current_joints.copy()
    
    for iteration in range(max_iterations):
        # Current pose
        T = robot.forward_kinematics(joints)
        current_pos = T[:3, 3]
        
        # Error
        error = target_pose[:3] - current_pos
        error_norm = np.linalg.norm(error)
        
        if error_norm < tolerance:
            return joints, True
        
        # Jacobian
        J = robot.compute_jacobian(joints)[:3, :]
        
        # Damped least squares
        JJT = J @ J.T
        damping_matrix = lambda_damping**2 * np.eye(JJT.shape[0])
        
        delta_q = J.T @ np.linalg.solve(JJT + damping_matrix, error)
        
        # Update with adaptive step size
        alpha = 0.5  # Step size
        joints = joints + alpha * delta_q
    
    return joints, False
```

---

## Singularity Analysis

### Detecting Singularities

```python
def analyze_singularities(robot, joints):
    """Analyze manipulator singularities"""
    J = robot.compute_jacobian(joints)
    
    # Singular value decomposition
    U, s, Vt = np.linalg.svd(J)
    
    # Condition number
    condition_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
    
    # Manipulability measure (Yoshikawa)
    manipulability = np.sqrt(np.linalg.det(J @ J.T))
    
    # Check singularity
    is_singular = condition_number > 1000 or manipulability < 0.01
    
    return {
        'singular_values': s,
        'condition_number': condition_number,
        'manipulability': manipulability,
        'is_singular': is_singular,
        'rank': np.sum(s > 1e-10)
    }

# Example
analysis = analyze_singularities(robot, joints)
print(f"Condition number: {analysis['condition_number']:.2f}")
print(f"Manipulability: {analysis['manipulability']:.4f}")
print(f"Singular: {analysis['is_singular']}")
```

### Manipulability Ellipsoid

```python
def plot_manipulability_ellipsoid(robot, joints):
    """Visualize manipulability ellipsoid"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    J = robot.compute_jacobian(joints)[:3, :]  # Position part
    
    # SVD
    U, s, Vt = np.linalg.svd(J)
    
    # Generate ellipsoid
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Scale by singular values
    for i in range(len(x)):
        for j in range(len(x)):
            point = np.array([x[i,j], y[i,j], z[i,j]])
            scaled = U @ np.diag(s) @ point
            x[i,j], y[i,j], z[i,j] = scaled
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, alpha=0.6, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Manipulability Ellipsoid')
    plt.show()
```

---

## Null Space Motion (Redundant Manipulators)

For redundant manipulators ($n > 6$), use null space for secondary objectives.

$$
\dot{q} = J^+ \dot{x} + (I - J^+ J) \dot{q}_0
$$

Where $(I - J^+ J)$ projects $\dot{q}_0$ onto the null space.

```python
def ik_with_null_space(current_joints, target_velocity, 
                       secondary_objective, robot):
    """
    IK with null space optimization
    
    Args:
        current_joints: Current joint configuration
        target_velocity: Desired end-effector velocity
        secondary_objective: Secondary task joint velocities
        robot: Robot object
    
    Returns:
        Joint velocities achieving primary task + secondary objective
    """
    J = robot.compute_jacobian(current_joints)
    J_pinv = np.linalg.pinv(J)
    
    # Primary task
    q_dot_primary = J_pinv @ target_velocity
    
    # Null space projector
    n = len(current_joints)
    I = np.eye(n)
    null_space_projector = I - J_pinv @ J
    
    # Secondary task (projected onto null space)
    q_dot_secondary = null_space_projector @ secondary_objective
    
    # Combined motion
    q_dot = q_dot_primary + q_dot_secondary
    
    return q_dot

# Example: Joint limit avoidance as secondary task
def joint_limit_avoidance(joints, joint_limits, gain=0.1):
    """Generate joint velocities to avoid limits"""
    q_dot_secondary = np.zeros_like(joints)
    
    for i, (q, limits) in enumerate(zip(joints, joint_limits)):
        q_mid = (limits['max'] + limits['min']) / 2
        q_range = limits['max'] - limits['min']
        
        # Gradient pushes toward middle
        q_dot_secondary[i] = -gain * (q - q_mid) / q_range
    
    return q_dot_secondary
```

---

## Velocity IK Example

Complete example with 3-DOF RRR manipulator.

```python
class RRRManipulatorWithJacobian:
    def __init__(self, L1=0.5, L2=0.4, L3=0.3):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
    
    def forward_kinematics(self, theta1, theta2, theta3):
        """Compute end-effector position"""
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        c12 = np.cos(theta1 + theta2)
        s12 = np.sin(theta1 + theta2)
        c123 = np.cos(theta1 + theta2 + theta3)
        s123 = np.sin(theta1 + theta2 + theta3)
        
        x = c1 * (self.L2*np.cos(theta2) + self.L3*np.cos(theta2 + theta3))
        y = s1 * (self.L2*np.cos(theta2) + self.L3*np.cos(theta2 + theta3))
        z = self.L1 + self.L2*np.sin(theta2) + self.L3*np.sin(theta2 + theta3)
        
        return np.array([x, y, z])
    
    def compute_jacobian(self, theta1, theta2, theta3):
        """Compute 3x3 Jacobian (position only)"""
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        s2 = np.sin(theta2)
        c2 = np.cos(theta2)
        s23 = np.sin(theta2 + theta3)
        c23 = np.cos(theta2 + theta3)
        
        L2c2_L3c23 = self.L2*c2 + self.L3*c23
        
        J = np.array([
            [-s1*L2c2_L3c23, -c1*(self.L2*s2 + self.L3*s23), -c1*self.L3*s23],
            [ c1*L2c2_L3c23, -s1*(self.L2*s2 + self.L3*s23), -s1*self.L3*s23],
            [ 0,              self.L2*c2 + self.L3*c23,       self.L3*c23    ]
        ])
        
        return J
    
    def inverse_kinematics(self, target_pos, initial_guess=None,
                          max_iter=100, tol=1e-3):
        """Numerical IK using Jacobian"""
        if initial_guess is None:
            joints = np.array([0.0, np.pi/4, 0.0])
        else:
            joints = initial_guess.copy()
        
        for i in range(max_iter):
            current_pos = self.forward_kinematics(*joints)
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < tol:
                return joints, True
            
            J = self.compute_jacobian(*joints)
            
            # Damped least squares
            lambda_damping = 0.01
            JJT = J @ J.T
            delta_q = J.T @ np.linalg.solve(
                JJT + lambda_damping**2 * np.eye(3), error
            )
            
            joints = joints + 0.5 * delta_q
        
        return joints, False

# Example usage
robot = RRRManipulatorWithJacobian()

# Forward kinematics
joints = np.array([np.pi/6, np.pi/4, np.pi/6])
pos = robot.forward_kinematics(*joints)
print(f"Forward kinematics: {pos}")

# Jacobian
J = robot.compute_jacobian(*joints)
print(f"\nJacobian:\n{J}")

# Inverse kinematics
target = np.array([0.3, 0.2, 0.8])
solution, success = robot.inverse_kinematics(target)
if success:
    print(f"\nIK solution: {np.degrees(solution)} degrees")
    print(f"Verification: {robot.forward_kinematics(*solution)}")
```

---

## Numerical Jacobian (Finite Differences)

When analytical Jacobian is difficult to derive.

```python
def numerical_jacobian(robot, joints, epsilon=1e-6):
    """
    Compute Jacobian using finite differences
    
    Args:
        robot: Robot with forward_kinematics method
        joints: Current joint configuration
        epsilon: Finite difference step size
    """
    n = len(joints)
    
    # Current position
    T0 = robot.forward_kinematics(*joints)
    pos0 = T0[:3, 3]
    
    J = np.zeros((3, n))
    
    for i in range(n):
        # Perturb joint i
        joints_plus = joints.copy()
        joints_plus[i] += epsilon
        
        # Forward kinematics
        T_plus = robot.forward_kinematics(*joints_plus)
        pos_plus = T_plus[:3, 3]
        
        # Finite difference
        J[:, i] = (pos_plus - pos0) / epsilon
    
    return J

# Verify against analytical Jacobian
J_analytical = robot.compute_jacobian(*joints)
J_numerical = numerical_jacobian(robot, joints)

print("Analytical Jacobian:")
print(J_analytical)
print("\nNumerical Jacobian:")
print(J_numerical)
print("\nDifference:")
print(np.abs(J_analytical - J_numerical))
```

---

## Further Reading

- [Siciliano, B. "Robotics: Modelling, Planning and Control"](https://www.springer.com/gp/book/9781846286414)
- [Lynch, K. "Modern Robotics"](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)
- [Jacobian Tutorial](https://www.cs.cmu.edu/~16311/current/labs/lab5/jacobian.pdf)
- [Singularity Analysis](https://www.sciencedirect.com/science/article/pii/S0094114X05001345)

