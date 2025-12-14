---
title: "DNN Policy Learning Theory"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "reinforcement-learning", "policy-gradient", "dnn", "mathematics"]
---


Deep Neural Network policy learning with mathematical foundations.

---

## Policy Gradient Methods

### Policy Parameterization

Policy $\pi_\theta(a|s)$ parameterized by neural network with weights $\theta$.

### Objective Function

Maximize expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

Where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is a trajectory.

---

## Policy Gradient Theorem

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
$$

Where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the return from time $t$.

### Derivation Sketch

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{\tau}[R(\tau)] \\
&= \nabla_\theta \int P(\tau|\theta) R(\tau) d\tau \\
&= \int \nabla_\theta P(\tau|\theta) R(\tau) d\tau \\
&= \int P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) d\tau \\
&= \mathbb{E}_{\tau}\left[\nabla_\theta \log P(\tau|\theta) R(\tau)\right]
\end{aligned}
$$

---

## REINFORCE Algorithm

### Update Rule

$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t
$$

### With Baseline

To reduce variance:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))
$$

Common baseline: $b(s_t) = V(s_t)$ (state value function)

---

## Actor-Critic Methods

### Architecture

- **Actor**: Policy network $\pi_\theta(a|s)$
- **Critic**: Value network $V_\phi(s)$

### Advantage Function

$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
$$

Approximation:

$$
A(s_t, a_t) \approx r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

### Update Rules

**Actor update:**

$$
\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t)
$$

**Critic update:**

$$
\phi \leftarrow \phi - \alpha_\phi \nabla_\phi \left(r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\right)^2
$$

---

## A3C (Asynchronous Advantage Actor-Critic)

Multiple agents explore in parallel, updating shared parameters.

### Advantage Estimation

n-step return:

$$
A(s_t, a_t) = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n}) - V(s_t)
$$

---

## PPO (Proximal Policy Optimization)

### Clipped Objective

$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

Where:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

This prevents too large policy updates.

---

## TRPO (Trust Region Policy Optimization)

### Constrained Optimization

$$
\begin{aligned}
\max_\theta \quad & \mathbb{E}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t\right] \\
\text{s.t.} \quad & \mathbb{E}_t[D_{KL}(\pi_{\theta_{old}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t))] \leq \delta
\end{aligned}
$$

Where $D_{KL}$ is the KL divergence.

---

## Deterministic Policy Gradient (DPG)

For continuous action spaces:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\pi}\left[\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)}\right]
$$

Where $\mu_\theta(s)$ is a deterministic policy.

---

## DDPG (Deep Deterministic Policy Gradient)

Combines DPG with DQN techniques:

**Actor update:**

$$
\nabla_\theta J \approx \mathbb{E}_{s_t}\left[\nabla_a Q(s,a|\phi)\big|_{s=s_t, a=\mu(s_t|\theta)} \nabla_\theta \mu(s|\theta)\big|_{s=s_t}\right]
$$

**Critic update:**

$$
L = \mathbb{E}\left[(r + \gamma Q'(s', \mu'(s'|\theta')|\phi') - Q(s,a|\phi))^2\right]
$$

Uses target networks (denoted with $'$) and experience replay.

---

## Python Implementation (Simple Actor-Critic)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.net(state)

# Training
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Get action probabilities
        probs = actor(torch.FloatTensor(state))
        action = torch.multinomial(probs, 1).item()
        
        # Take action
        next_state, reward, done = env.step(action)
        
        # Compute advantage
        value = critic(torch.FloatTensor(state))
        next_value = critic(torch.FloatTensor(next_state))
        advantage = reward + gamma * next_value * (1 - done) - value
        
        # Update critic
        critic_loss = advantage.pow(2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # Update actor
        log_prob = torch.log(probs[action])
        actor_loss = -log_prob * advantage.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        state = next_state
```

---