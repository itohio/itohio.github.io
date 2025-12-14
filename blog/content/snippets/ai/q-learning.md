---
title: "Q-Learning Theory"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "reinforcement-learning", "q-learning", "mathematics"]
---


Q-Learning algorithm theory with mathematical foundations.

---

## Markov Decision Process (MDP)

An MDP is defined by the tuple $(S, A, P, R, \gamma)$:

- $S$: Set of states
- $A$: Set of actions
- $P$: Transition probability $P(s'|s,a)$
- $R$: Reward function $R(s,a,s')$
- $\gamma \in [0,1]$: Discount factor

---

## Value Functions

### State Value Function

Expected return starting from state $s$:

$$
V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s\right]
$$

### Action Value Function (Q-Function)

Expected return starting from state $s$, taking action $a$:

$$
Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s, A_0 = a\right]
$$

---

## Bellman Equations

### Bellman Expectation Equation

$$
V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]
$$

$$
Q^\pi(s,a) = \sum_{s' \in S} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]
$$

### Bellman Optimality Equation

$$
V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]
$$

$$
Q^*(s,a) = \sum_{s' \in S} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]
$$

---

## Q-Learning Algorithm

### Update Rule

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]
$$

Where:
- $\alpha \in (0,1]$: Learning rate
- $\gamma \in [0,1]$: Discount factor
- $r_{t+1}$: Immediate reward
- $s_t, a_t$: Current state and action
- $s_{t+1}$: Next state

### TD Error

The temporal difference error:

$$
\delta_t = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)
$$

---

## Exploration vs Exploitation

### ε-Greedy Policy

$$
\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s,a') \\
\frac{\epsilon}{|A|} & \text{otherwise}
\end{cases}
$$

### Boltzmann Exploration (Softmax)

$$
\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}
$$

Where $\tau$ is the temperature parameter.

---

## Convergence Conditions

Q-Learning converges to $Q^*$ if:

1. All state-action pairs are visited infinitely often
2. Learning rate satisfies:
   $$
   \sum_{t=0}^{\infty} \alpha_t = \infty, \quad \sum_{t=0}^{\infty} \alpha_t^2 < \infty
   $$
3. Example: $\alpha_t = \frac{1}{1+t}$

---

## Python Implementation

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        # Q-Learning update
        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error

# Training loop
agent = QLearning(n_states=100, n_actions=4)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
```

---

## Double Q-Learning

To address overestimation bias:

$$
Q_1(s_t, a_t) \leftarrow Q_1(s_t, a_t) + \alpha \left[r_{t+1} + \gamma Q_2(s_{t+1}, \arg\max_{a'} Q_1(s_{t+1}, a')) - Q_1(s_t, a_t)\right]
$$

Randomly switch between updating $Q_1$ and $Q_2$.

---

## SARSA (On-Policy Alternative)

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\right]
$$

Key difference: Uses actual next action $a_{t+1}$ instead of $\max_{a'} Q(s_{t+1}, a')$.

---