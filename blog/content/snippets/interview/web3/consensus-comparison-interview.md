---
title: "Consensus Algorithms Comparison Interview Questions"
date: 2025-12-14
tags: ["consensus", "blockchain", "interview", "comparison", "security"]
---

Consensus algorithm comparison and general implementation interview questions.

## Q1: Compare different consensus algorithms.

**Answer**:

**Comparison Table**:

| Algorithm | Type | Finality | Block Time | Energy | Fault Tolerance |
|-----------|------|----------|------------|--------|-----------------|
| **Paxos** | Classic | Immediate | N/A | Low | (n-1)/2 failures |
| **BFT** | Byzantine | Immediate | Fast | Low | f Byzantine (n=3f+1) |
| **Tendermint** | BFT | Immediate | 1-6s | Low | 1/3 Byzantine |
| **Ouroboros** | PoS | Probabilistic | 1s | Very Low | 51% stake attack |
| **Bitcoin** | PoW | Probabilistic | 10min | Very High | 51% hash power |
| **Ethereum PoS** | PoS | Checkpoint | 12s | Very Low | 1/3 stake attack |
| **Polkadot** | NPoS | Fast | 6s | Very Low | 1/3 stake attack |
| **Solana** | PoH + PoS | Probabilistic | 0.4s | Very Low | 1/3 stake attack |

**Trade-offs**:

**Safety vs Liveness**:
- **Paxos/BFT/Tendermint**: Strong safety, requires synchrony
- **PoW/PoS**: Probabilistic safety, works asynchronously

**Finality**:
- **Immediate**: Paxos, BFT, Tendermint
- **Probabilistic**: Bitcoin, Ouroboros, Solana
- **Checkpoint**: Ethereum PoS
- **Fast**: Polkadot (GRANDPA)

**Energy Efficiency**:
- **Low**: Paxos, BFT, Tendermint
- **Very Low**: All PoS algorithms
- **Very High**: PoW (Bitcoin)

**Use Case Selection**:
- **Distributed Systems**: Paxos, BFT
- **Public Blockchains**: PoW, PoS variants
- **High Throughput**: Tendermint, Polkadot, Solana
- **Ultra-High Throughput**: Solana (65K+ TPS)
- **Formal Verification**: Ouroboros
- **Shared Security**: Polkadot

---

## Q2: How do you implement a simple consensus algorithm?

**Answer**:

**Simple BFT Implementation**:

```python
class SimpleBFT:
    def __init__(self, node_id, total_nodes):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = (total_nodes - 1) // 3  # Max Byzantine nodes
        self.quorum = 2 * self.f + 1
        self.is_primary = (node_id == 0)
        self.log = {}
        self.sequence = 0
    
    def propose(self, request):
        if not self.is_primary:
            return None
        
        self.sequence += 1
        proposal = {
            'sequence': self.sequence,
            'request': request,
            'prepares': {self.node_id},
            'commits': set()
        }
        self.log[self.sequence] = proposal
        
        # Broadcast pre-prepare
        return {
            'type': 'pre_prepare',
            'sequence': self.sequence,
            'request': request
        }
    
    def handle_pre_prepare(self, message):
        if self.is_primary:
            return None
        
        sequence = message['sequence']
        request = message['request']
        
        # Validate
        if not self.validate_request(request):
            return None
        
        # Store and broadcast prepare
        self.log[sequence] = {
            'request': request,
            'prepares': {self.node_id},
            'commits': set()
        }
        
        return {
            'type': 'prepare',
            'sequence': sequence,
            'node_id': self.node_id
        }
    
    def handle_prepare(self, message):
        sequence = message['sequence']
        node_id = message['node_id']
        
        if sequence not in self.log:
            return None
        
        self.log[sequence]['prepares'].add(node_id)
        
        # Check if we have quorum
        if len(self.log[sequence]['prepares']) >= self.quorum:
            # Broadcast commit
            return {
                'type': 'commit',
                'sequence': sequence,
                'node_id': self.node_id
            }
        
        return None
    
    def handle_commit(self, message):
        sequence = message['sequence']
        node_id = message['node_id']
        
        if sequence not in self.log:
            return None
        
        self.log[sequence]['commits'].add(node_id)
        
        # Check if we have quorum
        if len(self.log[sequence]['commits']) >= self.quorum:
            # Execute request
            result = self.execute(self.log[sequence]['request'])
            return {
                'type': 'reply',
                'sequence': sequence,
                'result': result
            }
        
        return None
```

---

## Q3: What are the security properties of consensus algorithms?

**Answer**:

**Safety Properties**:

1. **Agreement**: All honest nodes agree on same value
2. **Validity**: Only valid values are chosen
3. **Integrity**: Values cannot be tampered with

**Liveness Properties**:

1. **Termination**: Algorithm eventually terminates
2. **Progress**: System continues to make progress
3. **Responsiveness**: Responds within bounded time

**Fault Tolerance**:

- **Crash Faults**: Nodes stop responding
- **Byzantine Faults**: Nodes behave arbitrarily
- **Network Faults**: Messages delayed or lost

**Attack Vectors**:

1. **51% Attack**: Control majority of resources
2. **Sybil Attack**: Create many fake identities
3. **Eclipse Attack**: Isolate node from network
4. **Nothing-at-Stake**: Vote on multiple chains (PoS)
5. **Long-Range Attack**: Rewrite history (PoS)

**Mitigation Strategies**:

- **Checkpointing**: Lock in finalized blocks
- **Slashing**: Penalize malicious behavior
- **Time Locks**: Delay withdrawals
- **Validator Rotation**: Change validator set
- **Finality Gadgets**: Separate finality mechanism

---

