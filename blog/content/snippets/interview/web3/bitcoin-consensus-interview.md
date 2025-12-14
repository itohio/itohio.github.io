---
title: "Bitcoin (Nakamoto) Consensus Interview Questions"
date: 2025-12-14
tags: ["consensus", "blockchain", "interview", "bitcoin", "pow", "nakamoto"]
---

Bitcoin consensus algorithm interview questions covering Proof-of-Work (PoW) and Nakamoto consensus.

## Q1: How does Bitcoin (Nakamoto) consensus work?

**Answer**:

**Bitcoin** uses Proof-of-Work (PoW) consensus, also known as Nakamoto consensus.

**Sequence Diagram**:
```mermaid
sequenceDiagram
    participant Miner1
    participant Miner2
    participant Miner3
    participant Node1
    participant Node2
    participant Node3
    
    Note over Miner1,Miner3: Mining Competition
    
    Miner1->>Miner1: Select Transactions
    Miner1->>Miner1: Create Block Header
    Miner1->>Miner1: Hash Block (nonce=0)
    Miner1->>Miner1: Hash Block (nonce=1)
    Miner1->>Miner1: Hash Block (nonce=2)
    Note over Miner1: ... millions of hashes ...
    Miner1->>Miner1: Hash Block (nonce=N) ✓
    
    Miner2->>Miner2: Select Transactions
    Miner2->>Miner2: Create Block Header
    Miner2->>Miner2: Hash Block (nonce=0)
    Note over Miner2: ... mining ...
    
    Miner3->>Miner3: Select Transactions
    Miner3->>Miner3: Create Block Header
    Miner3->>Miner3: Hash Block (nonce=0)
    Note over Miner3: ... mining ...
    
    Note over Miner1: Block Found!
    Miner1->>Node1: block(header, transactions)
    Miner1->>Node2: block(header, transactions)
    Miner1->>Node3: block(header, transactions)
    
    Node1->>Node1: Validate Block
    Node1->>Node1: Verify PoW
    Node1->>Node1: Verify Transactions
    Node2->>Node2: Validate Block
    Node2->>Node2: Verify PoW
    Node2->>Node2: Verify Transactions
    Node3->>Node3: Validate Block
    Node3->>Node3: Verify PoW
    Node3->>Node3: Verify Transactions
    
    Node1->>Node1: Add to Chain
    Node2->>Node2: Add to Chain
    Node3->>Node3: Add to Chain
    
    Note over Miner2,Miner3: Block Received
    Miner2->>Miner2: Stop Mining Current Block
    Miner2->>Miner2: Start Mining Next Block
    Miner3->>Miner3: Stop Mining Current Block
    Miner3->>Miner3: Start Mining Next Block
    
    Note over Miner1,Miner3: Next Block Height
```

**Overall Flow Diagram**:
```mermaid
graph TB
    A[Transaction Pool<br/>Mempool] --> B[Miner 1]
    A --> C[Miner 2]
    A --> D[Miner 3]
    A --> E[Miner N]
    
    B --> F[Select Transactions<br/>Prioritize by Fee]
    C --> F
    D --> F
    E --> F
    
    F --> G[Create Block Candidate<br/>Previous Block Hash<br/>Merkle Root<br/>Timestamp<br/>Difficulty Target<br/>Nonce = 0]
    
    G --> H[Calculate Hash<br/>SHA256² Block Header]
    H --> I{Hash < Target<br/>Difficulty?}
    
    I -->|No| J[Increment Nonce]
    J --> K{Nonce<br/>Exhausted?}
    K -->|No| H
    K -->|Yes| L[Change ExtraNonce<br/>Update Merkle Root]
    L --> H
    
    I -->|Yes| M[Block Found!<br/>Valid PoW]
    M --> N[Broadcast Block<br/>to Network]
    
    N --> O[Node 1]
    N --> P[Node 2]
    N --> Q[Node 3]
    N --> R[Node N]
    
    O --> S[Validate Block]
    P --> S
    Q --> S
    R --> S
    
    S --> T{Block Valid?<br/>Valid PoW?<br/>Valid Transactions?<br/>Valid Merkle Root?}
    T -->|Yes| U[Add to Chain<br/>Update UTXO Set]
    T -->|No| V[Reject Block]
    
    U --> W{Longest Chain?}
    W -->|Yes| X[Continue Mining<br/>on This Chain]
    W -->|No| Y[Switch to<br/>Longest Chain]
    
    X --> A
    Y --> A
    V --> A
    
    style A fill:#FFE4B5
    style G fill:#87CEEB
    style M fill:#90EE90
    style I fill:#FFD700
    style T fill:#FFD700
    style W fill:#FFD700
```

**Individual Node Decision Diagram**:
```mermaid
graph TB
    A[Miner Node] --> B[Collect Transactions<br/>from Mempool]
    B --> C[Select Transactions<br/>Build Coinbase TX<br/>Calculate Fees]
    C --> D[Create Block Header<br/>Previous Hash = Chain Tip<br/>Merkle Root<br/>Timestamp<br/>Difficulty Target<br/>Nonce = 0]
    
    D --> E[Calculate Hash<br/>H = SHA256² Header]
    E --> F{Hash < Target?}
    
    F -->|No| G[Increment Nonce]
    G --> H{Nonce < 2³²?}
    H -->|Yes| E
    H -->|No| I[Increment ExtraNonce<br/>Rebuild Merkle Tree]
    I --> D
    
    F -->|Yes| J[Block Found!<br/>Valid PoW]
    J --> K[Broadcast Block<br/>to Peers]
    
    K --> L[Wait for Propagation]
    L --> M[Continue Mining<br/>Next Block]
    M --> B
    
    N[Receiving Node] --> O{Received<br/>Block?}
    O -->|Yes| P[Validate Block]
    O -->|No| Q[Continue Waiting]
    
    P --> R{Valid PoW?<br/>Hash < Target?}
    R -->|No| S[Reject Block]
    
    R -->|Yes| T{Valid Transactions?<br/>No Double Spends?<br/>Valid Merkle Root?}
    T -->|No| S
    
    T -->|Yes| U{Block Extends<br/>Current Chain?}
    U -->|Yes| V[Add to Chain<br/>Update UTXO]
    U -->|No| W{Longer Chain<br/>Available?}
    
    W -->|Yes| X[Reorganize Chain<br/>Switch to Longest]
    W -->|No| Y[Store as Orphan<br/>Wait for Parent]
    
    V --> Z[Block Accepted<br/>Continue]
    X --> Z
    Y --> AA{Parent<br/>Received?}
    AA -->|Yes| P
    AA -->|No| Q
    
    S --> Q
    Z --> Q
    Q --> O
    
    style A fill:#FFE4B5
    style F fill:#FFD700
    style R fill:#FFD700
    style T fill:#FFD700
    style U fill:#FFD700
    style J fill:#90EE90
    style V fill:#90EE90
```

**Bitcoin Consensus Process**:

**1. Transaction Collection**:
- Miners collect transactions from mempool
- Select transactions (prioritize fees)
- Create block candidate

**2. Block Structure**:
```
Block Header:
- Previous Block Hash
- Merkle Root (transactions)
- Timestamp
- Difficulty Target
- Nonce
```

**3. Mining (Proof-of-Work)**:
- Calculate hash: `SHA256(SHA256(BlockHeader))`
- Check if hash < target difficulty
- If not: Increment nonce, repeat
- If yes: Block found!

**4. Block Propagation**:
- Broadcast block to network
- Other nodes validate
- If valid: Add to chain

**5. Chain Selection**:
- Always extend longest valid chain
- Fork resolution: Longest chain wins
- Orphaned blocks: No reward

**Key Properties**:
- **Security**: Computational security
- **Decentralization**: Anyone can mine
- **Finality**: Probabilistic (6 confirmations)
- **Energy**: High energy consumption

**Example**:
```python
import hashlib
import time

class BitcoinMiner:
    def __init__(self, difficulty_target):
        self.difficulty_target = difficulty_target
    
    def mine_block(self, transactions, previous_hash):
        # Create block
        block = {
            'previous_hash': previous_hash,
            'merkle_root': self.calculate_merkle_root(transactions),
            'timestamp': int(time.time()),
            'nonce': 0,
            'transactions': transactions
        }
        
        # Mine (find nonce)
        while True:
            block['nonce'] += 1
            block_hash = self.hash_block(block)
            
            if int(block_hash, 16) < self.difficulty_target:
                return block, block_hash
    
    def hash_block(self, block):
        header = (
            block['previous_hash'] +
            block['merkle_root'] +
            str(block['timestamp']) +
            str(block['nonce'])
        )
        return hashlib.sha256(
            hashlib.sha256(header.encode()).digest()
        ).hexdigest()
    
    def calculate_merkle_root(self, transactions):
        # Simplified Merkle tree calculation
        if len(transactions) == 0:
            return "0" * 64
        
        # Hash all transactions
        hashes = [hashlib.sha256(str(tx).encode()).hexdigest() 
                  for tx in transactions]
        
        # Build Merkle tree
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # Duplicate last if odd
            
            hashes = [
                hashlib.sha256((hashes[i] + hashes[i+1]).encode()).hexdigest()
                for i in range(0, len(hashes), 2)
            ]
        
        return hashes[0]
```

**Difficulty Adjustment**:
- Every 2016 blocks (~2 weeks)
- Target time: 10 minutes per block
- Adjust difficulty to maintain rate

**Use Cases**:
- Bitcoin
- Litecoin
- Many PoW blockchains

---

