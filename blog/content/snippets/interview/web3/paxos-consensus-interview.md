---
title: "Paxos Consensus Interview Questions"
date: 2025-12-14
tags: ["consensus", "blockchain", "interview", "paxos", "distributed-systems"]
---

Paxos consensus algorithm interview questions covering the classic distributed consensus protocol.

## Q1: How does Paxos consensus work?

**Answer**:

**Paxos** is a consensus algorithm for distributed systems that ensures agreement among nodes even with failures.

**Sequence Diagram**:
```mermaid
sequenceDiagram
    participant Client
    participant Proposer
    participant Acceptor1
    participant Acceptor2
    participant Acceptor3
    participant Acceptor4
    participant Acceptor5
    
    Client->>Proposer: Request(value)
    
    Note over Proposer: Phase 1: Prepare
    Proposer->>Acceptor1: prepare(n)
    Proposer->>Acceptor2: prepare(n)
    Proposer->>Acceptor3: prepare(n)
    Proposer->>Acceptor4: prepare(n)
    Proposer->>Acceptor5: prepare(n)
    
    Acceptor1->>Proposer: promise(n, accepted_value)
    Acceptor2->>Proposer: promise(n, null)
    Acceptor3->>Proposer: promise(n, null)
    Acceptor4->>Proposer: promise(n, null)
    Acceptor5->>Proposer: promise(n, null)
    
    Note over Proposer: Quorum reached (3/5)
    Note over Proposer: Use accepted_value if any
    
    Note over Proposer: Phase 2: Accept
    Proposer->>Acceptor1: accept(n, v)
    Proposer->>Acceptor2: accept(n, v)
    Proposer->>Acceptor3: accept(n, v)
    Proposer->>Acceptor4: accept(n, v)
    Proposer->>Acceptor5: accept(n, v)
    
    Acceptor1->>Proposer: accepted(n, v)
    Acceptor2->>Proposer: accepted(n, v)
    Acceptor3->>Proposer: accepted(n, v)
    Acceptor4->>Proposer: accepted(n, v)
    Acceptor5->>Proposer: accepted(n, v)
    
    Note over Proposer: Quorum reached (3/5)
    Proposer->>Client: Consensus(value)
```

**Overall Flow Diagram**:
```mermaid
graph TB
    A[Client Request] --> B[Proposer Node]
    B --> C[Phase 1: Prepare<br/>Broadcast prepare n]
    C --> D[Acceptor 1]
    C --> E[Acceptor 2]
    C --> F[Acceptor 3]
    C --> G[Acceptor 4]
    C --> H[Acceptor 5]
    
    D --> I[Collect Responses]
    E --> I
    F --> I
    G --> I
    H --> I
    
    I --> J{Quorum<br/>n/2+1<br/>Promises?}
    J -->|Yes| K[Phase 2: Accept<br/>Broadcast accept n,v]
    J -->|No| L[Retry with<br/>Higher n]
    L --> C
    
    K --> M[Acceptor 1]
    K --> N[Acceptor 2]
    K --> O[Acceptor 3]
    K --> P[Acceptor 4]
    K --> Q[Acceptor 5]
    
    M --> R[Collect Accepts]
    N --> R
    O --> R
    P --> R
    Q --> R
    
    R --> S{Quorum<br/>n/2+1<br/>Accepts?}
    S -->|Yes| T[Consensus Reached<br/>Value Chosen]
    S -->|No| L
    
    style A fill:#FFE4B5
    style B fill:#87CEEB
    style T fill:#90EE90
    style J fill:#FFD700
    style S fill:#FFD700
```

**Individual Node Decision Diagram**:
```mermaid
graph TB
    A[Node Receives Message] --> B{Message<br/>Type?}
    
    B -->|Prepare n| C{Is n ><br/>highest_seen?}
    C -->|Yes| D[Update highest_seen = n<br/>Promise not to accept < n]
    D --> E[Return Promise<br/>with accepted_value]
    C -->|No| F[Return Reject]
    
    B -->|Accept n,v| G{Is n >=<br/>highest_seen?}
    G -->|Yes| H[Update highest_seen = n<br/>Store accepted_value = v]
    H --> I[Return Accepted]
    G -->|No| J[Return Reject]
    
    B -->|Learn Request| K[Value Already Chosen<br/>Return Value]
    
    E --> L[Send Response]
    F --> L
    I --> L
    J --> L
    K --> L
    
    style A fill:#FFE4B5
    style C fill:#FFD700
    style G fill:#FFD700
    style D fill:#90EE90
    style H fill:#90EE90
```

**Paxos Phases**:

**Phase 1: Prepare**
1. Proposer sends `prepare(n)` with proposal number `n`
2. Acceptors respond:
   - If `n > highest_seen`: Promise not to accept proposals < `n`, return highest accepted value
   - Otherwise: Reject

**Phase 2: Accept**
1. If majority promise: Proposer sends `accept(n, v)` with value `v`
2. Acceptors accept if `n >= highest_seen`
3. If majority accept: Consensus reached

**Key Properties**:
- **Safety**: Only one value can be chosen
- **Liveness**: Eventually reaches consensus (if no failures)
- **Fault Tolerance**: Works with up to (n-1)/2 failures

**Example**:
```python
class PaxosNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.highest_seen = 0
        self.accepted_value = None
        self.accepted_proposal = 0
    
    def prepare(self, proposal_num):
        if proposal_num > self.highest_seen:
            self.highest_seen = proposal_num
            return {
                'promise': True,
                'accepted_proposal': self.accepted_proposal,
                'accepted_value': self.accepted_value
            }
        return {'promise': False}
    
    def accept(self, proposal_num, value):
        if proposal_num >= self.highest_seen:
            self.highest_seen = proposal_num
            self.accepted_proposal = proposal_num
            self.accepted_value = value
            return {'accepted': True}
        return {'accepted': False}
```

**Use Cases**:
- Distributed databases
- Configuration management
- State machine replication

---

