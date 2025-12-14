---
title: "Byzantine Fault Tolerance (BFT) Consensus Interview Questions"
date: 2025-12-14
tags: ["consensus", "blockchain", "interview", "bft", "byzantine-fault-tolerance"]
---

Byzantine Fault Tolerance (BFT) consensus algorithm interview questions covering distributed systems that handle arbitrary failures.

## Q1: How does Byzantine Fault Tolerance (BFT) work?

**Answer**:

**Byzantine Fault Tolerance** handles arbitrary failures, including malicious behavior.

**Sequence Diagram**:
```mermaid
sequenceDiagram
    participant Client
    participant Primary
    participant Replica1
    participant Replica2
    participant Replica3
    participant Replica4
    
    Client->>Primary: Request(op)
    
    Note over Primary: Pre-Prepare Phase
    Primary->>Replica1: pre-prepare(n, v, m)
    Primary->>Replica2: pre-prepare(n, v, m)
    Primary->>Replica3: pre-prepare(n, v, m)
    Primary->>Replica4: pre-prepare(n, v, m)
    
    Note over Replica1,Replica4: Prepare Phase
    Replica1->>Replica2: prepare(n, v, i)
    Replica1->>Replica3: prepare(n, v, i)
    Replica1->>Replica4: prepare(n, v, i)
    Replica2->>Replica1: prepare(n, v, j)
    Replica2->>Replica3: prepare(n, v, j)
    Replica2->>Replica4: prepare(n, v, j)
    Replica3->>Replica1: prepare(n, v, k)
    Replica3->>Replica2: prepare(n, v, k)
    Replica3->>Replica4: prepare(n, v, k)
    Replica4->>Replica1: prepare(n, v, l)
    Replica4->>Replica2: prepare(n, v, l)
    Replica4->>Replica3: prepare(n, v, l)
    
    Note over Replica1,Replica4: Have 2f prepares
    
    Note over Replica1,Replica4: Commit Phase
    Replica1->>Replica2: commit(n, v, i)
    Replica1->>Replica3: commit(n, v, i)
    Replica1->>Replica4: commit(n, v, i)
    Replica2->>Replica1: commit(n, v, j)
    Replica2->>Replica3: commit(n, v, j)
    Replica2->>Replica4: commit(n, v, j)
    Replica3->>Replica1: commit(n, v, k)
    Replica3->>Replica2: commit(n, v, k)
    Replica3->>Replica4: commit(n, v, k)
    Replica4->>Replica1: commit(n, v, l)
    Replica4->>Replica2: commit(n, v, l)
    Replica4->>Replica3: commit(n, v, l)
    
    Note over Replica1,Replica4: Have 2f+1 commits
    
    Note over Replica1,Replica4: Execute & Reply
    Replica1->>Replica1: execute(op)
    Replica2->>Replica2: execute(op)
    Replica3->>Replica3: execute(op)
    Replica4->>Replica4: execute(op)
    
    Replica1->>Client: reply(result)
    Replica2->>Client: reply(result)
    Replica3->>Client: reply(result)
    Replica4->>Client: reply(result)
    
    Note over Client: Wait for f+1 matching replies
```

**Overall Flow Diagram**:
```mermaid
graph TB
    A[Client Request] --> B[Primary Node]
    B --> C[Pre-Prepare Phase<br/>Assign sequence n]
    C --> D[Broadcast Pre-Prepare<br/>n, request, view]
    
    D --> E[Replica 1]
    D --> F[Replica 2]
    D --> G[Replica 3]
    D --> H[Replica 4]
    
    E --> I[Prepare Phase<br/>Broadcast Prepare]
    F --> I
    G --> I
    H --> I
    
    I --> J{Collect 2f<br/>Prepare Messages}
    J -->|Yes| K[Commit Phase<br/>Broadcast Commit]
    J -->|No| L[Timeout<br/>View Change]
    L --> B
    
    K --> M[Replica 1]
    K --> N[Replica 2]
    K --> O[Replica 3]
    K --> P[Replica 4]
    
    M --> Q{Collect 2f+1<br/>Commit Messages}
    N --> Q
    O --> Q
    P --> Q
    
    Q -->|Yes| R[Execute Request<br/>Update State]
    Q -->|No| L
    
    R --> S[Reply Phase<br/>Send Reply to Client]
    S --> T[Client Waits for<br/>f+1 Matching Replies]
    
    style A fill:#FFE4B5
    style B fill:#87CEEB
    style R fill:#90EE90
    style J fill:#FFD700
    style Q fill:#FFD700
```

**Individual Node Decision Diagram**:
```mermaid
graph TB
    A[Node Receives Message] --> B{Node<br/>Role?}
    
    B -->|Primary| C{Message<br/>Type?}
    C -->|Client Request| D[Assign Sequence Number n]
    D --> E[Broadcast Pre-Prepare<br/>n, request, view]
    
    B -->|Replica| F{Message<br/>Type?}
    
    F -->|Pre-Prepare| G{Valid Sequence?<br/>Valid View?<br/>Valid Request?}
    G -->|Yes| H[Store Pre-Prepare<br/>Broadcast Prepare]
    G -->|No| I[Discard Message]
    
    F -->|Prepare| J{Valid Sequence?<br/>Valid View?<br/>Matching Pre-Prepare?}
    J -->|Yes| K[Store Prepare]
    K --> L{Have 2f<br/>Prepares?}
    L -->|Yes| M[Broadcast Commit]
    L -->|No| N[Wait for More]
    
    F -->|Commit| O{Valid Sequence?<br/>Valid View?}
    O -->|Yes| P[Store Commit]
    P --> Q{Have 2f+1<br/>Commits?}
    Q -->|Yes| R[Execute Request<br/>Update State]
    Q -->|No| S[Wait for More]
    
    R --> T[Send Reply to Client]
    
    F -->|View Change| U[Initiate View Change<br/>Select New Primary]
    
    style A fill:#FFE4B5
    style G fill:#FFD700
    style J fill:#FFD700
    style O fill:#FFD700
    style R fill:#90EE90
```

**BFT Requirements**:
- **Total Nodes**: `n = 3f + 1` (where `f` is max Byzantine nodes)
- **Honest Nodes**: `2f + 1` (majority)
- **Fault Tolerance**: Up to `f` Byzantine nodes

**BFT Phases**:

**1. Request Phase**:
- Client sends request to primary
- Primary broadcasts to all replicas

**2. Pre-Prepare Phase**:
- Primary assigns sequence number
- Broadcasts pre-prepare message

**3. Prepare Phase**:
- Replicas broadcast prepare messages
- Wait for `2f` matching prepares

**4. Commit Phase**:
- Replicas broadcast commit messages
- Wait for `2f + 1` commits (including self)

**5. Reply Phase**:
- Execute request
- Send reply to client
- Client waits for `f + 1` matching replies

**Example**:
```python
class BFTNode:
    def __init__(self, node_id, total_nodes):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = (total_nodes - 1) // 3
        self.quorum = 2 * self.f + 1
        self.log = {}
    
    def pre_prepare(self, sequence, request):
        # Primary assigns sequence number
        self.log[sequence] = {
            'request': request,
            'prepares': set([self.node_id]),
            'commits': set()
        }
        return {'pre_prepare': (sequence, request)}
    
    def prepare(self, sequence, request):
        if sequence in self.log:
            self.log[sequence]['prepares'].add(self.node_id)
            if len(self.log[sequence]['prepares']) >= self.quorum:
                return {'prepared': True}
        return {'prepared': False}
    
    def commit(self, sequence):
        if sequence in self.log:
            self.log[sequence]['commits'].add(self.node_id)
            if len(self.log[sequence]['commits']) >= self.quorum:
                # Execute request
                return {'committed': True, 'result': self.execute(sequence)}
        return {'committed': False}
```

**Use Cases**:
- Hyperledger Fabric
- Stellar
- Ripple

---

