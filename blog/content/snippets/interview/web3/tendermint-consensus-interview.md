---
title: "Tendermint Consensus Interview Questions"
date: 2025-12-14
tags: ["consensus", "blockchain", "interview", "tendermint", "cosmos", "bft"]
---

Tendermint consensus algorithm interview questions covering the Byzantine Fault Tolerant consensus used in Cosmos chains.

## Q1: How does Tendermint consensus work?

**Answer**:

**Tendermint** is a Byzantine Fault Tolerant consensus algorithm used in Cosmos chains.

**Sequence Diagram**:
```mermaid
sequenceDiagram
    participant Proposer
    participant Validator1
    participant Validator2
    participant Validator3
    participant Validator4
    
    Note over Proposer,Validator4: Height H, Round R
    
    Note over Proposer: Propose Phase
    Proposer->>Proposer: Create Block
    Proposer->>Validator1: proposal(H, R, block)
    Proposer->>Validator2: proposal(H, R, block)
    Proposer->>Validator3: proposal(H, R, block)
    Proposer->>Validator4: proposal(H, R, block)
    
    Note over Validator1,Validator4: Prevote Phase
    Validator1->>Validator2: prevote(H, R, blockID)
    Validator1->>Validator3: prevote(H, R, blockID)
    Validator1->>Validator4: prevote(H, R, blockID)
    Validator2->>Validator1: prevote(H, R, blockID)
    Validator2->>Validator3: prevote(H, R, blockID)
    Validator2->>Validator4: prevote(H, R, blockID)
    Validator3->>Validator1: prevote(H, R, blockID)
    Validator3->>Validator2: prevote(H, R, blockID)
    Validator3->>Validator4: prevote(H, R, blockID)
    Validator4->>Validator1: prevote(H, R, blockID)
    Validator4->>Validator2: prevote(H, R, blockID)
    Validator4->>Validator3: prevote(H, R, blockID)
    
    Note over Validator1,Validator4: Have 2/3+ prevotes
    
    Note over Validator1,Validator4: Precommit Phase
    Validator1->>Validator2: precommit(H, R, blockID)
    Validator1->>Validator3: precommit(H, R, blockID)
    Validator1->>Validator4: precommit(H, R, blockID)
    Validator2->>Validator1: precommit(H, R, blockID)
    Validator2->>Validator3: precommit(H, R, blockID)
    Validator2->>Validator4: precommit(H, R, blockID)
    Validator3->>Validator1: precommit(H, R, blockID)
    Validator3->>Validator2: precommit(H, R, blockID)
    Validator3->>Validator4: precommit(H, R, blockID)
    Validator4->>Validator1: precommit(H, R, blockID)
    Validator4->>Validator2: precommit(H, R, blockID)
    Validator4->>Validator3: precommit(H, R, blockID)
    
    Note over Validator1,Validator4: Have 2/3+ precommits
    
    Note over Validator1,Validator4: Commit Block
    Validator1->>Validator1: commit(H, block)
    Validator2->>Validator2: commit(H, block)
    Validator3->>Validator3: commit(H, block)
    Validator4->>Validator4: commit(H, block)
    
    Note over Proposer,Validator4: Height H+1, Round 0
```

**Overall Flow Diagram**:
```mermaid
graph TB
    A[Height N, Round 0] --> B[Select Proposer<br/>by Voting Power]
    B --> C[Proposer Creates Block]
    C --> D[Broadcast Proposal<br/>Block + POLRound]
    
    D --> E[Validator 1]
    D --> F[Validator 2]
    D --> G[Validator 3]
    D --> H[Validator 4]
    
    E --> I[Prevote Phase<br/>Validate Block]
    F --> I
    G --> I
    H --> I
    
    I --> J{Block Valid?}
    J -->|Yes| K[Broadcast Prevote<br/>BlockID]
    J -->|No| L[Broadcast Prevote<br/>nil]
    
    K --> M[Collect Prevotes]
    L --> M
    
    M --> N{Have 2/3+<br/>Prevotes for<br/>BlockID?}
    N -->|Yes| O[Precommit Phase<br/>Broadcast Precommit]
    N -->|No| P[Timeout<br/>Increment Round]
    P --> B
    
    O --> Q[Validator 1]
    O --> R[Validator 2]
    O --> S[Validator 3]
    O --> T[Validator 4]
    
    Q --> U[Collect Precommits]
    R --> U
    S --> U
    T --> U
    
    U --> V{Have 2/3+<br/>Precommits for<br/>BlockID?}
    V -->|Yes| W[Commit Block<br/>Update State]
    V -->|No| P
    
    W --> X[Height N+1<br/>Round 0]
    
    style A fill:#FFE4B5
    style B fill:#87CEEB
    style W fill:#90EE90
    style N fill:#FFD700
    style V fill:#FFD700
```

**Individual Node Decision Diagram**:
```mermaid
graph TB
    A[Node at Height H, Round R] --> B{Am I<br/>Proposer?}
    
    B -->|Yes| C[Create Block<br/>with Transactions]
    C --> D[Sign Block]
    D --> E[Broadcast Proposal<br/>Block + POLRound]
    E --> F[Wait for Prevotes]
    
    B -->|No| G[Wait for Proposal]
    
    G --> H{Received<br/>Proposal?}
    H -->|Yes| I{Block Valid?<br/>Check Signature<br/>Check Transactions}
    H -->|No| J[Timeout<br/>Prevote nil]
    
    I -->|Yes| K[Broadcast Prevote<br/>BlockID]
    I -->|No| L[Broadcast Prevote<br/>nil]
    
    F --> M[Prevote Phase]
    K --> M
    L --> M
    J --> M
    
    M --> N{Received 2/3+<br/>Prevotes for<br/>BlockID?}
    N -->|Yes| O[Lock on BlockID<br/>Broadcast Precommit]
    N -->|No| P{Timeout?}
    P -->|Yes| Q[Increment Round<br/>New Proposer]
    P -->|No| M
    Q --> B
    
    O --> R[Precommit Phase]
    R --> S{Received 2/3+<br/>Precommits for<br/>BlockID?}
    S -->|Yes| T[Commit Block<br/>Update State]
    S -->|No| U{Timeout?}
    U -->|Yes| Q
    U -->|No| R
    
    T --> V[Height H+1<br/>Round 0]
    V --> B
    
    style A fill:#FFE4B5
    style B fill:#FFD700
    style I fill:#FFD700
    style N fill:#FFD700
    style S fill:#FFD700
    style T fill:#90EE90
```

**Tendermint Phases**:

**1. Propose**:
- Proposer selected (round-robin by voting power)
- Creates block with transactions
- Broadcasts proposal

**2. Prevote**:
- Validators receive proposal
- Validate block
- Broadcast prevote (yes/no)
- Need `2/3+` prevotes to continue

**3. Precommit**:
- After `2/3+` prevotes, broadcast precommit
- Need `2/3+` precommits to commit
- If timeout: Move to next round

**4. Commit**:
- Block committed when `2/3+` precommits received
- State updated
- Move to next height

**Key Properties**:
- **Finality**: Blocks are final (no reorgs)
- **Safety**: Tolerates up to `1/3` Byzantine validators
- **Liveness**: Continues with `2/3+` honest validators
- **Fast**: ~1-6 second block time

**Example**:
```go
type TendermintState struct {
    Height int64
    Round  int32
    Step   Step
}

type Step int

const (
    StepPropose Step = iota
    StepPrevote
    StepPrecommit
    StepCommit
)

func (s *TendermintState) Propose(block *Block) {
    // Proposer creates and broadcasts block
    s.Step = StepPropose
    broadcastProposal(block)
}

func (s *TendermintState) Prevote(blockID BlockID) {
    // Validators vote on proposal
    s.Step = StepPrevote
    if validateBlock(blockID) {
        broadcastPrevote(blockID, true)
    } else {
        broadcastPrevote(blockID, false)
    }
}

func (s *TendermintState) Precommit(blockID BlockID) {
    // After 2/3+ prevotes, precommit
    if countPrevotes(blockID) >= getQuorum() {
        s.Step = StepPrecommit
        broadcastPrecommit(blockID, true)
    }
}

func (s *TendermintState) Commit(blockID BlockID) {
    // After 2/3+ precommits, commit block
    if countPrecommits(blockID) >= getQuorum() {
        s.Step = StepCommit
        commitBlock(blockID)
        s.Height++
        s.Round = 0
    }
}
```

**Use Cases**:
- Cosmos chains
- Binance Chain
- Terra

---

