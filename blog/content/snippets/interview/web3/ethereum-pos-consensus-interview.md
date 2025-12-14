---
title: "Ethereum Proof-of-Stake Consensus Interview Questions"
date: 2025-12-14
tags: ["consensus", "blockchain", "interview", "ethereum", "pos", "casper", "beacon-chain"]
---

Ethereum Proof-of-Stake consensus algorithm interview questions covering Casper FFG and LMD GHOST.

## Q1: How does Ethereum consensus work (Proof-of-Stake)?

**Answer**:

**Ethereum 2.0** uses Proof-of-Stake consensus (Casper FFG + LMD GHOST).

**Sequence Diagram**:
```mermaid
sequenceDiagram
    participant Proposer
    participant Validator1
    participant Validator2
    participant Validator3
    participant Validator4
    
    Note over Proposer,Validator4: Epoch E, Slot S
    
    Note over Proposer: Beacon Block Proposal
    Proposer->>Proposer: Select as Proposer (RANDAO)
    Proposer->>Proposer: Collect Attestations
    Proposer->>Proposer: Get Execution Layer Block
    Proposer->>Proposer: Build Beacon Block
    
    Proposer->>Validator1: beacon_block(slot, attestations, execution_payload)
    Proposer->>Validator2: beacon_block(slot, attestations, execution_payload)
    Proposer->>Validator3: beacon_block(slot, attestations, execution_payload)
    Proposer->>Validator4: beacon_block(slot, attestations, execution_payload)
    
    Note over Validator1,Validator4: Validate & Attest
    Validator1->>Validator1: Validate Block
    Validator1->>Validator1: Create Attestation
    Validator1->>Validator2: attestation(slot, block_root, source, target)
    Validator1->>Validator3: attestation(slot, block_root, source, target)
    Validator1->>Validator4: attestation(slot, block_root, source, target)
    
    Validator2->>Validator1: attestation(slot, block_root, source, target)
    Validator2->>Validator3: attestation(slot, block_root, source, target)
    Validator2->>Validator4: attestation(slot, block_root, source, target)
    
    Validator3->>Validator1: attestation(slot, block_root, source, target)
    Validator3->>Validator2: attestation(slot, block_root, source, target)
    Validator3->>Validator4: attestation(slot, block_root, source, target)
    
    Validator4->>Validator1: attestation(slot, block_root, source, target)
    Validator4->>Validator2: attestation(slot, block_root, source, target)
    Validator4->>Validator3: attestation(slot, block_root, source, target)
    
    Note over Validator1,Validator4: Fork Choice (LMD GHOST)
    Validator1->>Validator1: Follow Chain with Most Attestations
    Validator2->>Validator2: Follow Chain with Most Attestations
    Validator3->>Validator3: Follow Chain with Most Attestations
    Validator4->>Validator4: Follow Chain with Most Attestations
    
    Note over Proposer,Validator4: Slot S+1
    
    Note over Proposer,Validator4: End of Epoch
    Note over Proposer,Validator4: Calculate Checkpoints
    Note over Validator1,Validator4: 2/3+ Stake Votes for Checkpoints
    Note over Proposer,Validator4: Finalize Checkpoint (Casper FFG)
    Note over Proposer,Validator4: Epoch E+1 Starts
```

**Overall Flow Diagram**:
```mermaid
graph TB
    A[Epoch E Starts] --> B[Validator Set<br/>Active Validators]
    B --> C[Slot 0]
    
    C --> D[Select Beacon Chain<br/>Proposer via RANDAO]
    D --> E[Proposer Creates<br/>Beacon Block]
    E --> F[Include Attestations<br/>from Previous Epoch]
    F --> G[Broadcast Beacon Block]
    
    G --> H[Validators Receive Block]
    H --> I[Attestation Phase<br/>Validators Attest]
    
    I --> J[Validator 1 Attests]
    I --> K[Validator 2 Attests]
    I --> L[Validator 3 Attests]
    I --> M[Validator N Attests]
    
    J --> N[Collect Attestations]
    K --> N
    L --> N
    M --> N
    
    N --> O[Slot S+1]
    O --> P{End of Slot?}
    P -->|No| Q[Continue Attestations]
    Q --> O
    P -->|Yes| R{End of Epoch?}
    
    R -->|No| S[Next Slot]
    S --> D
    
    R -->|Yes| T[Epoch Transition]
    T --> U[Calculate Checkpoints<br/>Source & Target]
    U --> V{2/3+ Stake<br/>Voted for<br/>Checkpoint?}
    
    V -->|Yes| W[Finalize Checkpoint<br/>Blocks Cannot Revert]
    V -->|No| X[Not Finalized<br/>Continue]
    
    W --> Y[Update Validator Set<br/>Process Slashings<br/>Calculate Rewards]
    X --> Y
    Y --> Z[Epoch E+1 Starts]
    Z --> B
    
    style A fill:#FFE4B5
    style D fill:#87CEEB
    style W fill:#90EE90
    style V fill:#FFD700
    style T fill:#FFD700
```

**Individual Node Decision Diagram**:
```mermaid
graph TB
    A[Validator at Slot S] --> B{Am I Beacon<br/>Block Proposer?}
    
    B -->|Yes| C[Create Beacon Block]
    C --> D[Collect Attestations<br/>from Previous Epoch]
    D --> E[Select Transactions<br/>from Execution Layer]
    E --> F[Build Block<br/>Beacon Block Root<br/>State Root<br/>Attestations]
    F --> G[Sign Block]
    G --> H[Broadcast Beacon Block]
    
    B -->|No| I[Wait for Beacon Block]
    
    H --> J[Block Propagated]
    I --> K{Received Beacon<br/>Block in Time?}
    K -->|Yes| L[Validate Block]
    K -->|No| M[Miss Block<br/>Penalty Applied]
    
    L --> N{Block Valid?<br/>Valid Signature?<br/>Valid Attestations?<br/>Valid State Transition?}
    N -->|No| O[Reject Block]
    N -->|Yes| P[Accept Block]
    
    P --> Q{Should I Attest?<br/>Am I Assigned<br/>to Attest?}
    Q -->|Yes| R[Create Attestation<br/>Beacon Block Root<br/>Source Checkpoint<br/>Target Checkpoint]
    Q -->|No| S[Skip Attestation]
    
    R --> T[Sign Attestation]
    T --> U[Broadcast Attestation]
    
    J --> V[Other Validators Validate]
    V --> W[Fork Choice Rule<br/>LMD GHOST]
    W --> X[Follow Chain with<br/>Most Attestations]
    
    P --> Y[Update Local State]
    O --> Y
    U --> Y
    S --> Y
    M --> Y
    
    Y --> Z[Slot S+1]
    Z --> AA{End of Epoch?}
    AA -->|No| A
    AA -->|Yes| AB[Epoch Transition]
    
    AB --> AC[Calculate Checkpoints<br/>Source: Previous Justified<br/>Target: Current Epoch]
    AC --> AD{2/3+ Stake<br/>Voted for Both<br/>Checkpoints?}
    
    AD -->|Yes| AE[Finalize Checkpoint<br/>Blocks Locked]
    AD -->|No| AF[Not Finalized<br/>Continue]
    
    AE --> AG[Process Rewards/Penalties<br/>Update Validator Set<br/>Process Exits]
    AF --> AG
    AG --> AH[Epoch E+1 Starts]
    AH --> A
    
    style A fill:#FFE4B5
    style B fill:#FFD700
    style N fill:#FFD700
    style Q fill:#FFD700
    style AD fill:#FFD700
    style P fill:#90EE90
    style AE fill:#90EE90
```

**Ethereum PoS Components**:

**1. Beacon Chain**:
- Coordinates validators
- Manages validator set
- Handles finality

**2. Validator Duties**:
- **Proposing**: Create blocks (selected randomly)
- **Attesting**: Vote on blocks (every epoch)
- **Sync Committee**: Light client support

**3. Slot and Epoch**:
- **Slot**: 12 seconds (one block)
- **Epoch**: 32 slots (~6.4 minutes)
- **Finality**: ~2 epochs (~13 minutes)

**4. Fork Choice (LMD GHOST)**:
- Latest Message Driven Greedy Heaviest Observed Subtree
- Follows chain with most attestations
- Resolves forks

**5. Finality (Casper FFG)**:
- Finalized blocks cannot be reverted
- Requires `2/3+` validator stake
- Checkpoint every epoch

**Key Properties**:
- **Security**: Economic security (stake at risk)
- **Energy**: 99.9% less energy than PoW
- **Finality**: Checkpoint finality
- **Scalability**: Sharding support

**Example**:
```python
class EthereumValidator:
    def __init__(self, validator_index, balance):
        self.validator_index = validator_index
        self.balance = balance  # In ETH
        self.activation_epoch = None
        self.exit_epoch = None
    
    def propose_block(self, slot, parent_block):
        # Selected as proposer
        if self.is_proposer(slot):
            block = {
                'slot': slot,
                'parent_root': parent_block.hash,
                'state_root': self.get_state_root(),
                'body': self.get_transactions(),
                'signature': self.sign_block()
            }
            return block
        return None
    
    def attest(self, slot, block_root):
        # Attest to block
        attestation = {
            'slot': slot,
            'index': self.validator_index,
            'beacon_block_root': block_root,
            'source': self.get_source_checkpoint(),
            'target': self.get_target_checkpoint(),
            'signature': self.sign_attestation()
        }
        return attestation
    
    def get_rewards(self, epoch):
        # Calculate rewards based on participation
        base_reward = self.calculate_base_reward()
        if self.attested_correctly(epoch):
            return base_reward
        else:
            # Penalty for missing attestations
            return -base_reward
```

**Slashing Conditions**:
- **Double Voting**: Two different attestations in same epoch
- **Surround Voting**: Attestation surrounds another
- **Proposer Violations**: Invalid block proposals

**Use Cases**:
- Ethereum 2.0
- High-security PoS systems

---

