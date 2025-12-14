---
title: "Polkadot (NPoS) Consensus Interview Questions"
date: 2025-12-14
tags: ["consensus", "blockchain", "interview", "polkadot", "npos", "babe", "grandpa"]
---

Polkadot consensus algorithm interview questions covering Nominated Proof-of-Stake (NPoS) with BABE and GRANDPA.

## Q1: How does Polkadot (NPoS) consensus work?

**Answer**:

**Polkadot** uses Nominated Proof-of-Stake (NPoS) with BABE + GRANDPA.

**Sequence Diagram**:
```mermaid
sequenceDiagram
    participant Leader
    participant Validator1
    participant Validator2
    participant Validator3
    participant Validator4
    
    Note over Leader,Validator4: Era E, Slot S
    
    Note over Leader: BABE: Block Production
    Leader->>Leader: Compute VRF(era, slot, key)
    Leader->>Leader: Check VRF < threshold
    
    Note over Leader: Create Block
    Leader->>Leader: Collect Extrinsics
    Leader->>Leader: Build Block
    Leader->>Leader: Sign Block
    
    Leader->>Validator1: block(slot, extrinsics, vrf_proof)
    Leader->>Validator2: block(slot, extrinsics, vrf_proof)
    Leader->>Validator3: block(slot, extrinsics, vrf_proof)
    Leader->>Validator4: block(slot, extrinsics, vrf_proof)
    
    Note over Validator1,Validator4: Validate Block
    Validator1->>Validator1: Verify VRF Proof
    Validator1->>Validator1: Verify Signature
    Validator1->>Validator1: Validate Extrinsics
    Validator1->>Validator1: Validate State Transition
    
    Validator2->>Validator2: Verify VRF Proof
    Validator2->>Validator2: Verify Signature
    Validator2->>Validator2: Validate Extrinsics
    Validator2->>Validator2: Validate State Transition
    
    Validator3->>Validator3: Verify VRF Proof
    Validator3->>Validator3: Verify Signature
    Validator3->>Validator3: Validate Extrinsics
    Validator3->>Validator3: Validate State Transition
    
    Validator4->>Validator4: Verify VRF Proof
    Validator4->>Validator4: Verify Signature
    Validator4->>Validator4: Validate Extrinsics
    Validator4->>Validator4: Validate State Transition
    
    Note over Validator1,Validator4: Add to Chain
    Validator1->>Validator1: Update State
    Validator2->>Validator2: Update State
    Validator3->>Validator3: Update State
    Validator4->>Validator4: Update State
    
    Note over Leader,Validator4: Slot S+1
    
    Note over Leader,Validator4: GRANDPA: Finalization
    Note over Validator1,Validator4: Round R
    Validator1->>Validator2: vote(block_hash, stake_weight)
    Validator1->>Validator3: vote(block_hash, stake_weight)
    Validator1->>Validator4: vote(block_hash, stake_weight)
    
    Validator2->>Validator1: vote(block_hash, stake_weight)
    Validator2->>Validator3: vote(block_hash, stake_weight)
    Validator2->>Validator4: vote(block_hash, stake_weight)
    
    Validator3->>Validator1: vote(block_hash, stake_weight)
    Validator3->>Validator2: vote(block_hash, stake_weight)
    Validator3->>Validator4: vote(block_hash, stake_weight)
    
    Validator4->>Validator1: vote(block_hash, stake_weight)
    Validator4->>Validator2: vote(block_hash, stake_weight)
    Validator4->>Validator3: vote(block_hash, stake_weight)
    
    Note over Validator1,Validator4: 2/3+ Stake Voted
    Note over Validator1,Validator4: Finalize Block & Ancestors
    
    Note over Leader,Validator4: End of Era
    Note over Leader,Validator4: Phragmén Election
    Note over Leader,Validator4: Update Validator Set
    Note over Leader,Validator4: Era E+1 Starts
```

**Overall Flow Diagram**:
```mermaid
graph TB
    A[Era Starts] --> B[Phragmén Election<br/>Select Validators]
    B --> C[Validator Set<br/>Active Validators]
    
    C --> D[BABE: Block Production]
    D --> E[Slot S]
    E --> F[VRF Selection<br/>Compute VRF Output]
    F --> G{VRF Output <<br/>Threshold?}
    
    G -->|Yes| H[Selected as Leader]
    G -->|No| I[Not Selected]
    
    H --> J[Create Block<br/>Collect Extrinsics]
    J --> K[Sign Block with<br/>VRF Proof]
    K --> L[Broadcast Block]
    
    I --> M[Wait for Block]
    L --> N[Network Propagation]
    N --> M
    
    M --> O[Validate Block<br/>Check VRF Proof]
    O --> P{Block Valid?}
    P -->|Yes| Q[Add to Chain]
    P -->|No| R[Reject Block]
    
    Q --> S[Slot S+1]
    R --> S
    S --> T{End of Slot?}
    T -->|No| E
    T -->|Yes| U[GRANDPA: Finalization]
    
    U --> V[Round R]
    V --> W[Validators Vote<br/>on Best Chain]
    W --> X[Collect Votes<br/>with Stake Weight]
    X --> Y{2/3+ Stake<br/>Voted for<br/>Same Block?}
    
    Y -->|Yes| Z[Finalize Block<br/>and Ancestors]
    Y -->|No| AA[Round R+1]
    AA --> W
    
    Z --> AB[Blocks Finalized<br/>Cannot Revert]
    AB --> AC[Next Round]
    AC --> V
    
    Q --> AD{End of Era?}
    AD -->|No| E
    AD -->|Yes| AE[Era Transition]
    AE --> AF[Update Validator Set<br/>Process Rewards<br/>Phragmén Election]
    AF --> A
    
    style A fill:#FFE4B5
    style B fill:#87CEEB
    style Z fill:#90EE90
    style G fill:#FFD700
    style P fill:#FFD700
    style Y fill:#FFD700
```

**Individual Node Decision Diagram**:
```mermaid
graph TB
    A[Node at Slot S] --> B{Am I Validator?}
    
    B -->|No| C[Not Validator<br/>Just Relay Blocks]
    C --> D[Receive & Validate Blocks]
    D --> E[Forward Valid Blocks]
    E --> F[Next Slot]
    F --> A
    
    B -->|Yes| G{Am I Slot<br/>Leader?}
    
    G -->|Yes| H[Compute VRF<br/>slot, epoch, key]
    H --> I{VRF Output <<br/>Stake Threshold?}
    I -->|Yes| J[Create Block]
    I -->|No| K[Not Leader<br/>This Slot]
    
    J --> L[Collect Extrinsics<br/>from Transaction Pool]
    L --> M[Build Block<br/>Parent Hash<br/>Extrinsics<br/>VRF Proof]
    M --> N[Sign Block]
    N --> O[Broadcast Block]
    
    G -->|No| P[Wait for Block]
    K --> P
    
    P --> Q{Received Block<br/>in Time?}
    Q -->|Yes| R[Validate Block]
    Q -->|No| S[Empty Slot<br/>Continue]
    
    R --> T{Valid VRF Proof?<br/>Valid Signature?<br/>Valid Extrinsics?<br/>Valid State Transition?}
    T -->|Yes| U[Add Block to Chain<br/>Update State]
    T -->|No| V[Reject Block]
    
    O --> W[Block Propagated]
    W --> X[Other Validators Validate]
    
    U --> Y[GRANDPA: Finalization]
    S --> Y
    V --> Y
    
    Y --> Z{Am I in<br/>Finalization<br/>Committee?}
    Z -->|Yes| AA[Vote on Best Chain<br/>Based on GHOST Rule]
    Z -->|No| AB[Not in Committee<br/>Wait for Finalization]
    
    AA --> AC[Broadcast Vote<br/>with Stake Weight]
    AC --> AD[Collect Votes<br/>from Other Validators]
    AD --> AE{2/3+ Stake<br/>Voted for<br/>Same Block?}
    
    AE -->|Yes| AF[Finalize Block<br/>Lock Finalized Chain]
    AE -->|No| AG[Round R+1<br/>Continue Voting]
    AG --> AA
    
    AB --> AH[Wait for Finalization]
    AF --> AI[Blocks Finalized<br/>Cannot Revert]
    AH --> AI
    
    AI --> AJ[Next Slot S+1]
    AJ --> AK{End of Era?}
    AK -->|No| A
    AK -->|Yes| AL[Era Transition]
    
    AL --> AM[Phragmén Election<br/>Select New Validators<br/>Process Rewards<br/>Update Nominations]
    AM --> AN[New Era Starts]
    AN --> A
    
    style A fill:#FFE4B5
    style B fill:#FFD700
    style G fill:#FFD700
    style I fill:#FFD700
    style T fill:#FFD700
    style AE fill:#FFD700
    style U fill:#90EE90
    style AF fill:#90EE90
```

**Polkadot Consensus Components**:

**1. BABE (Blind Assignment for Blockchain Extension)**:
- **Block Production**: Creates blocks
- **VRF Selection**: Verifiable Random Function selects leaders
- **Slots**: Discrete time intervals
- **Multiple Leaders**: Can have multiple blocks per slot

**2. GRANDPA (GHOST-based Recursive Ancestor Deriving Prefix Agreement)**:
- **Finality Gadget**: Finalizes blocks
- **Not Block-by-Block**: Finalizes chains, not individual blocks
- **Fast Finality**: Can finalize many blocks at once
- **Safety**: Requires `2/3+` validator stake

**3. Validator Set**:
- **Validators**: Block producers and finalizers
- **Nominators**: Stake to validators
- **Election**: Phragmén algorithm selects validators
- **Rotation**: Validator set changes each era

**4. Era Structure**:
- **Era**: ~24 hours
- **Session**: ~1 hour
- **Epoch**: Multiple slots

**Key Properties**:
- **Hybrid**: BABE for liveness, GRANDPA for finality
- **Fast Blocks**: ~6 seconds
- **Fast Finality**: ~12-60 seconds
- **Shared Security**: All parachains share security

**Example**:
```rust
// BABE block production
fn produce_block(slot: Slot, parent: BlockHash) -> Option<Block> {
    // VRF to determine if selected
    let vrf_output = compute_vrf(slot, validator_key);
    let threshold = calculate_threshold(validator_stake, total_stake);
    
    if vrf_output < threshold {
        // Selected as leader
        let block = Block {
            parent_hash: parent,
            slot: slot,
            extrinsics: select_extrinsics(),
            state_root: compute_state_root(),
        };
        
        return Some(block);
    }
    
    None
}

// GRANDPA finalization
fn finalize_chain(
    validators: &[Validator],
    chain: &[Block]
) -> Option<BlockHash> {
    // Validators vote on chain
    let votes: Vec<Vote> = validators
        .iter()
        .map(|v| v.vote_on_chain(chain))
        .collect();
    
    // Check for 2/3+ majority
    let mut vote_counts: HashMap<BlockHash, u64> = HashMap::new();
    for vote in votes {
        *vote_counts.entry(vote.block_hash).or_insert(0) += vote.stake;
    }
    
    let total_stake: u64 = validators.iter().map(|v| v.stake).sum();
    let threshold = (total_stake * 2) / 3 + 1;
    
    // Find block with 2/3+ votes
    for (block_hash, votes) in vote_counts {
        if votes >= threshold {
            return Some(block_hash);
        }
    }
    
    None
}
```

**Phragmén Election**:
- Optimizes validator selection
- Maximizes minimum stake
- Ensures fair distribution

**Use Cases**:
- Polkadot
- Kusama
- Substrate-based chains

---

