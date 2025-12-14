---
title: "Architecture Interview Questions - Hard"
date: 2025-12-13
tags: ["architecture", "interview", "hard", "distributed-systems", "consensus"]
---

Hard-level software architecture interview questions covering advanced distributed systems, consensus, and complex patterns.

## Q1: Explain distributed consensus algorithms (Raft, Paxos).

**Answer**:

**Problem**: How do multiple nodes agree on a value in presence of failures?

```mermaid
graph TB
    subgraph Raft["Raft Consensus"]
        L[Leader] --> F1[Follower 1]
        L --> F2[Follower 2]
        L --> F3[Follower 3]
        
        F1 -.->|Vote| L
        F2 -.->|Vote| L
        F3 -.->|Vote| L
        
        L -->|Log Replication| F1
        L -->|Log Replication| F2
        L -->|Log Replication| F3
    end
    
    style L fill:#FFD700
    style F1 fill:#87CEEB
    style F2 fill:#87CEEB
    style F3 fill:#87CEEB
```

### Raft Algorithm

**Roles**:
- **Leader**: Handles all client requests, replicates log
- **Follower**: Passive, responds to leader/candidate
- **Candidate**: Seeks votes to become leader

**Leader Election**:

```mermaid
sequenceDiagram
    participant F1 as Follower 1
    participant F2 as Follower 2
    participant F3 as Follower 3
    
    Note over F1,F3: Leader timeout
    F1->>F1: Become Candidate
    F1->>F2: RequestVote
    F1->>F3: RequestVote
    F2->>F1: Vote Granted
    F3->>F1: Vote Granted
    F1->>F1: Become Leader
    F1->>F2: Heartbeat
    F1->>F3: Heartbeat
```

**Log Replication**:
1. Client sends command to leader
2. Leader appends to local log
3. Leader replicates to followers
4. Once majority acknowledges, leader commits
5. Leader notifies followers to commit

**Safety Properties**:
- **Election Safety**: At most one leader per term
- **Leader Append-Only**: Leader never overwrites log
- **Log Matching**: If two logs contain same entry, all preceding entries identical
- **Leader Completeness**: If entry committed, present in all future leaders
- **State Machine Safety**: If server applies log entry, no other server applies different entry at that index

### Paxos Algorithm

**Phases**:

```mermaid
graph LR
    A[Proposer] -->|Phase 1a: Prepare| B[Acceptors]
    B -->|Phase 1b: Promise| A
    A -->|Phase 2a: Accept| B
    B -->|Phase 2b: Accepted| C[Learners]
    
    style A fill:#FFD700
    style B fill:#87CEEB
    style C fill:#90EE90
```

**Phase 1 (Prepare)**:
- Proposer selects proposal number n
- Sends Prepare(n) to majority of acceptors
- Acceptors promise not to accept proposals < n

**Phase 2 (Accept)**:
- If majority promises, proposer sends Accept(n, value)
- Acceptors accept if haven't promised higher number
- Once majority accepts, value is chosen

**Comparison**:

| Aspect | Raft | Paxos |
|--------|------|-------|
| Understandability | Easier | Complex |
| Leader | Strong leader | No fixed leader |
| Log Structure | Strongly consistent | More flexible |
| Implementation | Simpler | More variants |

**Use Cases**:
- **Raft**: etcd, Consul, CockroachDB
- **Paxos**: Google Chubby, Apache ZooKeeper (ZAB variant)

---

## Q2: Design a globally distributed system with multi-region consistency.

**Answer**:

```mermaid
graph TB
    subgraph US["US Region"]
        US_LB[Load Balancer] --> US_APP1[App Server]
        US_LB --> US_APP2[App Server]
        US_APP1 --> US_DB[(Primary DB)]
        US_APP2 --> US_DB
        US_DB --> US_CACHE[Redis Cache]
    end
    
    subgraph EU["EU Region"]
        EU_LB[Load Balancer] --> EU_APP1[App Server]
        EU_LB --> EU_APP2[App Server]
        EU_APP1 --> EU_DB[(Primary DB)]
        EU_APP2 --> EU_DB
        EU_DB --> EU_CACHE[Redis Cache]
    end
    
    subgraph ASIA["Asia Region"]
        ASIA_LB[Load Balancer] --> ASIA_APP1[App Server]
        ASIA_LB --> ASIA_APP2[App Server]
        ASIA_APP1 --> ASIA_DB[(Primary DB)]
        ASIA_APP2 --> ASIA_DB
        ASIA_DB --> ASIA_CACHE[Redis Cache]
    end
    
    US_DB <-.->|Async Replication| EU_DB
    EU_DB <-.->|Async Replication| ASIA_DB
    ASIA_DB <-.->|Async Replication| US_DB
    
    CDN[Global CDN] --> US
    CDN --> EU
    CDN --> ASIA
    
    style CDN fill:#FFD700
```

### Key Challenges

**1. Data Consistency**:

```mermaid
graph LR
    A[Strong Consistency] -->|Slow| B[Synchronous<br/>Replication]
    C[Eventual Consistency] -->|Fast| D[Asynchronous<br/>Replication]
    E[Causal Consistency] -->|Balanced| F[Vector Clocks/<br/>CRDTs]
    
    style A fill:#FF6B6B
    style C fill:#90EE90
    style E fill:#FFD700
```

**Strategies**:
- **Strong Consistency**: Synchronous replication (slow, high latency)
- **Eventual Consistency**: Async replication (fast, temporary inconsistency)
- **Causal Consistency**: Preserve causality, allow concurrent updates

**2. Conflict Resolution**:

```mermaid
graph TB
    A[Concurrent Updates] --> B{Resolution Strategy}
    B -->|Last Write Wins| C[Timestamp-based]
    B -->|Application Logic| D[Custom Merge]
    B -->|CRDTs| E[Conflict-Free<br/>Data Types]
    B -->|Manual| F[Present to User]
    
    style A fill:#FFB6C1
    style E fill:#90EE90
```

**3. Latency Optimization**:
- **Read-Local**: Serve reads from nearest region
- **Write-Local**: Accept writes locally, replicate async
- **CDN**: Cache static content globally
- **Edge Computing**: Process at edge locations

**4. Failure Handling**:
- **Circuit Breakers**: Prevent cascade failures
- **Fallback**: Serve stale data if region unavailable
- **Health Checks**: Monitor region health
- **Automatic Failover**: Route traffic to healthy regions

### Implementation Patterns

**Multi-Master Replication**:
```mermaid
graph LR
    US[(US Master)] <-->|Bidirectional<br/>Replication| EU[(EU Master)]
    EU <-->|Bidirectional<br/>Replication| ASIA[(Asia Master)]
    ASIA <-->|Bidirectional<br/>Replication| US
    
    style US fill:#87CEEB
    style EU fill:#87CEEB
    style ASIA fill:#87CEEB
```

**CRDT (Conflict-Free Replicated Data Types)**:
- Guaranteed convergence without coordination
- Types: G-Counter, PN-Counter, LWW-Register, OR-Set
- Use: Collaborative editing, distributed counters

**Vector Clocks**:
- Track causality across replicas
- Detect concurrent updates
- Enable causal consistency

---

## Q3: Explain event sourcing and CQRS at scale.

**Answer**:

```mermaid
graph TB
    subgraph Write["Write Side Event Sourcing"]
        CMD[Command] --> AGG[Aggregate]
        AGG --> EVT[Event]
        EVT --> ES[(Event Store)]
        ES --> EB[Event Bus]
    end
    
    subgraph Read["Read Side CQRS"]
        EB --> P1[Projection 1<br/>User View]
        EB --> P2[Projection 2<br/>Analytics]
        EB --> P3[Projection 3<br/>Search Index]
        
        P1 --> RDB1[(Read DB 1)]
        P2 --> RDB2[(Read DB 2)]
        P3 --> RDB3[(Search)]
    end
    
    Q[Query] --> RDB1
    Q --> RDB2
    Q --> RDB3
    
    style CMD fill:#FFD700
    style ES fill:#87CEEB
    style EB fill:#DDA0DD
```

### Event Sourcing

**Core Concept**: Store all changes as sequence of events, not current state.

**Event Store Structure**:
```mermaid
graph LR
    A[Aggregate ID:<br/>Order-123] --> B[Event 1:<br/>OrderCreated]
    B --> C[Event 2:<br/>ItemAdded]
    C --> D[Event 3:<br/>PaymentProcessed]
    D --> E[Event 4:<br/>OrderShipped]
    
    style A fill:#FFE4B5
    style B fill:#87CEEB
    style C fill:#87CEEB
    style D fill:#87CEEB
    style E fill:#87CEEB
```

**Benefits**:
- Complete audit trail
- Time travel (reconstruct past states)
- Event replay for debugging
- Multiple projections from same events

**Challenges at Scale**:

**1. Event Store Growth**:
```mermaid
graph TB
    A[Millions of Events] --> B{Solution}
    B --> C[Snapshots]
    B --> D[Archiving]
    B --> E[Compaction]
    
    C --> F[Store State<br/>at Intervals]
    D --> G[Move Old Events<br/>to Cold Storage]
    E --> H[Merge Events<br/>for Same Aggregate]
    
    style A fill:#FF6B6B
    style F fill:#90EE90
    style G fill:#90EE90
    style H fill:#90EE90
```

**Snapshots**:
- Periodically save aggregate state
- Replay only events after snapshot
- Reduces reconstruction time

**2. Projection Lag**:
```mermaid
sequenceDiagram
    participant W as Write Side
    participant ES as Event Store
    participant P as Projection
    participant R as Read DB
    
    W->>ES: Save Event (t=0)
    ES->>P: Notify (t=1ms)
    P->>P: Process (t=10ms)
    P->>R: Update (t=15ms)
    
    Note over W,R: 15ms lag
```

**Solutions**:
- Accept eventual consistency
- Show "processing" state to users
- Use optimistic UI updates
- Prioritize critical projections

**3. Event Versioning**:
```mermaid
graph LR
    A[Event V1] --> B{Schema Change}
    B --> C[Upcasting]
    B --> D[Multiple Versions]
    B --> E[Event Migration]
    
    C --> F[Convert V1→V2<br/>on Read]
    D --> G[Handle Both<br/>Versions]
    E --> H[Rewrite Events<br/>to V2]
    
    style A fill:#FFE4B5
    style F fill:#90EE90
    style G fill:#FFD700
    style H fill:#87CEEB
```

### CQRS at Scale

**Read Model Optimization**:
- Denormalized for query performance
- Multiple read models for different use cases
- Can use different databases (SQL, NoSQL, Search)

**Scaling Reads**:
```mermaid
graph TB
    EB[Event Bus] --> P[Projection Service]
    P --> M1[Read Model 1<br/>PostgreSQL]
    P --> M2[Read Model 2<br/>Elasticsearch]
    P --> M3[Read Model 3<br/>Redis Cache]
    
    M1 --> R1[Read Replica 1]
    M1 --> R2[Read Replica 2]
    M1 --> R3[Read Replica 3]
    
    LB[Load Balancer] --> R1
    LB --> R2
    LB --> R3
    
    style EB fill:#DDA0DD
    style LB fill:#FFD700
```

**Scaling Writes**:
- Partition event store by aggregate ID
- Shard across multiple nodes
- Use distributed event bus (Kafka, Pulsar)

---

## Q4: Design a real-time collaborative editing system (like Google Docs).

**Answer**:

```mermaid
graph TB
    subgraph Clients["Multiple Clients"]
        C1[User 1<br/>Browser]
        C2[User 2<br/>Browser]
        C3[User 3<br/>Browser]
    end
    
    subgraph Backend["Backend Services"]
        WS[WebSocket<br/>Server]
        OT[Operational<br/>Transform]
        CRDT[CRDT Engine]
        SYNC[Sync Service]
    end
    
    subgraph Storage["Storage Layer"]
        MEM[(In-Memory<br/>State)]
        DB[(Persistent<br/>Storage)]
        CACHE[Redis Cache]
    end
    
    C1 <-->|WebSocket| WS
    C2 <-->|WebSocket| WS
    C3 <-->|WebSocket| WS
    
    WS --> OT
    WS --> CRDT
    OT --> SYNC
    CRDT --> SYNC
    
    SYNC --> MEM
    SYNC --> CACHE
    MEM -.->|Periodic Save| DB
    
    style WS fill:#FFD700
    style OT fill:#87CEEB
    style CRDT fill:#90EE90
```

### Key Challenges

**1. Concurrent Edits**:

```mermaid
sequenceDiagram
    participant U1 as User 1
    participant S as Server
    participant U2 as User 2
    
    Note over U1,U2: Initial: "Hello"
    
    par Concurrent Edits
        U1->>U1: Insert "!" at pos 5
        U2->>U2: Insert " World" at pos 5
    end
    
    U1->>S: Op1: Insert "!" at 5
    U2->>S: Op2: Insert " World" at 5
    
    S->>S: Transform Op2 against Op1
    S->>U1: Op2': Insert " World" at 5
    S->>U2: Op1': Insert "!" at 11
    
    Note over U1,U2: Final: "Hello World!"
```

**Solutions**:

**Operational Transformation (OT)**:
- Transform operations based on concurrent ops
- Maintains convergence and intention
- Complex to implement correctly

**CRDTs (Conflict-Free Replicated Data Types)**:
- Mathematically guaranteed convergence
- No central coordination needed
- Simpler than OT

**2. Real-Time Synchronization**:

```mermaid
graph LR
    A[Local Edit] --> B[Generate Op]
    B --> C[Apply Locally]
    B --> D[Send to Server]
    D --> E[Broadcast to Others]
    E --> F[Apply Remotely]
    
    style A fill:#FFE4B5
    style C fill:#90EE90
    style F fill:#87CEEB
```

**Optimizations**:
- **Optimistic Updates**: Apply locally immediately
- **Batching**: Group operations to reduce network calls
- **Compression**: Compress operation payloads
- **Presence**: Show who's editing what

**3. Scalability**:

```mermaid
graph TB
    LB[Load Balancer] --> WS1[WebSocket<br/>Server 1]
    LB --> WS2[WebSocket<br/>Server 2]
    LB --> WS3[WebSocket<br/>Server 3]
    
    WS1 --> PUB1[Pub/Sub]
    WS2 --> PUB1
    WS3 --> PUB1
    
    PUB1 --> WS1
    PUB1 --> WS2
    PUB1 --> WS3
    
    WS1 --> CACHE[(Redis<br/>Document State)]
    WS2 --> CACHE
    WS3 --> CACHE
    
    style LB fill:#FFD700
    style PUB1 fill:#DDA0DD
    style CACHE fill:#87CEEB
```

**Strategies**:
- **Sticky Sessions**: Route user to same server
- **Pub/Sub**: Broadcast operations across servers
- **Shared State**: Use Redis for document state
- **Sharding**: Partition documents across servers

**4. Persistence**:
- **Periodic Snapshots**: Save full document periodically
- **Operation Log**: Store all operations
- **Hybrid**: Snapshot + operations since snapshot

### Implementation Considerations

**Conflict Resolution**:
- Last Write Wins (LWW)
- Version Vectors
- Application-specific logic

**Offline Support**:
- Queue operations while offline
- Sync when reconnected
- Handle conflicts on reconnection

**Performance**:
- Sub-100ms latency for operations
- Support 100+ concurrent editors per document
- Handle documents up to 10MB

---

## Q5: Explain chaos engineering and how to implement it.

**Answer**:

```mermaid
graph TB
    A[Define Steady State] --> B[Hypothesize<br/>Normal Behavior]
    B --> C[Introduce<br/>Chaos]
    C --> D[Monitor<br/>System]
    D --> E{System<br/>Resilient?}
    E -->|No| F[Fix Issues]
    E -->|Yes| G[Increase<br/>Blast Radius]
    F --> A
    G --> C
    
    style A fill:#87CEEB
    style C fill:#FF6B6B
    style E fill:#FFD700
    style G fill:#90EE90
```

### Chaos Experiments

**Types of Failures to Inject**:

```mermaid
graph LR
    A[Chaos<br/>Engineering] --> B[Network]
    A --> C[Infrastructure]
    A --> D[Application]
    A --> E[State]
    
    B --> B1[Latency]
    B --> B2[Packet Loss]
    B --> B3[Partition]
    
    C --> C1[Server Crash]
    C --> C2[Disk Full]
    C --> C3[CPU Spike]
    
    D --> D1[Service Down]
    D --> D2[Slow Response]
    D --> D3[Error Injection]
    
    E --> E1[Data Corruption]
    E --> E2[Clock Skew]
    E --> E3[Resource Exhaustion]
    
    style A fill:#FFD700
    style B fill:#FF6B6B
    style C fill:#FF6B6B
    style D fill:#FF6B6B
    style E fill:#FF6B6B
```

### Implementation Levels

**1. Development**:
- Unit tests with mocked failures
- Integration tests with fault injection
- Local chaos testing

**2. Staging**:
- Automated chaos experiments
- Full system tests
- Performance under failure

**3. Production**:
- Controlled experiments
- Gradual rollout
- Automated rollback

### Chaos Tools

```mermaid
graph TB
    subgraph Tools["Chaos Engineering Tools"]
        A[Chaos Monkey] --> A1[Random Instance<br/>Termination]
        B[Chaos Kong] --> B1[Region Failure]
        C[Latency Monkey] --> C1[Network Delays]
        D[Pumba] --> D1[Docker Container<br/>Chaos]
        E[Gremlin] --> E1[Comprehensive<br/>Platform]
        F[Litmus] --> F1[Kubernetes<br/>Chaos]
    end
    
    style A fill:#FF6B6B
    style B fill:#FF6B6B
    style C fill:#FF6B6B
    style D fill:#FF6B6B
    style E fill:#FFD700
    style F fill:#87CEEB
```

### Best Practices

**Start Small**:
```mermaid
graph LR
    A[Dev Environment] --> B[Single Service]
    B --> C[Staging]
    C --> D[Production<br/>1% Traffic]
    D --> E[Production<br/>Full Traffic]
    
    style A fill:#90EE90
    style E fill:#FF6B6B
```

**Observability**:
- Comprehensive monitoring
- Distributed tracing
- Log aggregation
- Real-time alerting

**Safety Measures**:
- **Blast Radius**: Limit scope of experiments
- **Abort Conditions**: Auto-stop if critical metrics degrade
- **Business Hours**: Run during staffed hours initially
- **Gradual Rollout**: Increase scope over time

### Example Scenarios

**Network Partition**:
- Simulate split-brain scenario
- Verify consensus algorithm works
- Check data consistency

**Service Degradation**:
- Slow down database
- Verify timeouts and retries
- Check circuit breakers activate

**Resource Exhaustion**:
- Fill disk space
- Exhaust memory
- Max out CPU
- Verify graceful degradation

### Measuring Success

**Metrics**:
- **MTTR** (Mean Time To Recovery): How fast system recovers
- **Availability**: Percentage uptime during chaos
- **Error Rate**: Increase in errors
- **Latency**: Impact on response times

**Goals**:
- No customer-facing impact
- Automatic recovery
- Graceful degradation
- Clear alerts and runbooks

---

## Summary

Hard architecture topics:
- **Distributed Consensus**: Raft, Paxos for agreement
- **Global Distribution**: Multi-region consistency strategies
- **Event Sourcing + CQRS**: Scalable event-driven systems
- **Collaborative Editing**: OT, CRDTs for real-time sync
- **Chaos Engineering**: Testing resilience through failure injection

These patterns enable building highly available, scalable, and resilient distributed systems.

