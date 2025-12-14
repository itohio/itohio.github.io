---
title: "Tech Due Diligence Interview Questions - Medium"
date: 2025-12-13
tags: ["due-diligence", "interview", "medium", "deep-assessment"]
---

Medium-level technical due diligence interview questions covering deeper assessment methodologies.

## Q1: How do you conduct a comprehensive architecture review?

**Answer**:

```mermaid
graph TB
    A[Architecture<br/>Review] --> B[System Design]
    A --> C[Data Flow]
    A --> D[Integration Points]
    A --> E[Performance]
    A --> F[Scalability]
    
    style A fill:#FFD700
```

### Multi-Layer Assessment

```mermaid
graph TB
    A[Architecture] --> B1[Presentation<br/>Layer]
    B1 --> C1[Web/Mobile Apps<br/>API Gateway]
    
    A --> B2[Application<br/>Layer]
    B2 --> C2[Business Logic<br/>Services]
    
    A --> B3[Data<br/>Layer]
    B3 --> C3[Databases<br/>Caches]
    
    A --> B4[Infrastructure<br/>Layer]
    B4 --> C4[Servers<br/>Networks]
    
    style A fill:#FFD700
```

**Key Questions**:
- Is the architecture documented?
- Are concerns properly separated?
- How are services communicating?
- What are the failure points?
- How does data flow through the system?

### Architecture Patterns Assessment

```mermaid
graph LR
    A[Pattern Used] --> B{Appropriate<br/>for Scale?}
    
    B -->|Yes| C[✓ Good Fit]
    B -->|No| D[⚠ Risk]
    
    D --> E[Monolith serving<br/>1M+ users]
    D --> F[Microservices for<br/>simple CRUD]
    
    style C fill:#90EE90
    style E fill:#FF6B6B
    style F fill:#FF6B6B
```

---

## Q2: How do you assess data architecture and management?

**Answer**:

```mermaid
graph TB
    A[Data<br/>Architecture] --> B[Data Models]
    A --> C[Storage Strategy]
    A --> D[Data Pipeline]
    A --> E[Data Quality]
    A --> F[Compliance]
    
    style A fill:#FFD700
```

### Data Flow Analysis

```mermaid
graph LR
    A[Data Sources] --> B[Ingestion<br/>Layer]
    B --> C[Processing<br/>Layer]
    C --> D[Storage<br/>Layer]
    D --> E[Access<br/>Layer]
    E --> F[Consumers]
    
    style A fill:#FFE4B5
    style F fill:#90EE90
```

**Assessment Areas**:

```mermaid
graph TB
    A[Data Assessment] --> B1[Schema Design<br/>Normalized?<br/>Indexed?]
    A --> B2[Partitioning<br/>Sharding strategy?<br/>Distribution?]
    A --> B3[Replication<br/>Redundancy?<br/>Consistency?]
    A --> B4[Backup<br/>Frequency?<br/>Recovery time?]
    A --> B5[Access Patterns<br/>Read/Write ratio?<br/>Query performance?]
    
    style A fill:#FFD700
```

**Red Flags**:
- No data governance
- Inconsistent schemas
- No backup strategy
- PII not encrypted
- No data retention policy

---

## Q3: How do you evaluate API design and integration quality?

**Answer**:

```mermaid
graph TB
    A[API<br/>Evaluation] --> B[Design Quality]
    A --> C[Documentation]
    A --> D[Versioning]
    A --> E[Security]
    A --> F[Performance]
    
    style A fill:#FFD700
```

### API Design Assessment

```mermaid
graph TB
    A[API Design] --> B{RESTful<br/>Principles?}
    
    B --> C1[✓ Resource-based URLs]
    B --> C2[✓ HTTP methods correctly]
    B --> C3[✓ Status codes appropriate]
    B --> C4[✓ Consistent naming]
    B --> C5[✓ Pagination implemented]
    
    C1 --> D{Score}
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    
    D --> E1[Excellent: 5/5]
    D --> E2[Good: 3-4/5]
    D --> E3[Poor: <3/5]
    
    style E1 fill:#90EE90
    style E2 fill:#FFD700
    style E3 fill:#FF6B6B
```

### Integration Complexity

```mermaid
graph TB
    A[System] --> B1[External API 1]
    A --> B2[External API 2]
    A --> B3[External API 3]
    A --> B4[Database]
    A --> B5[Message Queue]
    
    B1 --> C{Integration<br/>Health}
    B2 --> C
    B3 --> C
    
    C --> D1[Error Handling?]
    C --> D2[Retry Logic?]
    C --> D3[Circuit Breaker?]
    C --> D4[Monitoring?]
    
    style A fill:#FFD700
```

**Questions**:
- How many external dependencies?
- What happens if one fails?
- Are there rate limits?
- How is authentication handled?
- What's the API versioning strategy?

---

## Q4: How do you assess testing strategy and quality assurance?

**Answer**:

```mermaid
graph TB
    A[Testing<br/>Strategy] --> B[Unit Tests]
    A --> C[Integration Tests]
    A --> D[E2E Tests]
    A --> E[Performance Tests]
    A --> F[Security Tests]
    
    style A fill:#FFD700
```

### Test Pyramid Assessment

```mermaid
graph TB
    A[E2E Tests<br/>Slow, Brittle<br/>10%] --> B[Integration Tests<br/>Medium Speed<br/>30%]
    B --> C[Unit Tests<br/>Fast, Reliable<br/>60%]
    
    style A fill:#FFB6C1
    style B fill:#FFD700
    style C fill:#90EE90
```

**Ideal distribution**: Many unit tests, fewer integration, minimal E2E

### Coverage Analysis

```mermaid
graph LR
    A[Code Coverage] --> B{Percentage}
    
    B --> C1[>80%<br/>Excellent]
    B --> C2[60-80%<br/>Good]
    B --> C3[40-60%<br/>Acceptable]
    B --> C4[<40%<br/>Poor]
    
    style C1 fill:#90EE90
    style C2 fill:#90EE90
    style C3 fill:#FFD700
    style C4 fill:#FF6B6B
```

**But**: Coverage % alone insufficient - check test quality!

**Assessment Questions**:
- What's the test coverage?
- How long do tests take to run?
- Are tests run in CI/CD?
- When was last test suite review?
- Are critical paths tested?

---

## Q5: How do you evaluate deployment and release processes?

**Answer**:

```mermaid
graph TB
    A[Deployment<br/>Process] --> B[Frequency]
    A --> C[Automation]
    A --> D[Rollback]
    A --> E[Blue-Green/Canary]
    A --> F[Monitoring]
    
    style A fill:#FFD700
```

### Deployment Maturity Model

```mermaid
graph TB
    A[Level 1<br/>Manual] --> B[Level 2<br/>Scripted]
    B --> C[Level 3<br/>Automated CI/CD]
    C --> D[Level 4<br/>Continuous Deployment]
    D --> E[Level 5<br/>Self-Healing]
    
    style A fill:#FF6B6B
    style B fill:#FFD700
    style C fill:#90EE90
    style D fill:#90EE90
    style E fill:#87CEEB
```

### Release Strategy

```mermaid
sequenceDiagram
    participant D as Developer
    participant CI as CI/CD
    participant S as Staging
    participant P as Production
    
    D->>CI: Push code
    CI->>CI: Run tests
    
    alt Tests Pass
        CI->>S: Deploy to staging
        S->>S: Smoke tests
        S->>P: Deploy canary (5%)
        P->>P: Monitor metrics
        
        alt Metrics Good
            P->>P: Roll out to 100%
        else Metrics Bad
            P->>P: Rollback
        end
    else Tests Fail
        CI->>D: Notify failure
    end
```

**Key Metrics**:
- Deployment frequency (daily? weekly?)
- Lead time (commit to production)
- Mean time to recovery (MTTR)
- Change failure rate

---

## Q6: How do you assess observability and monitoring?

**Answer**:

```mermaid
graph TB
    A[Observability] --> B[Metrics<br/>Numbers]
    A --> C[Logs<br/>Events]
    A --> D[Traces<br/>Requests]
    A --> E[Alerts<br/>Notifications]
    
    style A fill:#FFD700
```

### Three Pillars Assessment

```mermaid
graph TB
    A[Metrics] --> D[Dashboards]
    B[Logs] --> D
    C[Traces] --> D
    
    D --> E{Can Answer:}
    
    E --> F1[What is broken?]
    E --> F2[Why is it broken?]
    E --> F3[Where is it broken?]
    E --> F4[When did it break?]
    
    style D fill:#FFD700
    style F1 fill:#90EE90
    style F2 fill:#90EE90
    style F3 fill:#90EE90
    style F4 fill:#90EE90
```

**Assessment Questions**:
- What monitoring tools are used?
- What metrics are tracked?
- Are logs centralized?
- Is distributed tracing implemented?
- What's the alerting strategy?
- How long to detect issues?

### Alert Quality

```mermaid
graph TB
    A[Alert Fires] --> B{Actionable?}
    
    B -->|Yes| C{Urgent?}
    B -->|No| D[Noise<br/>Remove]
    
    C -->|Yes| E[Page On-Call]
    C -->|No| F[Ticket]
    
    style D fill:#FF6B6B
    style E fill:#FFD700
    style F fill:#90EE90
```

**Red Flags**:
- Alert fatigue (too many alerts)
- No runbooks
- Alerts without context
- No SLOs/SLIs defined

---

## Q7: How do you evaluate intellectual property and licensing?

**Answer**:

```mermaid
graph TB
    A[IP Assessment] --> B[Code Ownership]
    A --> C[Open Source<br/>Dependencies]
    A --> D[Third-Party<br/>Libraries]
    A --> E[Patents &<br/>Trademarks]
    
    style A fill:#FFD700
```

### License Risk Assessment

```mermaid
graph TB
    A[Dependencies] --> B{License Type}
    
    B --> C1[Permissive<br/>MIT, Apache, BSD]
    B --> C2[Weak Copyleft<br/>LGPL, MPL]
    B --> C3[Strong Copyleft<br/>GPL, AGPL]
    B --> C4[Proprietary<br/>Commercial]
    
    C1 --> D1[Low Risk]
    C2 --> D2[Medium Risk]
    C3 --> D3[High Risk]
    C4 --> D4[Review Terms]
    
    style D1 fill:#90EE90
    style D2 fill:#FFD700
    style D3 fill:#FF6B6B
    style D4 fill:#FFD700
```

**Key Questions**:
- Are all dependencies documented?
- Any GPL/AGPL dependencies?
- Are licenses compatible?
- Is there a license compliance process?
- Who owns the code? (employees? contractors?)
- Are IP assignments signed?

### Dependency Audit

```mermaid
sequenceDiagram
    participant S as Scan Codebase
    participant I as Identify Dependencies
    participant L as Check Licenses
    participant R as Risk Assessment
    
    S->>I: Extract all dependencies
    I->>L: For each dependency
    L->>L: Identify license
    L->>R: Assess compatibility
    
    alt High Risk Found
        R->>R: Flag for review
        R->>R: Plan remediation
    else Low Risk
        R->>R: Document and approve
    end
```

---

## Q8: How do you assess technical scalability limits?

**Answer**:

```mermaid
graph TB
    A[Scalability<br/>Assessment] --> B[Current Capacity]
    A --> C[Bottlenecks]
    A --> D[Growth Projections]
    A --> E[Cost Scaling]
    
    style A fill:#FFD700
```

### Capacity Planning

```mermaid
graph LR
    A[Current:<br/>10K users<br/>100 RPS] --> B[6 months:<br/>50K users<br/>500 RPS]
    B --> C[12 months:<br/>200K users<br/>2K RPS]
    C --> D{Can System<br/>Handle?}
    
    D -->|Yes| E[Scalable]
    D -->|No| F[Identify<br/>Constraints]
    
    style E fill:#90EE90
    style F fill:#FFD700
```

### Bottleneck Identification

```mermaid
graph TB
    A[System] --> B1[Database<br/>Queries]
    A --> B2[API<br/>Response Time]
    A --> B3[Memory<br/>Usage]
    A --> B4[CPU<br/>Usage]
    A --> B5[Network<br/>Bandwidth]
    
    B1 --> C{Load Test}
    B2 --> C
    B3 --> C
    B4 --> C
    B5 --> C
    
    C --> D[Identify<br/>Bottleneck]
    
    style D fill:#FFD700
```

**Assessment Approach**:
1. Profile current performance
2. Identify bottlenecks
3. Project growth
4. Calculate when limits hit
5. Estimate cost to scale

**Red Flags**:
- Already at capacity
- No horizontal scaling path
- Database can't shard
- Monolithic architecture at scale
- Exponential cost growth

---

## Q9: How do you evaluate disaster recovery and business continuity?

**Answer**:

```mermaid
graph TB
    A[DR/BC<br/>Assessment] --> B[Backup Strategy]
    A --> C[Recovery Plan]
    A --> D[Redundancy]
    A --> E[Testing]
    
    style A fill:#FFD700
```

### Key Metrics

```mermaid
graph TB
    A[DR Metrics] --> B[RPO<br/>Recovery Point<br/>Objective]
    A --> C[RTO<br/>Recovery Time<br/>Objective]
    
    B --> D[Max data loss<br/>acceptable]
    C --> E[Max downtime<br/>acceptable]
    
    D --> F{Current<br/>vs Target}
    E --> F
    
    F -->|Match| G[✓ Adequate]
    F -->|Gap| H[⚠ Risk]
    
    style G fill:#90EE90
    style H fill:#FF6B6B
```

### Disaster Scenarios

```mermaid
graph TB
    A[Disaster<br/>Scenarios] --> B1[Data Center<br/>Failure]
    A --> B2[Database<br/>Corruption]
    A --> B3[Ransomware<br/>Attack]
    A --> B4[Key Person<br/>Loss]
    A --> B5[DDoS<br/>Attack]
    
    B1 --> C{Plan<br/>Exists?}
    B2 --> C
    B3 --> C
    B4 --> C
    B5 --> C
    
    C -->|Yes| D[✓ Prepared]
    C -->|No| E[⚠ Vulnerable]
    
    style D fill:#90EE90
    style E fill:#FF6B6B
```

**Assessment Questions**:
- What's the backup frequency?
- Where are backups stored?
- How long to restore?
- When was last DR test?
- Is there geographic redundancy?
- What's the failover process?

---

## Q10: How do you assess technical team processes and culture?

**Answer**:

```mermaid
graph TB
    A[Team<br/>Processes] --> B[Development<br/>Workflow]
    A --> C[Code Review<br/>Process]
    A --> D[Knowledge<br/>Sharing]
    A --> E[Incident<br/>Management]
    
    style A fill:#FFD700
```

### Development Workflow Maturity

```mermaid
graph TB
    A[Feature Request] --> B[Design Review]
    B --> C[Implementation]
    C --> D[Code Review]
    D --> E[Testing]
    E --> F[Deployment]
    F --> G[Monitoring]
    
    G --> H{Issues?}
    H -->|Yes| I[Incident Response]
    H -->|No| J[Success]
    
    I --> K[Post-Mortem]
    K --> L[Improvements]
    
    style A fill:#FFE4B5
    style J fill:#90EE90
```

### Code Review Quality

```mermaid
graph LR
    A[PR Submitted] --> B{Review<br/>Process}
    
    B --> C1[✓ Timely<br/><24hrs]
    B --> C2[✓ Thorough<br/>Multiple reviewers]
    B --> C3[✓ Constructive<br/>Feedback]
    B --> C4[✓ Standards<br/>Enforced]
    
    C1 --> D{Quality<br/>Score}
    C2 --> D
    C3 --> D
    C4 --> D
    
    D --> E[High/Medium/Low]
    
    style E fill:#90EE90
```

**Assessment Areas**:
- Sprint planning effectiveness
- Estimation accuracy
- Velocity trends
- Bug escape rate
- Time to resolution
- Knowledge silos
- Documentation culture
- On-call rotation

**Red Flags**:
- No code review
- Cowboy coding
- Knowledge in one person
- No post-mortems
- Blame culture
- No process documentation

---

## Summary

Medium tech due diligence topics:
- **Architecture Review**: Multi-layer assessment, patterns
- **Data Architecture**: Models, pipelines, quality
- **API Quality**: Design, documentation, integrations
- **Testing Strategy**: Pyramid, coverage, automation
- **Deployment**: Maturity model, release strategy
- **Observability**: Metrics, logs, traces, alerts
- **IP & Licensing**: Ownership, dependencies, compliance
- **Scalability Limits**: Capacity planning, bottlenecks
- **Disaster Recovery**: RPO/RTO, redundancy, testing
- **Team Processes**: Workflow, reviews, culture

These deeper assessments reveal technical maturity and risks.

