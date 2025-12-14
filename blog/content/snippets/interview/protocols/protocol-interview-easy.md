---
title: "Protocol & Design Interview Questions - Easy"
date: 2025-12-13
tags: ["protocol", "interview", "easy", "networking", "design"]
---

Easy-level protocol and design interview questions covering fundamental concepts.

## Q1: What is a network protocol and why is it important?

**Answer**:

**Definition**: A set of rules and conventions for communication between network entities.

```mermaid
graph TB
    A[Protocol] --> B[Defines Format<br/>Message structure]
    A --> C[Defines Order<br/>Message sequence]
    A --> D[Defines Actions<br/>On send/receive]
    
    B --> E[Enables<br/>Communication]
    C --> E
    D --> E
    
    style A fill:#FFD700
    style E fill:#90EE90
```

**Why Important**:
- Standardization: Different systems can communicate
- Interoperability: Works across vendors
- Reliability: Error handling, retransmission
- Security: Authentication, encryption

**Examples**:
- **HTTP**: Web communication
- **TCP**: Reliable data transfer
- **DNS**: Name resolution
- **SMTP**: Email delivery

---

## Q2: Explain the OSI model layers.

**Answer**:

```mermaid
graph TB
    A[7. Application<br/>HTTP, FTP, SMTP] --> B[6. Presentation<br/>SSL/TLS, Encryption]
    B --> C[5. Session<br/>Session management]
    C --> D[4. Transport<br/>TCP, UDP]
    D --> E[3. Network<br/>IP, Routing]
    E --> F[2. Data Link<br/>Ethernet, MAC]
    F --> G[1. Physical<br/>Cables, Signals]
    
    style A fill:#FFE4B5
    style D fill:#87CEEB
    style E fill:#90EE90
    style G fill:#FFB6C1
```

### Layer Responsibilities

```mermaid
graph LR
    A[Application] --> B[User Interface<br/>APIs]
    C[Transport] --> D[End-to-End<br/>Delivery]
    E[Network] --> F[Routing<br/>Addressing]
    G[Physical] --> H[Bits on Wire<br/>Signals]
    
    style A fill:#FFE4B5
    style C fill:#87CEEB
    style E fill:#90EE90
    style G fill:#FFB6C1
```

**Mnemonic**: "All People Seem To Need Data Processing"

---

## Q3: What's the difference between TCP and UDP?

**Answer**:

```mermaid
graph TB
    A[Transport<br/>Protocols] --> B[TCP<br/>Transmission Control<br/>Protocol]
    A --> C[UDP<br/>User Datagram<br/>Protocol]
    
    B --> D1[Connection-oriented]
    B --> D2[Reliable]
    B --> D3[Ordered]
    B --> D4[Slower]
    
    C --> E1[Connectionless]
    C --> E2[Unreliable]
    C --> E3[Unordered]
    C --> E4[Faster]
    
    style B fill:#87CEEB
    style C fill:#FFD700
```

### TCP Three-Way Handshake

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    
    C->>S: SYN (seq=x)
    S->>C: SYN-ACK (seq=y, ack=x+1)
    C->>S: ACK (ack=y+1)
    
    Note over C,S: Connection Established
```

### When to Use Each

```mermaid
graph TB
    A{Use Case} --> B[Need Reliability?]
    
    B -->|Yes| C[TCP]
    B -->|No| D[Need Speed?]
    
    D -->|Yes| E[UDP]
    D -->|No| F[Still TCP]
    
    C --> G1[Web browsing<br/>File transfer<br/>Email]
    E --> G2[Video streaming<br/>Gaming<br/>VoIP]
    
    style C fill:#87CEEB
    style E fill:#FFD700
```

---

## Q4: Explain how DNS works.

**Answer**:

```mermaid
graph TB
    A[User types<br/>www.example.com] --> B[Browser checks<br/>cache]
    
    B --> C{In cache?}
    
    C -->|Yes| D[Use cached IP]
    C -->|No| E[Query DNS<br/>Resolver]
    
    E --> F[Recursive<br/>Lookup]
    
    F --> G[Root Server]
    G --> H[TLD Server<br/>.com]
    H --> I[Authoritative<br/>Server]
    
    I --> J[Return IP<br/>93.184.216.34]
    
    J --> K[Browser connects<br/>to IP]
    
    style A fill:#FFE4B5
    style J fill:#90EE90
    style K fill:#90EE90
```

### DNS Hierarchy

```mermaid
graph TB
    A[Root<br/>.] --> B1[.com]
    A --> B2[.org]
    A --> B3[.net]
    
    B1 --> C1[example.com]
    B1 --> C2[google.com]
    
    C1 --> D1[www.example.com]
    C1 --> D2[mail.example.com]
    
    style A fill:#FFD700
    style B1 fill:#87CEEB
    style C1 fill:#90EE90
```

**DNS Record Types**:
- **A**: IPv4 address
- **AAAA**: IPv6 address
- **CNAME**: Alias to another name
- **MX**: Mail server
- **TXT**: Text data

---

## Q5: What is HTTP and how does it work?

**Answer**:

```mermaid
graph LR
    A[Client] -->|HTTP Request| B[Server]
    B -->|HTTP Response| A
    
    style A fill:#FFE4B5
    style B fill:#90EE90
```

### HTTP Request

```mermaid
graph TB
    A[HTTP Request] --> B[Method<br/>GET, POST, PUT, DELETE]
    A --> C[URL<br/>/api/users/123]
    A --> D[Headers<br/>Content-Type, Auth]
    A --> E[Body<br/>Data optional]
    
    style A fill:#FFD700
```

### HTTP Response

```mermaid
graph TB
    A[HTTP Response] --> B[Status Code<br/>200, 404, 500]
    A --> C[Headers<br/>Content-Type, etc.]
    A --> D[Body<br/>HTML, JSON, etc.]
    
    B --> E1[2xx: Success]
    B --> E2[3xx: Redirect]
    B --> E3[4xx: Client Error]
    B --> E4[5xx: Server Error]
    
    style A fill:#FFD700
    style E1 fill:#90EE90
    style E3 fill:#FFD700
    style E4 fill:#FF6B6B
```

**Common Status Codes**:
- **200 OK**: Success
- **201 Created**: Resource created
- **400 Bad Request**: Invalid request
- **401 Unauthorized**: Authentication required
- **404 Not Found**: Resource doesn't exist
- **500 Internal Server Error**: Server error

---

## Q6: Explain REST API principles.

**Answer**:

```mermaid
graph TB
    A[REST<br/>Principles] --> B[Stateless<br/>No session state]
    A --> C[Resource-Based<br/>URLs as nouns]
    A --> D[HTTP Methods<br/>CRUD operations]
    A --> E[Representations<br/>JSON, XML]
    
    style A fill:#FFD700
```

### RESTful URL Design

```mermaid
graph TB
    A[Resources] --> B[Collection<br/>/users]
    A --> C[Single Item<br/>/users/123]
    
    B --> D1[GET /users<br/>List all]
    B --> D2[POST /users<br/>Create new]
    
    C --> E1[GET /users/123<br/>Get one]
    C --> E2[PUT /users/123<br/>Update]
    C --> E3[DELETE /users/123<br/>Delete]
    
    style A fill:#FFD700
    style D1 fill:#90EE90
    style D2 fill:#87CEEB
    style E1 fill:#90EE90
    style E2 fill:#FFD700
    style E3 fill:#FF6B6B
```

### HTTP Methods

```mermaid
graph LR
    A[CRUD] --> B[Create → POST]
    A --> C[Read → GET]
    A --> D[Update → PUT/PATCH]
    A --> E[Delete → DELETE]
    
    style A fill:#FFD700
```

**Best Practices**:
- Use nouns, not verbs in URLs
- Use plural for collections
- Use HTTP status codes correctly
- Version your API
- Use pagination for large collections

---

## Q7: What is WebSocket and when to use it?

**Answer**:

```mermaid
graph TB
    A[WebSocket] --> B[Full-Duplex<br/>Two-way communication]
    A --> C[Persistent<br/>Connection stays open]
    A --> D[Low Latency<br/>Real-time]
    
    style A fill:#FFD700
```

### HTTP vs WebSocket

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    
    Note over C,S: HTTP (Request-Response)
    C->>S: Request
    S->>C: Response
    Note over C,S: Connection closed
    
    C->>S: Request
    S->>C: Response
    Note over C,S: New connection
    
    Note over C,S: WebSocket (Persistent)
    C->>S: Upgrade to WebSocket
    S->>C: Upgrade accepted
    
    Note over C,S: Connection stays open
    C->>S: Message
    S->>C: Message
    C->>S: Message
    S->>C: Message
```

### Use Cases

```mermaid
graph TB
    A[WebSocket<br/>Use Cases] --> B1[Chat Applications<br/>Real-time messages]
    A --> B2[Live Updates<br/>Stock prices, sports]
    A --> B3[Collaborative Editing<br/>Google Docs]
    A --> B4[Gaming<br/>Multiplayer]
    A --> B5[IoT<br/>Sensor data]
    
    style A fill:#FFD700
```

**When NOT to use**:
- Simple request-response
- Infrequent updates
- One-way communication (use SSE)

---

## Q8: Explain load balancing basics.

**Answer**:

```mermaid
graph TB
    A[Clients] --> B[Load Balancer]
    
    B --> C1[Server 1]
    B --> C2[Server 2]
    B --> C3[Server 3]
    
    style B fill:#FFD700
    style C1 fill:#90EE90
    style C2 fill:#90EE90
    style C3 fill:#90EE90
```

### Load Balancing Algorithms

```mermaid
graph TB
    A[Algorithms] --> B1[Round Robin<br/>Rotate through servers]
    A --> B2[Least Connections<br/>Fewest active connections]
    A --> B3[IP Hash<br/>Same client → same server]
    A --> B4[Weighted<br/>Based on capacity]
    
    style A fill:#FFD700
```

### Round Robin Example

```mermaid
sequenceDiagram
    participant C1 as Client 1
    participant C2 as Client 2
    participant C3 as Client 3
    participant LB as Load Balancer
    participant S1 as Server 1
    participant S2 as Server 2
    participant S3 as Server 3
    
    C1->>LB: Request
    LB->>S1: Forward
    
    C2->>LB: Request
    LB->>S2: Forward
    
    C3->>LB: Request
    LB->>S3: Forward
    
    Note over LB: Next request goes to S1
```

**Benefits**:
- High availability
- Scalability
- No single point of failure
- Better performance

---

## Q9: What is API versioning and why is it important?

**Answer**:

```mermaid
graph TB
    A[API Versioning] --> B[Breaking Changes<br/>Incompatible updates]
    A --> C[Backward Compatibility<br/>Support old clients]
    A --> D[Gradual Migration<br/>Transition period]
    
    style A fill:#FFD700
```

### Versioning Strategies

```mermaid
graph TB
    A[Versioning<br/>Methods] --> B1[URL Path<br/>/v1/users]
    A --> B2[Query Parameter<br/>/users?version=1]
    A --> B3[Header<br/>Accept: application/vnd.api+json;version=1]
    A --> B4[Subdomain<br/>v1.api.example.com]
    
    B1 --> C1[✓ Most Common<br/>✓ Clear<br/>✗ URL changes]
    B2 --> C2[✓ Same URL<br/>✗ Easy to forget]
    B3 --> C3[✓ Clean URLs<br/>✗ Less visible]
    B4 --> C4[✓ Separate infrastructure<br/>✗ Complex]
    
    style B1 fill:#90EE90
```

### Version Lifecycle

```mermaid
graph LR
    A[v1 Released] --> B[v2 Released<br/>v1 Deprecated]
    B --> C[v3 Released<br/>v1 Sunset<br/>v2 Deprecated]
    C --> D[v2 Sunset]
    
    style A fill:#90EE90
    style B fill:#FFD700
    style C fill:#FFD700
    style D fill:#FF6B6B
```

**Best Practices**:
- Version from day one
- Document breaking changes
- Give deprecation notice (6-12 months)
- Support at least 2 versions
- Use semantic versioning

---

## Q10: Explain basic authentication methods.

**Answer**:

```mermaid
graph TB
    A[Authentication<br/>Methods] --> B[Basic Auth]
    A --> C[API Keys]
    A --> D[OAuth 2.0]
    A --> E[JWT]
    
    style A fill:#FFD700
```

### Basic Authentication

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    
    C->>S: GET /api/users
    S->>C: 401 Unauthorized
    
    C->>S: GET /api/users<br/>Authorization: Basic base64(user:pass)
    S->>S: Verify credentials
    S->>C: 200 OK + Data
```

**Format**: `Authorization: Basic base64(username:password)`

**Pros**: Simple
**Cons**: Not secure without HTTPS, credentials in every request

### API Key

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    
    Note over C: Has API key: abc123xyz
    
    C->>S: GET /api/users<br/>X-API-Key: abc123xyz
    S->>S: Validate key
    S->>C: 200 OK + Data
```

**Pros**: Simple, revocable
**Cons**: Long-lived, no user context

### JWT (JSON Web Token)

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    
    C->>S: POST /login<br/>{username, password}
    S->>S: Verify credentials
    S->>C: JWT token
    
    Note over C: Store token
    
    C->>S: GET /api/users<br/>Authorization: Bearer <JWT>
    S->>S: Verify token signature
    S->>C: 200 OK + Data
```

**JWT Structure**: `header.payload.signature`

**Pros**: Stateless, contains claims, secure
**Cons**: Can't revoke easily, size

### Comparison

```mermaid
graph TB
    A{Security<br/>Needs} --> B[Low<br/>Internal tools]
    A --> C[Medium<br/>Public API]
    A --> D[High<br/>User data]
    
    B --> E[API Key]
    C --> F[API Key + HTTPS]
    D --> G[OAuth 2.0 + JWT]
    
    style E fill:#FFD700
    style F fill:#87CEEB
    style G fill:#90EE90
```

---

## Summary

Key protocol and design concepts:
- **Protocols**: Rules for communication
- **OSI Model**: 7-layer network model
- **TCP vs UDP**: Reliable vs fast
- **DNS**: Name to IP resolution
- **HTTP**: Web communication protocol
- **REST**: Resource-based API design
- **WebSocket**: Real-time two-way communication
- **Load Balancing**: Distribute traffic
- **API Versioning**: Manage changes
- **Authentication**: Verify identity

These fundamentals enable building networked systems.

