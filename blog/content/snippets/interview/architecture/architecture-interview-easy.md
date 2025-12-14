---
title: "Architecture Interview Questions - Easy"
date: 2025-12-13
tags: ["architecture", "interview", "easy", "design-patterns", "software-architecture"]
---

Easy-level software architecture interview questions covering fundamental patterns, principles, and concepts.

## Q1: Explain the difference between monolithic and microservices architecture.

**Answer**:

### Monolithic Architecture

**Definition**: Single unified application where all components are tightly coupled and deployed together.

**Structure**:

```mermaid
graph TB
    A[Monolithic Application]
    A --> B[UI Layer]
    A --> C[Business Logic]
    A --> D[Data Access Layer]
    D --> E[(Single Database)]
    
    style A fill:#FFD700
```

**Pros**:
- Simple to develop and deploy
- Easy to test (single codebase)
- Better performance (no network calls between components)
- Simpler debugging

**Cons**:
- Hard to scale (must scale entire app)
- Tight coupling (changes affect everything)
- Long deployment times
- Technology lock-in

### Microservices Architecture

**Definition**: Application composed of small, independent services that communicate via APIs.

**Structure**:

```mermaid
graph TB
    A[API Gateway]
    A --> B[Service A]
    A --> C[Service B]
    A --> D[Service C]
    
    B --> E[(DB-A)]
    C --> F[(DB-B)]
    D --> G[(DB-C)]
    
    style A fill:#FFD700
```

**Pros**:
- Independent scaling
- Technology diversity
- Faster deployments
- Better fault isolation
- Team autonomy

**Cons**:
- Complex infrastructure
- Network latency
- Data consistency challenges
- Harder to debug
- Operational overhead

**When to Use Each**:
- **Monolith**: Small teams, simple apps, MVPs, startups
- **Microservices**: Large teams, complex domains, need to scale independently

---

## Q2: What are the SOLID principles?

**Answer**:

### S - Single Responsibility Principle (SRP)

**Definition**: A class should have only one reason to change.

**Example**:
```
❌ Bad:
class User {
    saveToDatabase()
    sendEmail()
    generateReport()
}

✅ Good:
class User { ... }
class UserRepository { saveToDatabase() }
class EmailService { sendEmail() }
class ReportGenerator { generateReport() }
```

### O - Open/Closed Principle (OCP)

**Definition**: Open for extension, closed for modification.

**Example**:
```
❌ Bad:
class PaymentProcessor {
    process(type) {
        if (type == "credit") { ... }
        else if (type == "paypal") { ... }
        // Adding new type requires modifying this class
    }
}

✅ Good:
interface PaymentMethod { process() }
class CreditCard implements PaymentMethod { ... }
class PayPal implements PaymentMethod { ... }
// Add new payment methods without changing existing code
```

### L - Liskov Substitution Principle (LSP)

**Definition**: Subtypes must be substitutable for their base types.

**Example**:
```
❌ Bad:
class Bird { fly() }
class Penguin extends Bird {
    fly() { throw Error("Can't fly!") }
}

✅ Good:
class Bird { ... }
class FlyingBird extends Bird { fly() }
class Penguin extends Bird { swim() }
```

### I - Interface Segregation Principle (ISP)

**Definition**: Clients shouldn't depend on interfaces they don't use.

**Example**:
```
❌ Bad:
interface Worker {
    work()
    eat()
    sleep()
}
class Robot implements Worker {
    eat() { /* robots don't eat! */ }
}

✅ Good:
interface Workable { work() }
interface Eatable { eat() }
interface Sleepable { sleep() }
class Human implements Workable, Eatable, Sleepable
class Robot implements Workable
```

### D - Dependency Inversion Principle (DIP)

**Definition**: Depend on abstractions, not concretions.

**Example**:
```
❌ Bad:
class EmailService { send() }
class Notification {
    emailService = new EmailService()  // Tight coupling
}

✅ Good:
interface MessageService { send() }
class EmailService implements MessageService { ... }
class SMSService implements MessageService { ... }
class Notification {
    constructor(messageService: MessageService) { ... }
}
```

**Benefits**: Maintainable, testable, flexible, scalable code.

---

## Q3: Explain the MVC (Model-View-Controller) pattern.

**Answer**:

**Definition**: Architectural pattern separating application into three interconnected components.

**Components**:

```mermaid
graph TB
    U[User] -->|interacts| C[Controller]
    C -->|updates| M[Model]
    C -->|selects| V[View]
    M -->|notifies| V
    
    M2[Model<br/>- Data<br/>- Logic]
    V2[View<br/>- UI<br/>- Display]
    C2[Controller<br/>- Handles input<br/>- Updates Model<br/>- Selects View]
    
    style C fill:#FFD700
```

### Model
- **Responsibility**: Data and business logic
- **Examples**: User class, database queries, validation
- **Independent**: Doesn't know about View or Controller

### View
- **Responsibility**: Presentation and UI
- **Examples**: HTML templates, React components, mobile screens
- **Observes**: Model for changes

### Controller
- **Responsibility**: Handle user input, coordinate Model and View
- **Examples**: Route handlers, event handlers
- **Mediator**: Between user and application

**Flow**:
1. User interacts with View (clicks button)
2. Controller receives input
3. Controller updates Model
4. Model notifies View of changes
5. View updates display

**Benefits**:
- Separation of concerns
- Parallel development
- Easier testing
- Code reusability

**Example Scenario** (E-commerce):
- **Model**: Product, Cart, Order classes with database logic
- **View**: Product listing page, shopping cart UI
- **Controller**: AddToCartController, CheckoutController

**Variations**:
- **MVP** (Model-View-Presenter): Presenter mediates all View-Model interaction
- **MVVM** (Model-View-ViewModel): Two-way data binding between View and ViewModel

---

## Q4: What is RESTful API design?

**Answer**:

**Definition**: Architectural style for designing networked applications using HTTP methods and resources.

**Core Principles**:

### 1. Resource-Based
Everything is a resource with a unique URI:
```
/users          - Collection of users
/users/123      - Specific user
/users/123/posts - User's posts
```

### 2. HTTP Methods (CRUD)
```
GET    /users       - Retrieve all users
GET    /users/123   - Retrieve user 123
POST   /users       - Create new user
PUT    /users/123   - Update user 123 (full)
PATCH  /users/123   - Update user 123 (partial)
DELETE /users/123   - Delete user 123
```

### 3. Stateless
Each request contains all information needed (no server-side session).

### 4. HTTP Status Codes
```
200 OK                  - Success
201 Created             - Resource created
204 No Content          - Success, no response body
400 Bad Request         - Invalid input
401 Unauthorized        - Authentication required
403 Forbidden           - No permission
404 Not Found           - Resource doesn't exist
500 Internal Server Error - Server error
```

### 5. Representation
Resources can have multiple representations (JSON, XML):
```json
GET /users/123
Accept: application/json

{
  "id": 123,
  "name": "Alice",
  "email": "alice@example.com"
}
```

**Best Practices**:

1. **Use nouns, not verbs**:
   - ✅ `GET /users`
   - ❌ `GET /getUsers`

2. **Use plural names**:
   - ✅ `/users/123`
   - ❌ `/user/123`

3. **Nested resources**:
   - ✅ `/users/123/posts/456`
   - ❌ `/posts/456` (if context matters)

4. **Filtering, sorting, pagination**:
   ```
   GET /users?role=admin&sort=name&page=2&limit=20
   ```

5. **Versioning**:
   ```
   /api/v1/users
   /api/v2/users
   ```

6. **HATEOAS** (Hypermedia):
   ```json
   {
     "id": 123,
     "name": "Alice",
     "_links": {
       "self": "/users/123",
       "posts": "/users/123/posts"
     }
   }
   ```

---

## Q5: Explain the difference between SQL and NoSQL databases.

**Answer**:

### SQL (Relational) Databases

**Structure**: Tables with fixed schema, rows and columns

**Examples**: PostgreSQL, MySQL, Oracle, SQL Server

**Schema**:

```mermaid
erDiagram
    USERS {
        int id PK
        string name
        string email
        int age
    }
```

**Example Data**:
| id | name  | email             | age |
|----|-------|-------------------|-----|
| 1  | Alice | alice@example.com | 30  |
| 2  | Bob   | bob@example.com   | 25  |

**Characteristics**:
- **ACID** transactions (Atomicity, Consistency, Isolation, Durability)
- **Structured** data with relationships
- **Schema-first**: Define structure before inserting data
- **Joins**: Combine data from multiple tables
- **Vertical scaling**: Add more CPU/RAM to single server

**When to Use**:
- Complex queries and relationships
- Transactions required (banking, e-commerce)
- Data integrity critical
- Structured, predictable data

### NoSQL Databases

**Types**:

1. **Document** (MongoDB, CouchDB):
   ```json
   {
     "_id": "123",
     "name": "Alice",
     "email": "alice@example.com",
     "addresses": [
       {"city": "NYC", "zip": "10001"},
       {"city": "LA", "zip": "90001"}
     ]
   }
   ```

2. **Key-Value** (Redis, DynamoDB):
   ```
   user:123 → {"name": "Alice", "email": "..."}
   session:abc → {"userId": 123, "expires": "..."}
   ```

3. **Column-Family** (Cassandra, HBase):
   ```
   Row Key: user:123
   Columns: name=Alice, email=..., age=30
   ```

4. **Graph** (Neo4j, ArangoDB):
   ```
   (Alice)-[:FOLLOWS]->(Bob)
   (Alice)-[:LIKES]->(Post1)
   ```

**Characteristics**:
- **Flexible schema**: Add fields without migration
- **Horizontal scaling**: Add more servers
- **Eventually consistent**: May not be immediately consistent
- **Denormalized**: Data duplication for performance

**When to Use**:
- Massive scale (millions of users)
- Flexible/evolving schema
- Hierarchical data
- High write throughput
- Geographic distribution

### Comparison

| Aspect | SQL | NoSQL |
|--------|-----|-------|
| Schema | Fixed | Flexible |
| Scaling | Vertical | Horizontal |
| Transactions | ACID | BASE (eventual consistency) |
| Relationships | Joins | Embedded/denormalized |
| Use Case | Complex queries | High scale, simple queries |

**Hybrid Approach**: Many systems use both (SQL for transactions, NoSQL for caching/analytics).

---

## Q6: What is the difference between authentication and authorization?

**Answer**:

### Authentication (AuthN)
**Question**: "Who are you?"

**Definition**: Verifying identity of a user.

**Methods**:
1. **Password-based**: Username + password
2. **Multi-factor (MFA)**: Password + SMS/app code
3. **Biometric**: Fingerprint, face recognition
4. **Token-based**: JWT, OAuth tokens
5. **Certificate-based**: SSL/TLS client certificates

**Flow**:
```
User → Provides credentials → System verifies → Issues token/session
```

**Example**:
```
POST /login
{
  "username": "alice",
  "password": "secret123"
}

Response:
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {"id": 123, "name": "Alice"}
}
```

### Authorization (AuthZ)
**Question**: "What can you do?"

**Definition**: Determining what an authenticated user can access.

**Models**:

1. **Role-Based (RBAC)**:
   ```
   User → Role → Permissions
   Alice → Admin → [read, write, delete]
   Bob → User → [read]
   ```

2. **Attribute-Based (ABAC)**:
   ```
   IF user.department == "HR" AND resource.type == "employee_record"
   THEN allow
   ```

3. **Access Control Lists (ACL)**:
   ```
   Document123:
   - Alice: read, write
   - Bob: read
   - Charlie: none
   ```

**Flow**:
```
User (authenticated) → Requests resource → System checks permissions → Allow/Deny
```

**Example**:
```
GET /admin/users
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

If user has "admin" role → 200 OK
If user doesn't have "admin" role → 403 Forbidden
```

### Comparison

| Aspect | Authentication | Authorization |
|--------|----------------|---------------|
| Question | Who are you? | What can you do? |
| Process | Verify identity | Check permissions |
| When | First (login) | Every request |
| Result | Token/session | Allow/deny access |
| Example | Login with password | Admin can delete users |

**Together in Practice**:
```
1. User logs in (Authentication)
   → System verifies credentials
   → Issues JWT token with user info and roles

2. User requests resource (Authorization)
   → System validates token (re-authentication)
   → Checks user's roles/permissions
   → Allows or denies access
```

---

## Q7: What is caching and what are common caching strategies?

**Answer**:

**Definition**: Storing frequently accessed data in fast storage to reduce latency and load.

**Cache Hierarchy**:

```mermaid
graph TB
    A[Browser Cache<br/>fastest, smallest]
    B[CDN Cache]
    C[Application Cache<br/>Redis, Memcached]
    D[Database Query Cache]
    E[(Database<br/>slowest, largest)]
    
    A --> B --> C --> D --> E
    
    style A fill:#90EE90
    style E fill:#FFB6C1
```

### Caching Strategies

#### 1. Cache-Aside (Lazy Loading)

**Flow**:
```
1. Check cache
2. If miss → Read from DB → Store in cache
3. If hit → Return from cache
```

**Pros**: Only cache what's needed  
**Cons**: Cache miss penalty, stale data possible

**Use**: General-purpose caching

#### 2. Write-Through

**Flow**:
```
1. Write to cache
2. Immediately write to database
3. Return success
```

**Pros**: Cache always consistent  
**Cons**: Write latency (two writes)

**Use**: Data consistency critical

#### 3. Write-Behind (Write-Back)

**Flow**:
```
1. Write to cache
2. Return success immediately
3. Asynchronously write to database later
```

**Pros**: Fast writes  
**Cons**: Risk of data loss

**Use**: High write throughput needed

#### 4. Read-Through

**Flow**:
```
1. Check cache
2. If miss → Cache loads from DB automatically
3. Return data
```

**Pros**: Simplified application code  
**Cons**: Cache must implement loading logic

#### 5. Refresh-Ahead

**Flow**:
```
1. Predict which data will be needed
2. Refresh cache before expiration
3. Always serve from cache
```

**Pros**: No cache miss penalty  
**Cons**: May refresh unused data

**Use**: Predictable access patterns

### Cache Eviction Policies

**LRU** (Least Recently Used): Remove oldest accessed  
**LFU** (Least Frequently Used): Remove least accessed  
**FIFO** (First In First Out): Remove oldest added  
**TTL** (Time To Live): Remove after expiration

### Cache Invalidation

**Problem**: "There are only two hard things in Computer Science: cache invalidation and naming things."

**Strategies**:
1. **TTL-based**: Expire after time
2. **Event-based**: Invalidate on updates
3. **Version-based**: Include version in cache key

**Example**:
```
# TTL-based
cache.set("user:123", data, ttl=3600)  # 1 hour

# Event-based
onUserUpdate(userId) {
    cache.delete(`user:${userId}`)
}

# Version-based
cache.set(`user:123:v2`, data)
```

---

## Q8: Explain the CAP theorem.

**Answer**:

**Definition**: In a distributed system, you can only guarantee 2 out of 3 properties:

### C - Consistency
**Definition**: All nodes see the same data at the same time.

**Example**: After writing to any node, all subsequent reads return that value.

### A - Availability
**Definition**: Every request receives a response (success or failure).

**Example**: System responds even if some nodes are down.

### P - Partition Tolerance
**Definition**: System continues operating despite network partitions.

**Example**: System works even when nodes can't communicate.

### Trade-offs

```mermaid
graph TB
    A[CAP Theorem]
    A --> B[CA<br/>Consistency + Availability<br/>Single-node RDBMS]
    A --> C[CP<br/>Consistency + Partition Tolerance<br/>MongoDB, HBase]
    A --> D[AP<br/>Availability + Partition Tolerance<br/>Cassandra, DynamoDB]
    
    style A fill:#FFD700
    style B fill:#87CEEB
    style C fill:#98FB98
    style D fill:#FFB6C1
```

**CA** (Consistency + Availability):
- Single-node systems (no partitions possible)
- Traditional RDBMS on single server
- **Example**: PostgreSQL on one machine

**CP** (Consistency + Partition Tolerance):
- Sacrifice availability during partitions
- Wait for consensus before responding
- **Examples**: MongoDB, HBase, Redis (with replication)

**AP** (Availability + Partition Tolerance):
- Sacrifice consistency (eventual consistency)
- Always respond, even with stale data
- **Examples**: Cassandra, DynamoDB, CouchDB

### Real-World Example

**Banking System** (CP):
```
During network partition:
- Can't guarantee all nodes have same balance
- Better to reject transactions (unavailable)
- Than allow inconsistent balances
```

**Social Media Feed** (AP):
```
During network partition:
- Better to show slightly stale feed
- Than show error message
- Eventual consistency acceptable
```

**Note**: In practice, partition tolerance is required (networks fail), so real choice is between **CP** and **AP**.

---

## Q9: What are design patterns? Name common ones.

**Answer**:

**Definition**: Reusable solutions to common software design problems.

### Creational Patterns (Object Creation)

**1. Singleton**: Ensure only one instance exists
```
class Database {
    private static instance;
    
    private constructor() {}
    
    static getInstance() {
        if (!instance) {
            instance = new Database();
        }
        return instance;
    }
}
```

**2. Factory**: Create objects without specifying exact class
```
class ShapeFactory {
    createShape(type) {
        if (type === "circle") return new Circle();
        if (type === "square") return new Square();
    }
}
```

**3. Builder**: Construct complex objects step by step
```
new UserBuilder()
    .setName("Alice")
    .setEmail("alice@example.com")
    .setAge(30)
    .build();
```

### Structural Patterns (Object Composition)

**4. Adapter**: Make incompatible interfaces work together
```
class PayPalAdapter implements PaymentProcessor {
    constructor(paypal) { this.paypal = paypal; }
    
    processPayment(amount) {
        this.paypal.sendMoney(amount);  // Adapt interface
    }
}
```

**5. Decorator**: Add behavior without modifying class
```
class LoggingDecorator {
    constructor(service) { this.service = service; }
    
    execute() {
        console.log("Starting...");
        this.service.execute();
        console.log("Done");
    }
}
```

**6. Proxy**: Control access to object
```
class CachedAPI {
    constructor(api) {
        this.api = api;
        this.cache = {};
    }
    
    fetch(url) {
        if (this.cache[url]) return this.cache[url];
        this.cache[url] = this.api.fetch(url);
        return this.cache[url];
    }
}
```

### Behavioral Patterns (Object Interaction)

**7. Observer**: Notify multiple objects of changes
```
class EventEmitter {
    listeners = [];
    
    subscribe(listener) { this.listeners.push(listener); }
    
    notify(event) {
        this.listeners.forEach(l => l.update(event));
    }
}
```

**8. Strategy**: Select algorithm at runtime
```
class PaymentProcessor {
    constructor(strategy) { this.strategy = strategy; }
    
    process(amount) {
        this.strategy.pay(amount);
    }
}

// Use: new PaymentProcessor(new CreditCardStrategy())
```

**9. Command**: Encapsulate requests as objects
```
class SaveCommand {
    constructor(document) { this.doc = document; }
    execute() { this.doc.save(); }
    undo() { this.doc.revert(); }
}
```

**10. Template Method**: Define algorithm skeleton, let subclasses override steps
```
class DataProcessor {
    process() {
        this.readData();
        this.processData();  // Override in subclass
        this.saveData();
    }
}
```

---

## Q10: What is the difference between horizontal and vertical scaling?

**Answer**:

### Vertical Scaling (Scale Up)

**Definition**: Add more resources (CPU, RAM, disk) to existing machine.

```mermaid
graph LR
    A[Before<br/>4 CPU<br/>8 GB RAM] -->|Scale Up| B[After<br/>16 CPU<br/>64 GB RAM]
    
    style A fill:#FFB6C1
    style B fill:#90EE90
```

**Pros**:
- Simple (no code changes)
- No distributed system complexity
- Consistent data (single database)
- Lower latency (no network calls)

**Cons**:
- Hardware limits (can't scale infinitely)
- Expensive (high-end hardware costs more)
- Single point of failure
- Downtime during upgrades

**When to Use**:
- Small to medium applications
- Monolithic architecture
- Relational databases (PostgreSQL, MySQL)
- When simplicity is priority

### Horizontal Scaling (Scale Out)

**Definition**: Add more machines to distribute load.

```mermaid
graph TB
    A[Load Balancer]
    A --> B[Server 1]
    A --> C[Server 2]
    A --> D[Server 3]
    
    E[Before: Single Server] -.->|Scale Out| A
    
    style A fill:#FFD700
    style E fill:#FFB6C1
```

**Pros**:
- Nearly unlimited scaling
- Better fault tolerance (redundancy)
- Cost-effective (commodity hardware)
- No downtime (add servers without stopping)

**Cons**:
- Complex (distributed systems)
- Data consistency challenges
- Network latency
- Requires load balancing

**When to Use**:
- Large-scale applications
- Microservices
- NoSQL databases (Cassandra, MongoDB)
- High availability required

### Comparison

| Aspect | Vertical | Horizontal |
|--------|----------|------------|
| Cost | Expensive | Cost-effective |
| Limit | Hardware limit | Nearly unlimited |
| Complexity | Simple | Complex |
| Downtime | Yes | No |
| Fault Tolerance | Single point of failure | Redundant |
| Data Consistency | Easy | Challenging |

### Hybrid Approach

Most systems use both:

```mermaid
graph TB
    A[Load Balancer]
    A --> B[Server 1<br/>16 CPU, 64 GB]
    A --> C[Server 2<br/>16 CPU, 64 GB]
    A --> D[Server 3<br/>16 CPU, 64 GB]
    
    style A fill:#FFD700
    style B fill:#90EE90
    style C fill:#90EE90
    style D fill:#90EE90
```

**Strategy**: Scale vertically until cost-prohibitive, then scale horizontally.

---

## Summary

Key architecture concepts:
- **Architectural styles**: Monolithic vs. Microservices
- **Design principles**: SOLID
- **Patterns**: MVC, design patterns
- **APIs**: REST principles
- **Databases**: SQL vs. NoSQL
- **Security**: Authentication vs. Authorization
- **Performance**: Caching strategies
- **Distributed systems**: CAP theorem
- **Scaling**: Vertical vs. Horizontal

These fundamentals form the foundation for more advanced architectural decisions.

