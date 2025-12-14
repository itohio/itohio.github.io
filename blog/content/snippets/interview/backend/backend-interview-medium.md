---
title: "Backend Interview Questions - Medium"
date: 2025-12-14
tags: ["backend", "interview", "medium", "microservices", "databases", "performance"]
---

Medium-level backend interview questions covering microservices, advanced databases, message queues, and performance optimization.

## Q1: Explain microservices architecture and its trade-offs.

**Answer**:

```mermaid
graph TB
    A[Microservices] --> B[Independent<br/>Services]
    A --> C[Decentralized<br/>Data]
    A --> D[Technology<br/>Diversity]
    A --> E[Scalability<br/>Per service]
    
    style A fill:#FFD700
```

### Monolith vs Microservices

```mermaid
graph TB
    subgraph Monolith
        M[Single Application] --> M1[User Service]
        M --> M2[Order Service]
        M --> M3[Payment Service]
        M --> DB[(Single Database)]
    end
    
    subgraph Microservices
        MS1[User Service] --> DB1[(User DB)]
        MS2[Order Service] --> DB2[(Order DB)]
        MS3[Payment Service] --> DB3[(Payment DB)]
        
        MS1 <-->|API| MS2
        MS2 <-->|API| MS3
    end
```

### Advantages

- **Independent deployment**: Deploy services separately
- **Technology flexibility**: Different tech per service
- **Scalability**: Scale services independently
- **Fault isolation**: One service failure doesn't crash all
- **Team autonomy**: Teams own services

### Disadvantages

- **Complexity**: Distributed system challenges
- **Data consistency**: No ACID across services
- **Network latency**: Inter-service communication
- **Testing**: Integration testing harder
- **Deployment**: More moving parts

### Communication Patterns

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

// Synchronous (REST)
func CreateOrderSync(userID string, items []Item) (*Order, error) {
	// Call user service
	userResp, err := http.Get(fmt.Sprintf("http://user-service/users/%s", userID))
	if err != nil {
		return nil, err
	}
	defer userResp.Body.Close()

	var user User
	json.NewDecoder(userResp.Body).Decode(&user)

	// Call inventory service
	checkReq := CheckInventoryRequest{Items: items}
	reqBody, _ := json.Marshal(checkReq)

	invResp, err := http.Post(
		"http://inventory-service/check",
		"application/json",
		bytes.NewBuffer(reqBody),
	)
	if err != nil {
		return nil, err
	}
	defer invResp.Body.Close()

	var available bool
	json.NewDecoder(invResp.Body).Decode(&available)

	if available {
		// Create order
		return createOrderInDB(userID, items)
	}

	return nil, fmt.Errorf("items not available")
}

// Asynchronous (Message Queue)
func CreateOrderAsync(userID string, items []Item, mq MessageQueue) (*Order, error) {
	// Create order with pending status
	order, err := createOrderInDB(userID, items)
	if err != nil {
		return nil, err
	}
	order.Status = "pending"

	// Publish event
	event := OrderCreatedEvent{
		OrderID: order.ID,
		UserID:  userID,
		Items:   items,
	}

	if err := mq.Publish("order.created", event); err != nil {
		return nil, err
	}

	return order, nil
}
```

---

## Q2: What are database transactions and ACID properties?

**Answer**:

```mermaid
graph TB
    A[ACID] --> B[Atomicity<br/>All or nothing]
    A --> C[Consistency<br/>Valid state]
    A --> D[Isolation<br/>Concurrent execution]
    A --> E[Durability<br/>Permanent changes]
    
    style A fill:#FFD700
```

### Transaction Example

```sql
BEGIN TRANSACTION;

-- Deduct from account A
UPDATE accounts SET balance = balance - 100 WHERE id = 1;

-- Add to account B
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- If both succeed, commit
COMMIT;

-- If any fails, rollback
ROLLBACK;
```

### Isolation Levels

```mermaid
graph TB
    A[Isolation Levels] --> B[Read Uncommitted<br/>Dirty reads possible]
    A --> C[Read Committed<br/>No dirty reads]
    A --> D[Repeatable Read<br/>Consistent reads]
    A --> E[Serializable<br/>Full isolation]
    
    B --> F[Lowest isolation<br/>Highest performance]
    E --> G[Highest isolation<br/>Lowest performance]
    
    style B fill:#FF6B6B
    style E fill:#90EE90
```

### Concurrency Problems

```javascript
// Dirty Read
// Transaction 1
UPDATE accounts SET balance = 1000 WHERE id = 1;
// Transaction 2 reads balance = 1000 (uncommitted)
ROLLBACK; // Transaction 1 rolls back

// Non-Repeatable Read
// Transaction 1
SELECT balance FROM accounts WHERE id = 1; // Returns 100
// Transaction 2 updates balance to 200 and commits
SELECT balance FROM accounts WHERE id = 1; // Returns 200 (different!)

// Phantom Read
// Transaction 1
SELECT COUNT(*) FROM accounts WHERE balance > 100; // Returns 5
// Transaction 2 inserts new account with balance 200 and commits
SELECT COUNT(*) FROM accounts WHERE balance > 100; // Returns 6 (phantom!)
```

### Implementation

```go
package main

import (
	"context"
	"database/sql"
	"fmt"

	_ "github.com/lib/pq"
)

func TransferMoney(ctx context.Context, db *sql.DB, fromID, toID string, amount float64) error {
	// Begin transaction
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback() // Rollback if not committed

	// Deduct from sender
	result, err := tx.ExecContext(ctx,
		`UPDATE accounts 
		 SET balance = balance - $1 
		 WHERE id = $2 AND balance >= $1`,
		amount, fromID,
	)
	if err != nil {
		return err
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		return fmt.Errorf("insufficient funds")
	}

	// Add to receiver
	_, err = tx.ExecContext(ctx,
		`UPDATE accounts 
		 SET balance = balance + $1 
		 WHERE id = $2`,
		amount, toID,
	)
	if err != nil {
		return err
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return err
	}

	return nil
}

// Usage
func main() {
	db, err := sql.Open("postgres", "postgres://user:pass@localhost/mydb?sslmode=disable")
	if err != nil {
		panic(err)
	}
	defer db.Close()

	ctx := context.Background()
	if err := TransferMoney(ctx, db, "account1", "account2", 100.0); err != nil {
		fmt.Println("Transfer failed:", err)
	} else {
		fmt.Println("Transfer successful")
	}
}
```

---

## Q3: Explain message queues and when to use them.

**Answer**:

```mermaid
graph TB
    A[Message Queue] --> B[Asynchronous<br/>Processing]
    A --> C[Decoupling<br/>Services]
    A --> D[Load Leveling<br/>Handle spikes]
    A --> E[Reliability<br/>Guaranteed delivery]
    
    style A fill:#FFD700
```

### Architecture

```mermaid
graph LR
    A[Producer] -->|Publish| B[Message Queue]
    B -->|Consume| C[Consumer 1]
    B -->|Consume| D[Consumer 2]
    B -->|Consume| E[Consumer 3]
    
    style B fill:#FFD700
```

### Use Cases

**1. Background Jobs**:
```go
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"

	"github.com/hibiken/asynq"
)

const TypeEmailDelivery = "email:deliver"

type EmailPayload struct {
	To       string `json:"to"`
	Template string `json:"template"`
}

// Producer: API endpoint
func CreateUserHandler(w http.ResponseWriter, r *http.Request, client *asynq.Client) {
	var req CreateUserRequest
	json.NewDecoder(r.Body).Decode(&req)

	// Create user in database
	user := createUserInDB(req)

	// Queue email sending (don't wait)
	payload, _ := json.Marshal(EmailPayload{
		To:       user.Email,
		Template: "welcome",
	})

	task := asynq.NewTask(TypeEmailDelivery, payload)
	client.Enqueue(task)

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(user)
}

// Consumer: Worker process
func HandleEmailDelivery(ctx context.Context, t *asynq.Task) error {
	var payload EmailPayload
	if err := json.Unmarshal(t.Payload(), &payload); err != nil {
		return err
	}

	log.Printf("Sending email to %s with template %s", payload.To, payload.Template)
	return sendEmail(payload.To, payload.Template)
}

// Worker setup
func StartWorker() {
	srv := asynq.NewServer(
		asynq.RedisClientOpt{Addr: "localhost:6379"},
		asynq.Config{Concurrency: 10},
	)

	mux := asynq.NewServeMux()
	mux.HandleFunc(TypeEmailDelivery, HandleEmailDelivery)

	if err := srv.Run(mux); err != nil {
		log.Fatal(err)
	}
}
```

**2. Event-Driven Architecture**:
```javascript
// Order service publishes event
await eventBus.publish('order.created', {
  orderId: order.id,
  userId: user.id,
  total: order.total
});

// Multiple services subscribe
// Inventory service
eventBus.subscribe('order.created', async (event) => {
  await inventory.reserve(event.orderId);
});

// Notification service
eventBus.subscribe('order.created', async (event) => {
  await notifications.send(event.userId, 'Order confirmed');
});

// Analytics service
eventBus.subscribe('order.created', async (event) => {
  await analytics.track('order_created', event);
});
```

### RabbitMQ Example

```javascript
const amqp = require('amqplib');

// Producer
async function publishMessage(queue, message) {
  const connection = await amqp.connect('amqp://localhost');
  const channel = await connection.createChannel();
  
  await channel.assertQueue(queue, { durable: true });
  
  channel.sendToQueue(
    queue,
    Buffer.from(JSON.stringify(message)),
    { persistent: true }
  );
  
  await channel.close();
  await connection.close();
}

// Consumer
async function consumeMessages(queue, handler) {
  const connection = await amqp.connect('amqp://localhost');
  const channel = await connection.createChannel();
  
  await channel.assertQueue(queue, { durable: true });
  channel.prefetch(1); // Process one at a time
  
  channel.consume(queue, async (msg) => {
    const content = JSON.parse(msg.content.toString());
    
    try {
      await handler(content);
      channel.ack(msg); // Acknowledge success
    } catch (error) {
      channel.nack(msg, false, true); // Requeue on failure
    }
  });
}
```

---

## Q4: What is database replication and sharding?

**Answer**:

```mermaid
graph TB
    A[Scaling Strategies] --> B[Replication<br/>Copy data]
    A --> C[Sharding<br/>Split data]
    
    B --> D[Read scalability]
    C --> E[Write scalability]
    
    style A fill:#FFD700
```

### Replication

```mermaid
graph TB
    A[Primary<br/>Read/Write] --> B[Replica 1<br/>Read only]
    A --> C[Replica 2<br/>Read only]
    A --> D[Replica 3<br/>Read only]
    
    E[Write] --> A
    F[Read] --> B
    G[Read] --> C
    H[Read] --> D
    
    style A fill:#FFD700
    style B fill:#90EE90
    style C fill:#90EE90
    style D fill:#90EE90
```

**Types**:
- **Master-Slave**: One primary, multiple replicas
- **Master-Master**: Multiple primaries (bidirectional)
- **Synchronous**: Wait for replica confirmation
- **Asynchronous**: Don't wait for replicas

### Sharding

```mermaid
graph TB
    A[Application] --> B{Router}
    
    B -->|Users 1-1000| C[Shard 1]
    B -->|Users 1001-2000| D[Shard 2]
    B -->|Users 2001-3000| E[Shard 3]
    
    style B fill:#FFD700
```

**Sharding Strategies**:

**1. Range-Based**:
```javascript
function getShard(userId) {
  if (userId <= 1000) return 'shard1';
  if (userId <= 2000) return 'shard2';
  return 'shard3';
}
```

**2. Hash-Based**:
```javascript
function getShard(userId) {
  const hash = hashFunction(userId);
  const shardIndex = hash % NUM_SHARDS;
  return `shard${shardIndex}`;
}
```

**3. Geographic**:
```javascript
function getShard(userId) {
  const user = getUser(userId);
  if (user.country === 'US') return 'shard-us';
  if (user.country === 'EU') return 'shard-eu';
  return 'shard-asia';
}
```

### Implementation

```javascript
class ShardedDatabase {
  constructor(shards) {
    this.shards = shards; // Array of database connections
  }
  
  getShard(key) {
    const hash = this.hash(key);
    const index = hash % this.shards.length;
    return this.shards[index];
  }
  
  async insert(key, data) {
    const shard = this.getShard(key);
    return await shard.query('INSERT INTO users VALUES ($1, $2)', [key, data]);
  }
  
  async get(key) {
    const shard = this.getShard(key);
    return await shard.query('SELECT * FROM users WHERE id = $1', [key]);
  }
  
  hash(key) {
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      hash = ((hash << 5) - hash) + key.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash);
  }
}
```

---

## Q5: Explain API rate limiting strategies.

**Answer**:

```mermaid
graph TB
    A[Rate Limiting] --> B[Prevent Abuse<br/>DDoS protection]
    A --> C[Fair Usage<br/>Resource allocation]
    A --> D[Cost Control<br/>API limits]
    
    style A fill:#FFD700
```

### Algorithms

**1. Fixed Window**:
```go
package main

import (
	"fmt"
	"sync"
	"time"
)

type FixedWindowLimiter struct {
	mu      sync.Mutex
	limits  map[string]int
	limit   int
	window  time.Duration
}

func NewFixedWindowLimiter(limit int, window time.Duration) *FixedWindowLimiter {
	return &FixedWindowLimiter{
		limits: make(map[string]int),
		limit:  limit,
		window: window,
	}
}

func (f *FixedWindowLimiter) Allow(userID string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()

	now := time.Now()
	windowStart := now.Truncate(f.window)
	key := fmt.Sprintf("%s:%d", userID, windowStart.Unix())

	count := f.limits[key]
	if count >= f.limit {
		return false
	}

	f.limits[key] = count + 1
	return true
}
```

**2. Sliding Window**:
```go
package main

import (
	"sync"
	"time"
)

type SlidingWindowLimiter struct {
	mu       sync.Mutex
	requests map[string][]time.Time
	limit    int
	window   time.Duration
}

func NewSlidingWindowLimiter(limit int, window time.Duration) *SlidingWindowLimiter {
	return &SlidingWindowLimiter{
		requests: make(map[string][]time.Time),
		limit:    limit,
		window:   window,
	}
}

func (s *SlidingWindowLimiter) Allow(userID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	windowStart := now.Add(-s.window)

	// Get and filter old requests
	requests := s.requests[userID]
	filtered := make([]time.Time, 0)
	for _, t := range requests {
		if t.After(windowStart) {
			filtered = append(filtered, t)
		}
	}

	if len(filtered) >= s.limit {
		s.requests[userID] = filtered
		return false
	}

	filtered = append(filtered, now)
	s.requests[userID] = filtered
	return true
}
```

**3. Token Bucket**:
```go
package main

import (
	"sync"
	"time"
)

type TokenBucket struct {
	mu         sync.Mutex
	capacity   float64
	tokens     float64
	refillRate float64 // tokens per second
	lastRefill time.Time
}

func NewTokenBucket(capacity, refillRate float64) *TokenBucket {
	return &TokenBucket{
		capacity:   capacity,
		tokens:     capacity,
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

func (tb *TokenBucket) refill() {
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tokensToAdd := elapsed * tb.refillRate

	tb.tokens = min(tb.capacity, tb.tokens+tokensToAdd)
	tb.lastRefill = now
}

func (tb *TokenBucket) Allow(tokens float64) bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	tb.refill()

	if tb.tokens >= tokens {
		tb.tokens -= tokens
		return true
	}

	return false
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// Usage with multiple users
type RateLimiter struct {
	mu      sync.Mutex
	buckets map[string]*TokenBucket
	capacity float64
	refillRate float64
}

func NewRateLimiter(capacity, refillRate float64) *RateLimiter {
	return &RateLimiter{
		buckets: make(map[string]*TokenBucket),
		capacity: capacity,
		refillRate: refillRate,
	}
}

func (rl *RateLimiter) Allow(userID string) bool {
	rl.mu.Lock()
	bucket, exists := rl.buckets[userID]
	if !exists {
		bucket = NewTokenBucket(rl.capacity, rl.refillRate)
		rl.buckets[userID] = bucket
	}
	rl.mu.Unlock()

	return bucket.Allow(1)
}
```

### Redis Implementation

```javascript
const redis = require('redis');
const client = redis.createClient();

async function rateLimitRedis(userId, limit, windowSeconds) {
  const key = `ratelimit:${userId}`;
  
  const current = await client.incr(key);
  
  if (current === 1) {
    await client.expire(key, windowSeconds);
  }
  
  if (current > limit) {
    const ttl = await client.ttl(key);
    return {
      allowed: false,
      remaining: 0,
      resetIn: ttl
    };
  }
  
  return {
    allowed: true,
    remaining: limit - current
  };
}

// Middleware
function rateLimitMiddleware(limit, window) {
  return async (req, res, next) => {
    const userId = req.user?.id || req.ip;
    
    const result = await rateLimitRedis(userId, limit, window);
    
    res.set({
      'X-RateLimit-Limit': limit,
      'X-RateLimit-Remaining': result.remaining,
      'X-RateLimit-Reset': result.resetIn
    });
    
    if (!result.allowed) {
      return res.status(429).json({
        error: 'Too many requests',
        retryAfter: result.resetIn
      });
    }
    
    next();
  };
}

// Usage
app.use('/api/', rateLimitMiddleware(100, 60)); // 100 requests per minute
```

---

## Q6: What is connection pooling and why is it important?

**Answer**:

```mermaid
graph TB
    A[Connection Pool] --> B[Reuse Connections<br/>Avoid overhead]
    A --> C[Limit Connections<br/>Prevent exhaustion]
    A --> D[Performance<br/>Faster queries]
    
    style A fill:#FFD700
```

### Without Connection Pool

```mermaid
sequenceDiagram
    participant A as App
    participant DB as Database
    
    loop Each Request
        A->>DB: Open connection (slow)
        A->>DB: Execute query
        A->>DB: Close connection
    end
    
    Note over A,DB: High overhead
```

### With Connection Pool

```mermaid
sequenceDiagram
    participant A as App
    participant P as Pool
    participant DB as Database
    
    Note over P: Pre-created connections
    
    A->>P: Get connection (fast)
    P->>A: Return connection
    A->>DB: Execute query
    A->>P: Release connection
    
    Note over P: Connection reused
```

### Implementation

```go
package main

import (
	"context"
	"database/sql"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	_ "github.com/lib/pq"
)

// Create connection pool
func NewDB() (*sql.DB, error) {
	connStr := "postgres://user:password@localhost:5432/mydb?sslmode=disable"
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, err
	}

	// Configure pool
	db.SetMaxOpenConns(20)                 // Maximum connections
	db.SetMaxIdleConns(5)                  // Minimum idle connections
	db.SetConnMaxLifetime(time.Hour)       // Max connection lifetime
	db.SetConnMaxIdleTime(30 * time.Second) // Max idle time

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, err
	}

	return db, nil
}

// Use pool with explicit connection
func GetUser(ctx context.Context, db *sql.DB, userID string) (*User, error) {
	// Get connection from pool (automatic)
	var user User
	err := db.QueryRowContext(ctx,
		"SELECT id, name, email FROM users WHERE id = $1",
		userID,
	).Scan(&user.ID, &user.Name, &user.Email)

	if err != nil {
		return nil, err
	}

	return &user, nil
}

// Direct pool usage
func GetAllUsers(ctx context.Context, db *sql.DB) ([]User, error) {
	rows, err := db.QueryContext(ctx, "SELECT id, name, email FROM users")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var users []User
	for rows.Next() {
		var user User
		if err := rows.Scan(&user.ID, &user.Name, &user.Email); err != nil {
			return nil, err
		}
		users = append(users, user)
	}

	return users, rows.Err()
}

// Graceful shutdown
func main() {
	db, err := NewDB()
	if err != nil {
		log.Fatal(err)
	}

	// Setup graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Shutting down gracefully...")
		db.Close()
		os.Exit(0)
	}()

	// Your application logic here
	ctx := context.Background()
	user, err := GetUser(ctx, db, "123")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("User: %+v", user)
}
```

### Monitoring

```javascript
// Pool events
pool.on('connect', (client) => {
  console.log('New client connected');
});

pool.on('acquire', (client) => {
  console.log('Client acquired from pool');
});

pool.on('remove', (client) => {
  console.log('Client removed from pool');
});

pool.on('error', (err, client) => {
  console.error('Pool error:', err);
});

// Pool stats
function getPoolStats() {
  return {
    total: pool.totalCount,
    idle: pool.idleCount,
    waiting: pool.waitingCount
  };
}
```

---

## Q7: Explain API versioning strategies.

**Answer**:

```mermaid
graph TB
    A[API Versioning] --> B[URL Path<br/>/v1/users]
    A --> C[Query Parameter<br/>?version=1]
    A --> D[Header<br/>Accept: v1]
    A --> E[Subdomain<br/>v1.api.com]
    
    style A fill:#FFD700
    style B fill:#90EE90
```

### URL Path Versioning

```javascript
// Version 1
app.get('/api/v1/users', (req, res) => {
  res.json({
    users: [
      { id: 1, name: 'Alice' }
    ]
  });
});

// Version 2 (breaking change: added email)
app.get('/api/v2/users', (req, res) => {
  res.json({
    users: [
      { id: 1, name: 'Alice', email: 'alice@example.com' }
    ]
  });
});
```

### Header Versioning

```javascript
function versionMiddleware(req, res, next) {
  const version = req.headers['api-version'] || '1';
  req.apiVersion = version;
  next();
}

app.use(versionMiddleware);

app.get('/api/users', (req, res) => {
  if (req.apiVersion === '1') {
    return res.json({ users: getUsersV1() });
  }
  
  if (req.apiVersion === '2') {
    return res.json({ users: getUsersV2() });
  }
  
  res.status(400).json({ error: 'Unsupported API version' });
});
```

### Content Negotiation

```javascript
app.get('/api/users', (req, res) => {
  const accept = req.headers.accept;
  
  if (accept.includes('application/vnd.api.v1+json')) {
    return res.json({ users: getUsersV1() });
  }
  
  if (accept.includes('application/vnd.api.v2+json')) {
    return res.json({ users: getUsersV2() });
  }
  
  res.status(406).json({ error: 'Not acceptable' });
});
```

### Deprecation Strategy

```javascript
// Deprecation warning
app.get('/api/v1/users', (req, res) => {
  res.set({
    'X-API-Deprecated': 'true',
    'X-API-Deprecation-Date': '2024-12-31',
    'X-API-Sunset-Date': '2025-06-30',
    'Link': '</api/v2/users>; rel="successor-version"'
  });
  
  res.json({ users: getUsersV1() });
});

// Version lifecycle
const versions = {
  v1: { status: 'deprecated', sunsetDate: '2025-06-30' },
  v2: { status: 'current' },
  v3: { status: 'beta' }
};
```

---

## Q8: What is database connection leak and how to prevent it?

**Answer**:

```mermaid
graph TB
    A[Connection Leak] --> B[Connections Not<br/>Released]
    A --> C[Pool Exhausted<br/>No available connections]
    A --> D[Application Hangs<br/>Waiting for connection]
    
    style A fill:#FF6B6B
```

### Connection Leak Example

```javascript
// ❌ Connection leak
async function badQuery(userId) {
  const client = await pool.connect();
  
  const result = await client.query(
    'SELECT * FROM users WHERE id = $1',
    [userId]
  );
  
  // Forgot to release!
  return result.rows[0];
}

// After many requests, pool is exhausted
```

### Prevention Strategies

**1. Always Use try-finally**:
```javascript
// ✅ Proper connection handling
async function goodQuery(userId) {
  const client = await pool.connect();
  
  try {
    const result = await client.query(
      'SELECT * FROM users WHERE id = $1',
      [userId]
    );
    return result.rows[0];
  } finally {
    client.release(); // Always released
  }
}
```

**2. Use Pool Directly**:
```javascript
// ✅ Pool handles connection automatically
async function betterQuery(userId) {
  const result = await pool.query(
    'SELECT * FROM users WHERE id = $1',
    [userId]
  );
  return result.rows[0];
}
```

**3. Wrapper Function**:
```javascript
// ✅ Wrapper ensures cleanup
async function withConnection(callback) {
  const client = await pool.connect();
  
  try {
    return await callback(client);
  } finally {
    client.release();
  }
}

// Usage
const user = await withConnection(async (client) => {
  const result = await client.query(
    'SELECT * FROM users WHERE id = $1',
    [userId]
  );
  return result.rows[0];
});
```

### Monitoring and Detection

```javascript
// Monitor pool health
setInterval(() => {
  const stats = {
    total: pool.totalCount,
    idle: pool.idleCount,
    waiting: pool.waitingCount
  };
  
  console.log('Pool stats:', stats);
  
  // Alert if pool is exhausted
  if (stats.idle === 0 && stats.waiting > 0) {
    console.error('WARNING: Connection pool exhausted!');
  }
}, 10000);

// Set connection timeout
const pool = new Pool({
  connectionTimeoutMillis: 5000, // Fail fast
  idleTimeoutMillis: 30000 // Close idle connections
});
```

---

## Q9: Explain database indexing strategies for complex queries.

**Answer**:

```mermaid
graph TB
    A[Index Types] --> B[B-Tree<br/>Default, sorted]
    A --> C[Hash<br/>Equality only]
    A --> D[GiST<br/>Geometric]
    A --> E[GIN<br/>Full-text, arrays]
    
    style A fill:#FFD700
```

### Composite Indexes

```sql
-- Single column index
CREATE INDEX idx_users_email ON users(email);

-- Composite index (order matters!)
CREATE INDEX idx_users_name_email ON users(last_name, first_name);

-- Efficient for:
SELECT * FROM users WHERE last_name = 'Smith'; -- ✓ Uses index
SELECT * FROM users WHERE last_name = 'Smith' AND first_name = 'John'; -- ✓ Uses index

-- Not efficient for:
SELECT * FROM users WHERE first_name = 'John'; -- ✗ Doesn't use index
```

### Partial Indexes

```sql
-- Index only active users
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- Smaller index, faster queries for active users
SELECT * FROM users WHERE email = 'alice@example.com' AND status = 'active';
```

### Expression Indexes

```sql
-- Index on computed value
CREATE INDEX idx_users_lower_email ON users(LOWER(email));

-- Efficient for case-insensitive search
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';
```

### Covering Indexes

```sql
-- Include additional columns
CREATE INDEX idx_users_email_name ON users(email) INCLUDE (first_name, last_name);

-- Query can be satisfied entirely from index (no table lookup)
SELECT first_name, last_name FROM users WHERE email = 'alice@example.com';
```

### Full-Text Search

```sql
-- Create GIN index for full-text search
CREATE INDEX idx_posts_content_fts ON posts USING GIN(to_tsvector('english', content));

-- Full-text search
SELECT * FROM posts 
WHERE to_tsvector('english', content) @@ to_tsquery('english', 'database & performance');
```

### Query Optimization

```sql
-- Analyze query plan
EXPLAIN ANALYZE
SELECT * FROM users 
WHERE email = 'alice@example.com' 
AND status = 'active';

-- Output shows:
-- - Index Scan vs Seq Scan
-- - Estimated vs actual rows
-- - Execution time
```

---

## Q10: What is the N+1 query problem and how to solve it?

**Answer**:

```mermaid
graph TB
    A[N+1 Problem] --> B[1 Query<br/>Get main records]
    A --> C[N Queries<br/>Get related records]
    A --> D[Performance Issue<br/>Too many queries]
    
    style A fill:#FF6B6B
```

### Problem Example

```javascript
// ❌ N+1 Problem
async function getPostsWithAuthors() {
  // 1 query: Get all posts
  const posts = await db.query('SELECT * FROM posts LIMIT 10');
  
  // N queries: Get author for each post
  for (const post of posts) {
    post.author = await db.query(
      'SELECT * FROM users WHERE id = $1',
      [post.author_id]
    );
  }
  
  return posts;
}

// Total: 1 + 10 = 11 queries!
```

### Solution 1: JOIN

```javascript
// ✅ Single query with JOIN
async function getPostsWithAuthors() {
  const result = await db.query(`
    SELECT 
      posts.*,
      users.id as author_id,
      users.name as author_name,
      users.email as author_email
    FROM posts
    JOIN users ON posts.author_id = users.id
    LIMIT 10
  `);
  
  return result.rows;
}

// Total: 1 query
```

### Solution 2: Eager Loading

```javascript
// ✅ Load all authors at once
async function getPostsWithAuthors() {
  // 1 query: Get posts
  const posts = await db.query('SELECT * FROM posts LIMIT 10');
  
  // 1 query: Get all authors
  const authorIds = posts.map(p => p.author_id);
  const authors = await db.query(
    'SELECT * FROM users WHERE id = ANY($1)',
    [authorIds]
  );
  
  // Map authors to posts
  const authorMap = new Map(authors.rows.map(a => [a.id, a]));
  posts.forEach(post => {
    post.author = authorMap.get(post.author_id);
  });
  
  return posts;
}

// Total: 2 queries (much better!)
```

### Solution 3: DataLoader (Batching)

```javascript
const DataLoader = require('dataloader');

// Create DataLoader
const userLoader = new DataLoader(async (userIds) => {
  const users = await db.query(
    'SELECT * FROM users WHERE id = ANY($1)',
    [userIds]
  );
  
  const userMap = new Map(users.rows.map(u => [u.id, u]));
  return userIds.map(id => userMap.get(id));
});

// Usage
async function getPostsWithAuthors() {
  const posts = await db.query('SELECT * FROM posts LIMIT 10');
  
  // DataLoader batches these into single query
  await Promise.all(
    posts.map(async (post) => {
      post.author = await userLoader.load(post.author_id);
    })
  );
  
  return posts;
}
```

### ORM Example (Sequelize)

```javascript
// ❌ N+1 with ORM
const posts = await Post.findAll();
for (const post of posts) {
  post.author = await post.getAuthor(); // N queries
}

// ✅ Eager loading with ORM
const posts = await Post.findAll({
  include: [{ model: User, as: 'author' }]
});
```

---

## Summary

Medium backend topics:
- **Microservices**: Architecture, trade-offs, communication
- **ACID**: Transactions, isolation levels, concurrency
- **Message Queues**: Async processing, decoupling, reliability
- **Replication & Sharding**: Read/write scaling, strategies
- **Rate Limiting**: Algorithms, Redis implementation
- **Connection Pooling**: Performance, leak prevention
- **API Versioning**: Strategies, deprecation
- **Connection Leaks**: Detection, prevention
- **Index Strategies**: Composite, partial, covering
- **N+1 Problem**: Detection, solutions, DataLoader

These concepts enable building scalable, performant backend systems.

