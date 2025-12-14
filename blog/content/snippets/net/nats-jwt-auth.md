---
title: "NATS Setup with JWT Authorization"
date: 2024-12-12T22:00:00Z
draft: false
description: "NATS server setup and JWT-based authorization cheatsheet"
type: "snippet"
tags: ["nats", "messaging", "jwt", "authorization", "microservices", "nats-knowhow"]
category: "nats"
---



NATS is a high-performance messaging system. This guide covers setup and JWT-based authorization using nsc (NATS Security CLI) for secure, decentralized authentication.

## Installation

```bash
# Install NATS server
# macOS
brew install nats-server

# Linux
wget https://github.com/nats-io/nats-server/releases/latest/download/nats-server-linux-amd64.zip
unzip nats-server-linux-amd64.zip
sudo mv nats-server /usr/local/bin/

# Install nsc (NATS Security CLI)
curl -L https://raw.githubusercontent.com/nats-io/nsc/master/install.sh | sh

# Install nats CLI
go install github.com/nats-io/natscli/nats@latest
```

---

## Docker Setup

### Docker Run

```bash
# Run NATS server
docker run -d \
  --name nats \
  -p 4222:4222 \
  -p 8222:8222 \
  -p 6222:6222 \
  nats:latest

# With JetStream enabled
docker run -d \
  --name nats-js \
  -p 4222:4222 \
  -p 8222:8222 \
  -v nats-data:/data \
  nats:latest -js

# Connect
docker exec -it nats nats-cli
```

### Docker Compose

```yaml
version: '3.8'

services:
  nats:
    image: nats:latest
    container_name: nats
    command: 
      - "-js"                    # Enable JetStream
      - "-m=8222"                # Monitoring port
      - "-sd=/data"              # Store directory
    ports:
      - "4222:4222"              # Client connections
      - "8222:8222"              # HTTP monitoring
      - "6222:6222"              # Cluster routing
    volumes:
      - nats-data:/data
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost:8222/healthz"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # NATS with JWT auth
  nats-auth:
    image: nats:latest
    container_name: nats-auth
    command:
      - "-c=/config/nats-server.conf"
    ports:
      - "4223:4222"
      - "8223:8222"
    volumes:
      - ./nats-config:/config
      - nats-auth-data:/data
    restart: unless-stopped

  # NATS Surveyor (monitoring UI)
  nats-surveyor:
    image: natsio/nats-surveyor:latest
    container_name: nats-surveyor
    command:
      - "-s=http://nats:8222"
    ports:
      - "7777:7777"
    depends_on:
      - nats
    restart: unless-stopped

volumes:
  nats-data:
  nats-auth-data:
```

**nats-server.conf** (for JWT auth):

```conf
# NATS Server Configuration with JWT

port: 4222
http_port: 8222

# JetStream
jetstream {
    store_dir: /data
    max_memory_store: 1GB
    max_file_store: 10GB
}

# JWT Authorization
operator: /config/operator.jwt

resolver: {
    type: full
    dir: /config/jwt
    allow_delete: false
    interval: "2m"
}

# System account
system_account: SYS

# Logging
debug: false
trace: false
logtime: true
```

### Docker Compose with Multiple Services

```yaml
version: '3.8'

services:
  # NATS Server
  nats:
    image: nats:latest
    container_name: nats
    command: ["-js", "-m=8222"]
    ports:
      - "4222:4222"
      - "8222:8222"
    volumes:
      - nats-data:/data
    networks:
      - app-network
    restart: unless-stopped

  # Publisher service
  publisher:
    build: ./publisher
    container_name: publisher
    environment:
      NATS_URL: nats://nats:4222
    depends_on:
      - nats
    networks:
      - app-network
    restart: unless-stopped

  # Subscriber service
  subscriber:
    build: ./subscriber
    container_name: subscriber
    environment:
      NATS_URL: nats://nats:4222
    depends_on:
      - nats
    networks:
      - app-network
    restart: unless-stopped

volumes:
  nats-data:

networks:
  app-network:
    driver: bridge
```

---

## JWT Authorization Setup

### Step 1: Create Operator

```bash
# Initialize nsc
nsc init

# Create operator (top-level authority)
nsc add operator -n MyOperator

# Generate operator JWT
nsc describe operator
```

### Step 2: Create Account

```bash
# Create account (tenant/organization)
nsc add account -n MyAccount

# Set account limits (optional)
nsc edit account MyAccount \
    --max-connections 1000 \
    --max-data 10GB \
    --max-exports 10 \
    --max-imports 10 \
    --max-payload 1MB \
    --max-subscriptions 1000

# Describe account
nsc describe account MyAccount
```

### Step 3: Create Users

```bash
# Create user with permissions
nsc add user -n alice \
    --allow-pub "orders.>" \
    --allow-sub "orders.alice.>" \
    --allow-pub-response

# Create user with different permissions
nsc add user -n bob \
    --allow-pub "inventory.>" \
    --allow-sub "inventory.>" \
    --deny-pub "inventory.delete"

# Create admin user
nsc add user -n admin \
    --allow-pub ">" \
    --allow-sub ">"

# Generate user credentials
nsc generate creds -a MyAccount -n alice > alice.creds
nsc generate creds -a MyAccount -n bob > bob.creds
```

### Step 4: Configure NATS Server

```conf
# nats-server.conf
port: 4222
http_port: 8222

# JWT Authentication
operator: /path/to/operator.jwt
resolver: {
    type: full
    dir: '/path/to/jwt'
}

# System Account (for monitoring)
system_account: SYS

# Logging
debug: false
trace: false
logtime: true
log_file: "/var/log/nats/nats-server.log"

# Limits
max_connections: 10000
max_payload: 1MB
max_pending: 10MB
```

### Step 5: Push Account JWT to Server

```bash
# Push account JWT to server resolver
nsc push -a MyAccount

# Or manually copy JWT files
cp ~/.nsc/nats/MyOperator/accounts/MyAccount/*.jwt /path/to/jwt/
```

### Step 6: Start Server

```bash
# Start with config
nats-server -c nats-server.conf

# Or with operator JWT directly
nats-server --operator /path/to/operator.jwt
```

## Client Connection Examples

### Go Client

```go
package main

import (
    "log"
    "github.com/nats-io/nats.go"
)

func main() {
    // Connect with credentials
    nc, err := nats.Connect("nats://localhost:4222",
        nats.UserCredentials("alice.creds"),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer nc.Close()

    // Publish
    err = nc.Publish("orders.new", []byte("Order #123"))
    if err != nil {
        log.Fatal(err)
    }

    // Subscribe
    sub, err := nc.Subscribe("orders.alice.>", func(m *nats.Msg) {
        log.Printf("Received: %s", string(m.Data))
    })
    if err != nil {
        log.Fatal(err)
    }
    defer sub.Unsubscribe()

    // Keep alive
    select {}
}
```

### Python Client

```python
import asyncio
from nats.aio.client import Client as NATS

async def main():
    nc = NATS()
    
    # Connect with credentials
    await nc.connect(
        servers=["nats://localhost:4222"],
        user_credentials="alice.creds"
    )
    
    # Subscribe
    async def message_handler(msg):
        print(f"Received: {msg.data.decode()}")
    
    await nc.subscribe("orders.alice.>", cb=message_handler)
    
    # Publish
    await nc.publish("orders.new", b"Order #123")
    
    # Keep alive
    await asyncio.sleep(60)
    await nc.close()

if __name__ == '__main__':
    asyncio.run(main())
```

### CLI

```bash
# Subscribe
nats sub --creds=alice.creds "orders.alice.>"

# Publish
nats pub --creds=alice.creds "orders.new" "Order #123"

# Request-Reply
nats req --creds=alice.creds "orders.status" "123"
```

## Permission Patterns

### Subject-Based Permissions

```bash
# Allow all under namespace
--allow-pub "orders.>"
--allow-sub "orders.>"

# Specific subjects only
--allow-pub "orders.create"
--allow-pub "orders.update"

# Deny specific subjects
--deny-pub "orders.delete"

# Wildcards
--allow-sub "orders.*.status"  # Single token wildcard
--allow-sub "orders.>"          # Multi-token wildcard
```

### Response Permissions

```bash
# Allow publishing responses (for request-reply)
--allow-pub-response

# Limit response time
--response-ttl 5s
```

### Time-Based Permissions

```bash
# User expires after 30 days
--expiry 30d

# Bearer token (one-time use)
--bearer
```

## Account Limits

```bash
# Connection limits
--max-connections 100

# Data limits
--max-data 1GB
--max-payload 1MB

# Subscription limits
--max-subscriptions 1000

# Export/Import limits (for account-to-account communication)
--max-exports 10
--max-imports 10
```

## Monitoring

### Server Stats

```bash
# HTTP monitoring endpoint
curl http://localhost:8222/varz

# Connection stats
curl http://localhost:8222/connz

# Subscription stats
curl http://localhost:8222/subsz

# Account stats
curl http://localhost:8222/accountz
```

### System Account

```bash
# Create system account
nsc add account -n SYS
nsc add user -n sys --account SYS

# Subscribe to system events
nats sub --creds=sys.creds '$SYS.>'

# Account stats
nats req --creds=sys.creds '$SYS.REQ.ACCOUNT.<account-id>.CONNZ' ''
```

## Common Patterns

### Request-Reply

```go
// Server
nc.Subscribe("orders.status", func(m *nats.Msg) {
    status := getOrderStatus(string(m.Data))
    m.Respond([]byte(status))
})

// Client
msg, err := nc.Request("orders.status", []byte("123"), 2*time.Second)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Status: %s\n", msg.Data)
```

### Queue Groups (Load Balancing)

```go
// Multiple workers in same queue group
nc.QueueSubscribe("orders.process", "workers", func(m *nats.Msg) {
    // Only one worker receives each message
    processOrder(m.Data)
})
```

### JetStream (Persistence)

```go
// Enable JetStream
js, err := nc.JetStream()

// Create stream
js.AddStream(&nats.StreamConfig{
    Name:     "ORDERS",
    Subjects: []string{"orders.>"},
    Storage:  nats.FileStorage,
})

// Publish to stream
js.Publish("orders.new", []byte("Order #123"))

// Durable consumer
js.Subscribe("orders.>", func(m *nats.Msg) {
    m.Ack()
}, nats.Durable("order-processor"))
```

## Security Best Practices

1. **Least Privilege**: Grant minimum necessary permissions
2. **Credential Rotation**: Regularly rotate user credentials
3. **Expiry**: Set expiration on user JWTs
4. **Monitoring**: Monitor for unauthorized access attempts
5. **TLS**: Enable TLS for production
6. **Separate Accounts**: Use different accounts for different services

## TLS Configuration

```conf
# nats-server.conf with TLS
tls {
    cert_file: "/path/to/server-cert.pem"
    key_file: "/path/to/server-key.pem"
    ca_file: "/path/to/ca.pem"
    verify: true
}
```

## Notes

- JWT auth is decentralized - server doesn't need central auth service
- Credentials files contain both JWT and seed (private key)
- Operator signs accounts, accounts sign users
- Permissions are enforced at server level
- Use JetStream for guaranteed delivery

## Gotchas/Warnings

- ⚠️ **Credentials security**: Protect .creds files - they contain private keys
- ⚠️ **Subject design**: Plan subject namespace carefully
- ⚠️ **Wildcard permissions**: Be careful with `>` wildcard
- ⚠️ **Account limits**: Set appropriate limits to prevent abuse

## Custom Resolver Implementation

### Memory Resolver (Go)

```go
package main

import (
    "encoding/json"
    "fmt"
    "io/ioutil"
    "sync"
    
    "github.com/nats-io/jwt/v2"
    "github.com/nats-io/nats-server/v2/server"
)

// MemoryResolver stores account JWTs in memory
type MemoryResolver struct {
    mu       sync.RWMutex
    accounts map[string]string // account public key -> JWT
}

func NewMemoryResolver() *MemoryResolver {
    return &MemoryResolver{
        accounts: make(map[string]string),
    }
}

// Fetch implements the AccountResolver interface
func (mr *MemoryResolver) Fetch(name string) (string, error) {
    mr.mu.RLock()
    defer mr.mu.RUnlock()
    
    jwt, ok := mr.accounts[name]
    if !ok {
        return "", fmt.Errorf("account not found: %s", name)
    }
    return jwt, nil
}

// Store adds an account JWT to the resolver
func (mr *MemoryResolver) Store(name, jwt string) error {
    mr.mu.Lock()
    defer mr.mu.Unlock()
    
    mr.accounts[name] = jwt
    return nil
}

// Delete removes an account JWT
func (mr *MemoryResolver) Delete(name string) error {
    mr.mu.Lock()
    defer mr.mu.Unlock()
    
    delete(mr.accounts, name)
    return nil
}

// List returns all account public keys
func (mr *MemoryResolver) List() []string {
    mr.mu.RLock()
    defer mr.mu.RUnlock()
    
    keys := make([]string, 0, len(mr.accounts))
    for k := range mr.accounts {
        keys = append(keys, k)
    }
    return keys
}

// Usage with NATS Server
func main() {
    // Create custom resolver
    resolver := NewMemoryResolver()
    
    // Load account JWTs from files
    accountJWT, _ := ioutil.ReadFile("account.jwt")
    
    // Parse JWT to get public key
    claim, _ := jwt.DecodeAccountClaims(string(accountJWT))
    
    // Store in resolver
    resolver.Store(claim.Subject, string(accountJWT))
    
    // Configure NATS server with custom resolver
    opts := &server.Options{
        Port:     4222,
        HTTPPort: 8222,
    }
    
    // Set operator JWT
    operatorJWT, _ := ioutil.ReadFile("operator.jwt")
    opts.TrustedOperators = []*jwt.OperatorClaims{
        // Parse operator JWT
    }
    
    // Set custom resolver
    opts.AccountResolver = resolver
    
    // Start server
    s, err := server.NewServer(opts)
    if err != nil {
        panic(err)
    }
    
    s.Start()
    s.WaitForShutdown()
}
```

### HTTP Resolver (Go)

```go
package main

import (
    "fmt"
    "io/ioutil"
    "net/http"
    "time"
)

// HTTPResolver fetches account JWTs from HTTP endpoint
type HTTPResolver struct {
    baseURL string
    client  *http.Client
}

func NewHTTPResolver(baseURL string) *HTTPResolver {
    return &HTTPResolver{
        baseURL: baseURL,
        client: &http.Client{
            Timeout: 5 * time.Second,
        },
    }
}

// Fetch implements the AccountResolver interface
func (hr *HTTPResolver) Fetch(accountID string) (string, error) {
    url := fmt.Sprintf("%s/accounts/%s", hr.baseURL, accountID)
    
    resp, err := hr.client.Get(url)
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return "", fmt.Errorf("account not found: %s", accountID)
    }
    
    jwt, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return "", err
    }
    
    return string(jwt), nil
}

// HTTP Server for serving JWTs
type JWTServer struct {
    accounts map[string]string
}

func (s *JWTServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // Extract account ID from URL
    accountID := r.URL.Path[len("/accounts/"):]
    
    jwt, ok := s.accounts[accountID]
    if !ok {
        http.NotFound(w, r)
        return
    }
    
    w.Header().Set("Content-Type", "application/jwt")
    w.Write([]byte(jwt))
}

// Start JWT server
func startJWTServer() {
    server := &JWTServer{
        accounts: make(map[string]string),
    }
    
    // Load JWTs from files
    // server.accounts[accountID] = jwtString
    
    http.ListenAndServe(":8080", server)
}
```

### Database Resolver (Go with PostgreSQL)

```go
package main

import (
    "database/sql"
    "fmt"
    
    _ "github.com/lib/pq"
)

// DBResolver fetches account JWTs from database
type DBResolver struct {
    db *sql.DB
}

func NewDBResolver(connStr string) (*DBResolver, error) {
    db, err := sql.Open("postgres", connStr)
    if err != nil {
        return nil, err
    }
    
    // Create table if not exists
    _, err = db.Exec(`
        CREATE TABLE IF NOT EXISTS account_jwts (
            account_id VARCHAR(255) PRIMARY KEY,
            jwt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    `)
    if err != nil {
        return nil, err
    }
    
    return &DBResolver{db: db}, nil
}

// Fetch implements the AccountResolver interface
func (dr *DBResolver) Fetch(accountID string) (string, error) {
    var jwt string
    err := dr.db.QueryRow(
        "SELECT jwt FROM account_jwts WHERE account_id = $1",
        accountID,
    ).Scan(&jwt)
    
    if err == sql.ErrNoRows {
        return "", fmt.Errorf("account not found: %s", accountID)
    }
    if err != nil {
        return "", err
    }
    
    return jwt, nil
}

// Store adds or updates an account JWT
func (dr *DBResolver) Store(accountID, jwt string) error {
    _, err := dr.db.Exec(`
        INSERT INTO account_jwts (account_id, jwt)
        VALUES ($1, $2)
        ON CONFLICT (account_id)
        DO UPDATE SET jwt = $2, updated_at = CURRENT_TIMESTAMP
    `, accountID, jwt)
    
    return err
}

// Delete removes an account JWT
func (dr *DBResolver) Delete(accountID string) error {
    _, err := dr.db.Exec(
        "DELETE FROM account_jwts WHERE account_id = $1",
        accountID,
    )
    return err
}

// List returns all account IDs
func (dr *DBResolver) List() ([]string, error) {
    rows, err := dr.db.Query("SELECT account_id FROM account_jwts")
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var accounts []string
    for rows.Next() {
        var accountID string
        if err := rows.Scan(&accountID); err != nil {
            return nil, err
        }
        accounts = append(accounts, accountID)
    }
    
    return accounts, nil
}
```

### Cached Resolver (Go)

```go
package main

import (
    "sync"
    "time"
)

// CachedResolver wraps another resolver with caching
type CachedResolver struct {
    underlying AccountResolver
    cache      map[string]*cacheEntry
    mu         sync.RWMutex
    ttl        time.Duration
}

type cacheEntry struct {
    jwt       string
    expiresAt time.Time
}

func NewCachedResolver(underlying AccountResolver, ttl time.Duration) *CachedResolver {
    cr := &CachedResolver{
        underlying: underlying,
        cache:      make(map[string]*cacheEntry),
        ttl:        ttl,
    }
    
    // Start cleanup goroutine
    go cr.cleanup()
    
    return cr
}

// Fetch implements the AccountResolver interface with caching
func (cr *CachedResolver) Fetch(accountID string) (string, error) {
    // Check cache first
    cr.mu.RLock()
    entry, ok := cr.cache[accountID]
    cr.mu.RUnlock()
    
    if ok && time.Now().Before(entry.expiresAt) {
        return entry.jwt, nil
    }
    
    // Cache miss or expired, fetch from underlying resolver
    jwt, err := cr.underlying.Fetch(accountID)
    if err != nil {
        return "", err
    }
    
    // Update cache
    cr.mu.Lock()
    cr.cache[accountID] = &cacheEntry{
        jwt:       jwt,
        expiresAt: time.Now().Add(cr.ttl),
    }
    cr.mu.Unlock()
    
    return jwt, nil
}

// Invalidate removes an entry from cache
func (cr *CachedResolver) Invalidate(accountID string) {
    cr.mu.Lock()
    delete(cr.cache, accountID)
    cr.mu.Unlock()
}

// cleanup removes expired entries periodically
func (cr *CachedResolver) cleanup() {
    ticker := time.NewTicker(cr.ttl / 2)
    defer ticker.Stop()
    
    for range ticker.C {
        cr.mu.Lock()
        now := time.Now()
        for key, entry := range cr.cache {
            if now.After(entry.expiresAt) {
                delete(cr.cache, key)
            }
        }
        cr.mu.Unlock()
    }
}
```

### NATS Server Configuration with Custom Resolver

```conf
# nats-server.conf with custom resolver

port: 4222
http_port: 8222

# Operator JWT
operator: /path/to/operator.jwt

# Custom resolver (URL-based)
resolver: URL(http://localhost:8080/accounts/)

# Or use NATS-based resolver
# resolver: NATS(nats://resolver-server:4222)

# System account
system_account: SYS

# TLS
tls {
    cert_file: "/path/to/server-cert.pem"
    key_file: "/path/to/server-key.pem"
}

# Logging
debug: false
trace: false
logtime: true
```

### Testing Custom Resolver

```go
package main

import (
    "testing"
    "time"
)

func TestMemoryResolver(t *testing.T) {
    resolver := NewMemoryResolver()
    
    // Test Store and Fetch
    accountID := "ACCOUNT123"
    jwt := "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
    
    err := resolver.Store(accountID, jwt)
    if err != nil {
        t.Fatalf("Store failed: %v", err)
    }
    
    fetched, err := resolver.Fetch(accountID)
    if err != nil {
        t.Fatalf("Fetch failed: %v", err)
    }
    
    if fetched != jwt {
        t.Errorf("Expected %s, got %s", jwt, fetched)
    }
    
    // Test Delete
    err = resolver.Delete(accountID)
    if err != nil {
        t.Fatalf("Delete failed: %v", err)
    }
    
    _, err = resolver.Fetch(accountID)
    if err == nil {
        t.Error("Expected error after delete")
    }
}

func TestCachedResolver(t *testing.T) {
    underlying := NewMemoryResolver()
    underlying.Store("ACC1", "jwt1")
    
    cached := NewCachedResolver(underlying, 1*time.Second)
    
    // First fetch - cache miss
    jwt1, _ := cached.Fetch("ACC1")
    
    // Second fetch - cache hit
    jwt2, _ := cached.Fetch("ACC1")
    
    if jwt1 != jwt2 {
        t.Error("Cache not working")
    }
    
    // Wait for expiry
    time.Sleep(2 * time.Second)
    
    // Should fetch from underlying again
    jwt3, _ := cached.Fetch("ACC1")
    if jwt3 != jwt1 {
        t.Error("Expired cache not refreshed")
    }
}
```