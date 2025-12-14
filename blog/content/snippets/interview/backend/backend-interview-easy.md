---
title: "Backend Interview Questions - Easy"
date: 2025-12-14
tags: ["backend", "interview", "easy", "api", "database", "http"]
---

Easy-level backend interview questions covering HTTP, REST APIs, databases, and server fundamentals.

## Q1: What is REST and what are its principles?

**Answer**:

```mermaid
graph TB
    A[REST<br/>Representational State Transfer] --> B[Stateless<br/>No session state]
    A --> C[Client-Server<br/>Separation]
    A --> D[Cacheable<br/>Responses]
    A --> E[Uniform Interface<br/>Standard methods]
    
    style A fill:#FFD700
```

### REST Principles

**1. Stateless**: Each request contains all information needed
**2. Client-Server**: Separation of concerns
**3. Cacheable**: Responses can be cached
**4. Uniform Interface**: Standard HTTP methods
**5. Layered System**: Client doesn't know if connected to end server
**6. Code on Demand** (optional): Server can send executable code

### HTTP Methods (CRUD)

```mermaid
graph LR
    A[CRUD] --> B[Create → POST]
    A --> C[Read → GET]
    A --> D[Update → PUT/PATCH]
    A --> E[Delete → DELETE]
    
    style A fill:#FFD700
```

**Example REST API**:
```
GET    /api/users          # List all users
GET    /api/users/123      # Get user by ID
POST   /api/users          # Create new user
PUT    /api/users/123      # Update user (full)
PATCH  /api/users/123      # Update user (partial)
DELETE /api/users/123      # Delete user
```

---

## Q2: What are HTTP status codes and their meanings?

**Answer**:

```mermaid
graph TB
    A[HTTP Status Codes] --> B[1xx Informational]
    A --> C[2xx Success]
    A --> D[3xx Redirection]
    A --> E[4xx Client Error]
    A --> F[5xx Server Error]
    
    style A fill:#FFD700
    style C fill:#90EE90
    style E fill:#FFD700
    style F fill:#FF6B6B
```

### Common Status Codes

**2xx Success**:
- `200 OK`: Request succeeded
- `201 Created`: Resource created
- `204 No Content`: Success, no body

**3xx Redirection**:
- `301 Moved Permanently`: Resource moved
- `302 Found`: Temporary redirect
- `304 Not Modified`: Use cached version

**4xx Client Error**:
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: No permission
- `404 Not Found`: Resource doesn't exist
- `429 Too Many Requests`: Rate limited

**5xx Server Error**:
- `500 Internal Server Error`: Server error
- `502 Bad Gateway`: Invalid response from upstream
- `503 Service Unavailable`: Server overloaded
- `504 Gateway Timeout`: Upstream timeout

---

## Q3: What is the difference between SQL and NoSQL databases?

**Answer**:

```mermaid
graph TB
    subgraph SQL["SQL (Relational)"]
        S1[Structured Schema]
        S2[Tables & Rows]
        S3[ACID Transactions]
        S4[Vertical Scaling]
    end
    
    subgraph NoSQL["NoSQL (Non-Relational)"]
        N1[Flexible Schema]
        N2[Documents/Key-Value]
        N3[BASE/Eventual Consistency]
        N4[Horizontal Scaling]
    end
```

### SQL Example (PostgreSQL)

```sql
-- Create table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert
INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');

-- Query
SELECT * FROM users WHERE email = 'alice@example.com';

-- Join
SELECT orders.*, users.name 
FROM orders 
JOIN users ON orders.user_id = users.id;
```

### NoSQL Example (MongoDB)

```javascript
// Insert document
db.users.insertOne({
  name: "Alice",
  email: "alice@example.com",
  created_at: new Date(),
  preferences: {
    theme: "dark",
    notifications: true
  },
  tags: ["premium", "verified"]
});

// Query
db.users.findOne({ email: "alice@example.com" });

// Update
db.users.updateOne(
  { email: "alice@example.com" },
  { $set: { "preferences.theme": "light" } }
);
```

### When to Use Each

```mermaid
graph TB
    A{Choose Database} --> B[Complex Relationships?]
    A --> C[Need Transactions?]
    A --> D[Flexible Schema?]
    A --> E[Massive Scale?]
    
    B -->|Yes| F[SQL]
    C -->|Yes| F
    D -->|Yes| G[NoSQL]
    E -->|Yes| G
    
    style F fill:#87CEEB
    style G fill:#90EE90
```

---

## Q4: What is database indexing and why is it important?

**Answer**:

```mermaid
graph TB
    A[Database Index] --> B[Faster Queries<br/>O log n vs O n]
    A --> C[Trade-off<br/>Storage & write speed]
    A --> D[B-Tree Structure<br/>Sorted data]
    
    style A fill:#FFD700
```

### Without Index

```mermaid
sequenceDiagram
    participant Q as Query
    participant DB as Database
    
    Q->>DB: SELECT * FROM users WHERE email = 'alice@example.com'
    
    Note over DB: Full table scan
    DB->>DB: Check row 1
    DB->>DB: Check row 2
    DB->>DB: Check row 3
    DB->>DB: ... (1 million rows)
    
    DB->>Q: Result (slow)
```

### With Index

```mermaid
sequenceDiagram
    participant Q as Query
    participant I as Index
    participant DB as Database
    
    Q->>I: SELECT * FROM users WHERE email = 'alice@example.com'
    
    Note over I: Binary search in index
    I->>I: Navigate B-tree
    I->>DB: Get row at position X
    
    DB->>Q: Result (fast)
```

### Creating Indexes

```sql
-- Create index
CREATE INDEX idx_users_email ON users(email);

-- Composite index
CREATE INDEX idx_users_name_email ON users(name, email);

-- Unique index
CREATE UNIQUE INDEX idx_users_email_unique ON users(email);

-- Check query plan
EXPLAIN SELECT * FROM users WHERE email = 'alice@example.com';
```

**Trade-offs**:
- ✅ Faster reads
- ❌ Slower writes (index must be updated)
- ❌ More storage space

---

## Q5: What is authentication vs authorization?

**Answer**:

```mermaid
graph TB
    A[Authentication] --> B[Who are you?<br/>Identity verification]
    C[Authorization] --> D[What can you do?<br/>Permission check]
    
    B --> E[Login credentials]
    D --> F[Access control]
    
    style A fill:#87CEEB
    style C fill:#90EE90
```

### Authentication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant S as Server
    participant DB as Database
    
    U->>S: POST /login<br/>{email, password}
    S->>DB: Find user by email
    DB->>S: User data
    S->>S: Verify password hash
    
    alt Valid credentials
        S->>S: Generate JWT token
        S->>U: 200 OK {token}
    else Invalid credentials
        S->>U: 401 Unauthorized
    end
```

### Authorization Flow

```mermaid
sequenceDiagram
    participant U as User
    participant S as Server
    participant DB as Database
    
    U->>S: GET /admin/users<br/>Authorization: Bearer <token>
    S->>S: Verify token
    S->>S: Extract user ID
    S->>DB: Get user role
    DB->>S: Role: "admin"
    
    alt Has permission
        S->>U: 200 OK {users}
    else No permission
        S->>U: 403 Forbidden
    end
```

### Implementation Example

```go
package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/golang-jwt/jwt/v5"
)

type contextKey string

const userContextKey contextKey = "user"

// User represents authenticated user
type User struct {
	ID    string `json:"id"`
	Email string `json:"email"`
	Role  string `json:"role"`
}

// AuthMiddleware validates JWT token
func AuthMiddleware(secretKey string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			authHeader := r.Header.Get("Authorization")
			if authHeader == "" {
				http.Error(w, `{"error":"No token provided"}`, http.StatusUnauthorized)
				return
			}

			parts := strings.Split(authHeader, " ")
			if len(parts) != 2 || parts[0] != "Bearer" {
				http.Error(w, `{"error":"Invalid authorization header"}`, http.StatusUnauthorized)
				return
			}

			token, err := jwt.Parse(parts[1], func(token *jwt.Token) (interface{}, error) {
				return []byte(secretKey), nil
			})

			if err != nil || !token.Valid {
				http.Error(w, `{"error":"Invalid token"}`, http.StatusUnauthorized)
				return
			}

			claims, ok := token.Claims.(jwt.MapClaims)
			if !ok {
				http.Error(w, `{"error":"Invalid token claims"}`, http.StatusUnauthorized)
				return
			}

			user := &User{
				ID:    claims["id"].(string),
				Email: claims["email"].(string),
				Role:  claims["role"].(string),
			}

			ctx := context.WithValue(r.Context(), userContextKey, user)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// AuthorizeMiddleware checks user roles
func AuthorizeMiddleware(allowedRoles ...string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			user, ok := r.Context().Value(userContextKey).(*User)
			if !ok {
				http.Error(w, `{"error":"Not authenticated"}`, http.StatusUnauthorized)
				return
			}

			allowed := false
			for _, role := range allowedRoles {
				if user.Role == role {
					allowed = true
					break
				}
			}

			if !allowed {
				http.Error(w, `{"error":"Insufficient permissions"}`, http.StatusForbidden)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// Usage
func main() {
	mux := http.NewServeMux()

	// Protected admin endpoint
	adminHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"message": "Admin access granted"})
	})

	mux.Handle("/admin/users",
		AuthMiddleware("your-secret-key")(
			AuthorizeMiddleware("admin")(adminHandler),
		),
	)

	http.ListenAndServe(":8080", mux)
}
```

---

## Q6: What is JWT (JSON Web Token)?

**Answer**:

```mermaid
graph TB
    A[JWT] --> B[Header<br/>Algorithm & type]
    A --> C[Payload<br/>Claims/data]
    A --> D[Signature<br/>Verification]
    
    style A fill:#FFD700
```

### JWT Structure

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

[Header].[Payload].[Signature]
```

**Header**:
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

**Payload**:
```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022,
  "exp": 1516242622
}
```

**Signature**:
```
HMACSHA256(
  base64UrlEncode(header) + "." + base64UrlEncode(payload),
  secret
)
```

### Creating and Verifying JWT

```go
package main

import (
	"errors"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

type Claims struct {
	UserID string `json:"userId"`
	Email  string `json:"email"`
	Role   string `json:"role"`
	jwt.RegisteredClaims
}

// CreateToken generates a JWT token
func CreateToken(user User, secretKey string) (string, error) {
	claims := Claims{
		UserID: user.ID,
		Email:  user.Email,
		Role:   user.Role,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(secretKey))
}

// VerifyToken validates and decodes a JWT token
func VerifyToken(tokenString, secretKey string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		return []byte(secretKey), nil
	})

	if err != nil {
		return nil, err
	}

	if !token.Valid {
		return nil, errors.New("invalid token")
	}

	claims, ok := token.Claims.(*Claims)
	if !ok {
		return nil, errors.New("invalid token claims")
	}

	return claims, nil
}

// Login handler example
func LoginHandler(w http.ResponseWriter, r *http.Request) {
	// Authenticate user (check credentials)
	user := User{
		ID:    "123",
		Email: "user@example.com",
		Role:  "admin",
	}

	token, err := CreateToken(user, "your-secret-key")
	if err != nil {
		http.Error(w, "Failed to create token", http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(map[string]string{
		"token": token,
	})
}
```

---

## Q7: What is CORS and why is it needed?

**Answer**:

```mermaid
graph TB
    A[CORS<br/>Cross-Origin Resource Sharing] --> B[Security Mechanism<br/>Browser protection]
    A --> C[Same-Origin Policy<br/>Default restriction]
    A --> D[Allow Cross-Origin<br/>Explicit permission]
    
    style A fill:#FFD700
```

### Same-Origin Policy

```mermaid
graph LR
    A[https://example.com] -->|✓ Same origin| B[https://example.com/api]
    A -->|✗ Different port| C[https://example.com:8080]
    A -->|✗ Different domain| D[https://api.example.com]
    A -->|✗ Different protocol| E[http://example.com]
    
    style B fill:#90EE90
    style C fill:#FF6B6B
    style D fill:#FF6B6B
    style E fill:#FF6B6B
```

### CORS Flow

```mermaid
sequenceDiagram
    participant B as Browser
    participant S as Server
    
    Note over B: Frontend at example.com
    Note over S: API at api.example.com
    
    B->>S: OPTIONS /api/users<br/>Origin: https://example.com
    
    S->>B: 200 OK<br/>Access-Control-Allow-Origin: https://example.com<br/>Access-Control-Allow-Methods: GET, POST
    
    B->>S: GET /api/users<br/>Origin: https://example.com
    
    S->>B: 200 OK<br/>Access-Control-Allow-Origin: https://example.com<br/>{users}
```

### Implementing CORS

```go
package main

import (
	"net/http"
	"strings"
)

// CORSMiddleware handles CORS headers
func CORSMiddleware(allowedOrigins []string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")

			// Check if origin is allowed
			allowed := false
			for _, allowedOrigin := range allowedOrigins {
				if allowedOrigin == "*" || allowedOrigin == origin {
					allowed = true
					break
				}
			}

			if allowed {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
				w.Header().Set("Access-Control-Allow-Credentials", "true")
			}

			// Handle preflight requests
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// Usage
func main() {
	mux := http.NewServeMux()

	// Your handlers
	mux.HandleFunc("/api/users", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"message": "Users endpoint"})
	})

	// Wrap with CORS middleware
	handler := CORSMiddleware([]string{"https://example.com"})(mux)

	http.ListenAndServe(":8080", handler)
}

// Simple CORS middleware (allow all)
func SimpleCORSMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}
```

---

## Q8: What is the difference between GET and POST?

**Answer**:

```mermaid
graph TB
    subgraph GET
        G1[Retrieve data]
        G2[Parameters in URL]
        G3[Cacheable]
        G4[Idempotent]
        G5[Bookmarkable]
    end
    
    subgraph POST
        P1[Submit data]
        P2[Parameters in body]
        P3[Not cacheable]
        P4[Not idempotent]
        P5[Not bookmarkable]
    end
```

### GET Request

```http
GET /api/users?page=1&limit=10 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
```

```go
package main

import (
	"encoding/json"
	"net/http"
	"strconv"
)

// GET handler
func GetUsersHandler(w http.ResponseWriter, r *http.Request) {
	// Parse query parameters
	pageStr := r.URL.Query().Get("page")
	limitStr := r.URL.Query().Get("limit")

	page, _ := strconv.Atoi(pageStr)
	limit, _ := strconv.Atoi(limitStr)

	if page == 0 {
		page = 1
	}
	if limit == 0 {
		limit = 10
	}

	// Fetch users from database
	users := fetchUsers(page, limit)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"users": users,
		"page":  page,
		"limit": limit,
	})
}
```

### POST Request

```http
POST /api/users HTTP/1.1
Host: api.example.com
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "Alice",
  "email": "alice@example.com"
}
```

```go
package main

import (
	"encoding/json"
	"net/http"
)

type CreateUserRequest struct {
	Name  string `json:"name"`
	Email string `json:"email"`
}

type UserResponse struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Email string `json:"email"`
}

// POST handler
func CreateUserHandler(w http.ResponseWriter, r *http.Request) {
	var req CreateUserRequest

	// Decode JSON body
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate
	if req.Name == "" || req.Email == "" {
		http.Error(w, "Name and email are required", http.StatusBadRequest)
		return
	}

	// Create user in database
	user := createUser(req.Name, req.Email)

	w.Header().Set("Content-Type", "application/json")
	w.WriteStatus(http.StatusCreated)
	json.NewEncoder(w).Encode(UserResponse{
		ID:    user.ID,
		Name:  user.Name,
		Email: user.Email,
	})
}
```

### Comparison

| Feature | GET | POST |
|---------|-----|------|
| Purpose | Retrieve | Submit |
| Data location | URL | Body |
| Cacheable | Yes | No |
| Idempotent | Yes | No |
| Size limit | Yes (~2KB) | No |
| Visible in URL | Yes | No |
| Bookmarkable | Yes | No |

---

## Q9: What is database normalization?

**Answer**:

```mermaid
graph TB
    A[Normalization] --> B[Reduce Redundancy<br/>Eliminate duplicates]
    A --> C[Improve Integrity<br/>Consistent data]
    A --> D[Normal Forms<br/>1NF, 2NF, 3NF]
    
    style A fill:#FFD700
```

### Unnormalized Data

```sql
-- ❌ Redundant data
CREATE TABLE orders (
    order_id INT,
    customer_name VARCHAR(100),
    customer_email VARCHAR(100),
    customer_address TEXT,
    product_name VARCHAR(100),
    product_price DECIMAL,
    quantity INT
);

-- Problems:
-- 1. Customer data repeated for each order
-- 2. Product data repeated for each order
-- 3. Update anomalies (change email in multiple places)
```

### Normalized Data (3NF)

```sql
-- ✅ Separate tables, no redundancy

CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    address TEXT
);

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    order_date TIMESTAMP DEFAULT NOW()
);

CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(order_id),
    product_id INT REFERENCES products(product_id),
    quantity INT,
    price_at_purchase DECIMAL
);
```

### Normal Forms

**1NF (First Normal Form)**:
- Atomic values (no lists in cells)
- No repeating groups

**2NF (Second Normal Form)**:
- 1NF + No partial dependencies
- Non-key attributes depend on entire primary key

**3NF (Third Normal Form)**:
- 2NF + No transitive dependencies
- Non-key attributes depend only on primary key

---

## Q10: What is caching and common caching strategies?

**Answer**:

```mermaid
graph TB
    A[Caching] --> B[Faster Response<br/>Reduce load]
    A --> C[Store Frequently<br/>Accessed Data]
    A --> D[Multiple Levels<br/>Browser, CDN, Server]
    
    style A fill:#FFD700
```

### Cache Layers

```mermaid
graph TB
    A[User] --> B[Browser Cache]
    B --> C[CDN Cache]
    C --> D[Application Cache<br/>Redis, Memcached]
    D --> E[Database]
    
    style B fill:#FFE4B5
    style C fill:#FFD700
    style D fill:#87CEEB
    style E fill:#90EE90
```

### Caching Strategies

**1. Cache-Aside (Lazy Loading)**:
```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

type User struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Email string `json:"email"`
}

func GetUser(ctx context.Context, rdb *redis.Client, userID string) (*User, error) {
	// Check cache first
	cacheKey := fmt.Sprintf("user:%s", userID)
	cached, err := rdb.Get(ctx, cacheKey).Result()

	if err == nil {
		// Cache hit
		var user User
		if err := json.Unmarshal([]byte(cached), &user); err == nil {
			return &user, nil
		}
	}

	// Cache miss: fetch from database
	user, err := fetchUserFromDB(userID)
	if err != nil {
		return nil, err
	}

	// Store in cache
	userData, _ := json.Marshal(user)
	rdb.Set(ctx, cacheKey, userData, time.Hour)

	return user, nil
}
```

**2. Write-Through**:
```go
func UpdateUser(ctx context.Context, rdb *redis.Client, userID string, data User) (*User, error) {
	// Update database
	user, err := updateUserInDB(userID, data)
	if err != nil {
		return nil, err
	}

	// Update cache
	cacheKey := fmt.Sprintf("user:%s", userID)
	userData, _ := json.Marshal(user)
	rdb.Set(ctx, cacheKey, userData, time.Hour)

	return user, nil
}
```

**3. Write-Behind (Write-Back)**:
```go
func UpdateUserAsync(ctx context.Context, rdb *redis.Client, queue Queue, userID string, data User) error {
	// Update cache immediately
	cacheKey := fmt.Sprintf("user:%s", userID)
	userData, _ := json.Marshal(data)
	if err := rdb.Set(ctx, cacheKey, userData, time.Hour).Err(); err != nil {
		return err
	}

	// Queue database update (async)
	return queue.Enqueue("update-user", map[string]interface{}{
		"userId": userID,
		"data":   data,
	})
}
```

### Cache Invalidation

```mermaid
graph TB
    A[Cache Invalidation] --> B[TTL<br/>Time-based expiry]
    A --> C[Manual<br/>Explicit delete]
    A --> D[Event-based<br/>On data change]
    
    style A fill:#FFD700
```

```go
package main

import (
	"context"
	"time"

	"github.com/redis/go-redis/v9"
)

func CacheInvalidationExamples(ctx context.Context, rdb *redis.Client) {
	// TTL (Time To Live)
	rdb.Set(ctx, "key", "value", time.Hour) // Expires in 1 hour

	// Manual invalidation
	rdb.Del(ctx, "user:123")

	// Pattern-based invalidation
	keys, _ := rdb.Keys(ctx, "user:*").Result()
	if len(keys) > 0 {
		rdb.Del(ctx, keys...)
	}

	// Or use SCAN for large datasets (more efficient)
	iter := rdb.Scan(ctx, 0, "user:*", 0).Iterator()
	for iter.Next(ctx) {
		rdb.Del(ctx, iter.Val())
	}
}
```

---

## Summary

Key backend concepts:
- **REST**: Stateless, uniform interface, HTTP methods
- **HTTP Status Codes**: 2xx success, 4xx client error, 5xx server error
- **SQL vs NoSQL**: Relational vs document, ACID vs BASE
- **Indexing**: B-tree, faster queries, trade-offs
- **Authentication vs Authorization**: Identity vs permissions
- **JWT**: Token-based auth, stateless
- **CORS**: Cross-origin security, browser protection
- **GET vs POST**: Retrieve vs submit, idempotent vs not
- **Normalization**: Reduce redundancy, normal forms
- **Caching**: Strategies, invalidation, layers

These fundamentals are essential for backend development.

