---
title: "Authentication Methods"
date: 2024-12-12
draft: false
category: "auth"
tags: ["authentication", "session", "jwt", "oauth", "oidc", "saml"]
---

Comprehensive guide to authentication methods: sessions, JWT, OAuth 2.0, OIDC, and SAML.

## Authentication vs Authorization

**Authentication (AuthN)**: *Who are you?*
- Verifying identity
- Credentials: username/password, tokens, biometrics
- Result: User identity established

**Authorization (AuthZ)**: *What can you do?*
- Verifying permissions
- Access control: roles, permissions, policies
- Result: Access granted or denied

---

## 1. Session-Based Authentication

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant Server
    participant SessionStore
    
    User->>Browser: Enter credentials
    Browser->>Server: POST /login (username, password)
    Server->>Server: Validate credentials
    Server->>SessionStore: Create session
    SessionStore-->>Server: Session ID
    Server->>Browser: Set-Cookie: session_id=abc123
    Browser->>User: Login successful
    
    Note over Browser,Server: Subsequent requests
    
    User->>Browser: Access protected resource
    Browser->>Server: GET /profile (Cookie: session_id=abc123)
    Server->>SessionStore: Validate session
    SessionStore-->>Server: Session data
    Server->>Browser: Protected resource
    Browser->>User: Display profile
```

**Pros:**
- Simple to implement
- Server controls session lifecycle
- Can revoke sessions instantly

**Cons:**
- Requires server-side session storage
- Doesn't scale well horizontally
- CSRF vulnerability if not protected

**Use Case:** Traditional web applications, admin panels

---

## 2. Token-Based Authentication (JWT)

```mermaid
sequenceDiagram
    participant User
    participant Client
    participant Server
    
    User->>Client: Enter credentials
    Client->>Server: POST /login (username, password)
    Server->>Server: Validate credentials
    Server->>Server: Generate JWT
    Server->>Client: JWT token
    Client->>Client: Store JWT (localStorage/memory)
    Client->>User: Login successful
    
    Note over Client,Server: Subsequent requests
    
    User->>Client: Access protected resource
    Client->>Server: GET /api/data (Authorization: Bearer JWT)
    Server->>Server: Verify JWT signature
    Server->>Server: Validate claims (exp, iat)
    Server->>Client: Protected resource
    Client->>User: Display data
```

**JWT Structure:**

```text
header.payload.signature
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

**JWT Payload Example:**

```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "email": "john@example.com",
  "roles": ["user", "admin"],
  "iat": 1516239022,
  "exp": 1516242622
}
```

**Pros:**
- Stateless (no server-side storage)
- Scales horizontally
- Works across domains
- Mobile-friendly

**Cons:**
- Cannot revoke before expiry (use short TTL + refresh tokens)
- Larger than session IDs
- Vulnerable if stolen (store securely)

**Use Case:** APIs, microservices, SPAs, mobile apps

---

## 3. OAuth 2.0

```mermaid
sequenceDiagram
    participant User
    participant Client
    participant AuthServer as Authorization Server
    participant ResourceServer as Resource Server
    
    User->>Client: Click "Login with Google"
    Client->>AuthServer: Redirect to /authorize
    Note right of Client: client_id, redirect_uri, scope
    AuthServer->>User: Show login page
    User->>AuthServer: Enter credentials & authorize
    AuthServer->>Client: Redirect with auth code
    Client->>AuthServer: POST /token
    Note right of Client: code, client_id, client_secret
    AuthServer->>Client: Access token + Refresh token
    
    Note over Client,ResourceServer: Access protected resources
    
    Client->>ResourceServer: GET /api/data (Bearer token)
    ResourceServer->>ResourceServer: Validate token
    ResourceServer->>Client: Protected resource
    Client->>User: Display data
```

### OAuth 2.0 Grant Types

**1. Authorization Code** (most secure for web apps)

```text
Client → Redirect to Auth Server → User Authorizes → Auth Code → Exchange for Token
```

**2. Client Credentials** (machine-to-machine)

```text
Client ID + Secret → Access Token
```

**3. Resource Owner Password** (legacy, avoid)

```text
Username + Password → Access Token
```

**4. Implicit** (deprecated, use Authorization Code + PKCE)

**Use Case:** Third-party integrations (Login with Google, GitHub)

---

## 4. OpenID Connect (OIDC)

OAuth 2.0 + Identity Layer

```text
OAuth 2.0 Flow → Access Token + ID Token (JWT with user info)
```

**ID Token Claims:**

```json
{
  "iss": "https://auth.example.com",
  "sub": "user123",
  "aud": "client_id",
  "exp": 1516242622,
  "iat": 1516239022,
  "email": "user@example.com",
  "email_verified": true,
  "name": "John Doe"
}
```

**Use Case:** SSO, federated identity

---

## 5. SAML 2.0

XML-based authentication protocol

```text
Service Provider → Identity Provider → SAML Assertion → SP Validates
```

**Use Case:** Enterprise SSO, legacy systems

---

## Comparison Table

| Method | Stateless | Scalability | Revocation | Mobile | Complexity |
|--------|-----------|-------------|------------|--------|------------|
| **Session** | ❌ | ⭐⭐ | ✅ Instant | ⭐⭐ | ⭐ Low |
| **JWT** | ✅ | ⭐⭐⭐⭐⭐ | ❌ (use short TTL) | ⭐⭐⭐⭐⭐ | ⭐⭐ Medium |
| **OAuth 2.0** | ✅ | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ High |
| **OIDC** | ✅ | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ High |
| **SAML** | ✅ | ⭐⭐⭐ | ✅ | ⭐ | ⭐⭐⭐⭐⭐ Very High |

---

## Decision Tree

```text
Need third-party login? → OAuth 2.0 / OIDC
  |
  No
  ↓
Building API/Microservices? → JWT
  |
  No
  ↓
Traditional web app? → Sessions
  |
  No
  ↓
Enterprise SSO? → SAML / OIDC
```

