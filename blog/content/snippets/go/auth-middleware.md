---
title: "Go Authentication Middleware"
date: 2024-12-12
draft: false
category: "go"
tags: ["go-knowhow", "go", "auth", "middleware", "jwt", "oauth"]
---


Authentication and authorization middleware patterns for Go web applications. Includes JWT, OAuth2, Auth0, and CORS implementations.

## Use Case

- Protect API endpoints with authentication
- Implement role-based access control
- Integrate with OAuth providers (Auth0, Google, GitHub)
- Handle CORS for frontend applications

---

## JWT Middleware

### Basic JWT Middleware

```go
package middleware

import (
    "context"
    "net/http"
    "strings"
    
    "github.com/golang-jwt/jwt/v5"
)

type contextKey string

const UserContextKey contextKey = "user"

type Claims struct {
    UserID string   `json:"user_id"`
    Email  string   `json:"email"`
    Roles  []string `json:"roles"`
    jwt.RegisteredClaims
}

// JWTMiddleware validates JWT tokens
func JWTMiddleware(secret []byte) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            // Extract token from Authorization header
            authHeader := r.Header.Get("Authorization")
            if authHeader == "" {
                http.Error(w, "Missing authorization header", http.StatusUnauthorized)
                return
            }
            
            // Bearer token format: "Bearer <token>"
            parts := strings.Split(authHeader, " ")
            if len(parts) != 2 || parts[0] != "Bearer" {
                http.Error(w, "Invalid authorization header format", http.StatusUnauthorized)
                return
            }
            
            tokenString := parts[1]
            
            // Parse and validate token
            token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
                // Validate signing method
                if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
                    return nil, jwt.ErrSignatureInvalid
                }
                return secret, nil
            })
            
            if err != nil {
                http.Error(w, "Invalid token", http.StatusUnauthorized)
                return
            }
            
            if !token.Valid {
                http.Error(w, "Token is not valid", http.StatusUnauthorized)
                return
            }
            
            // Extract claims
            claims, ok := token.Claims.(*Claims)
            if !ok {
                http.Error(w, "Invalid token claims", http.StatusUnauthorized)
                return
            }
            
            // Add claims to context
            ctx := context.WithValue(r.Context(), UserContextKey, claims)
            next.ServeHTTP(w, r.WithContext(ctx))
        })
    }
}

// GetUserFromContext extracts user claims from context
func GetUserFromContext(ctx context.Context) (*Claims, bool) {
    claims, ok := ctx.Value(UserContextKey).(*Claims)
    return claims, ok
}

// RequireRoles middleware checks if user has required roles
func RequireRoles(roles ...string) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            claims, ok := GetUserFromContext(r.Context())
            if !ok {
                http.Error(w, "Unauthorized", http.StatusUnauthorized)
                return
            }
            
            // Check if user has any of the required roles
            hasRole := false
            for _, requiredRole := range roles {
                for _, userRole := range claims.Roles {
                    if userRole == requiredRole {
                        hasRole = true
                        break
                    }
                }
                if hasRole {
                    break
                }
            }
            
            if !hasRole {
                http.Error(w, "Forbidden", http.StatusForbidden)
                return
            }
            
            next.ServeHTTP(w, r)
        })
    }
}
```

### Usage Example

```go
package main

import (
    "encoding/json"
    "net/http"
    "time"
    
    "github.com/golang-jwt/jwt/v5"
    "github.com/gorilla/mux"
)

var jwtSecret = []byte("your-secret-key-change-this")

func main() {
    r := mux.NewRouter()
    
    // Public routes
    r.HandleFunc("/login", loginHandler).Methods("POST")
    
    // Protected routes
    api := r.PathPrefix("/api").Subrouter()
    api.Use(JWTMiddleware(jwtSecret))
    
    api.HandleFunc("/profile", profileHandler).Methods("GET")
    
    // Admin-only routes
    admin := api.PathPrefix("/admin").Subrouter()
    admin.Use(RequireRoles("admin"))
    admin.HandleFunc("/users", listUsersHandler).Methods("GET")
    
    http.ListenAndServe(":8080", r)
}

func loginHandler(w http.ResponseWriter, r *http.Request) {
    var creds struct {
        Email    string `json:"email"`
        Password string `json:"password"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&creds); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }
    
    // Validate credentials (check database)
    // This is a simplified example
    if creds.Email != "user@example.com" || creds.Password != "password" {
        http.Error(w, "Invalid credentials", http.StatusUnauthorized)
        return
    }
    
    // Create JWT token
    claims := &Claims{
        UserID: "user123",
        Email:  creds.Email,
        Roles:  []string{"user"},
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(15 * time.Minute)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
        },
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    tokenString, err := token.SignedString(jwtSecret)
    if err != nil {
        http.Error(w, "Error generating token", http.StatusInternalServerError)
        return
    }
    
    json.NewEncoder(w).Encode(map[string]string{
        "token": tokenString,
    })
}

func profileHandler(w http.ResponseWriter, r *http.Request) {
    claims, _ := GetUserFromContext(r.Context())
    json.NewEncoder(w).Encode(claims)
}

func listUsersHandler(w http.ResponseWriter, r *http.Request) {
    // Admin-only endpoint
    json.NewEncoder(w).Encode(map[string]string{
        "message": "List of users (admin only)",
    })
}
```

---

## Auth0 Integration

```go
package middleware

import (
    "context"
    "encoding/json"
    "errors"
    "net/http"
    "net/url"
    "strings"
    "time"
    
    "github.com/golang-jwt/jwt/v5"
)

type Auth0Config struct {
    Domain   string
    Audience string
}

type JWKS struct {
    Keys []JSONWebKey `json:"keys"`
}

type JSONWebKey struct {
    Kty string   `json:"kty"`
    Kid string   `json:"kid"`
    Use string   `json:"use"`
    N   string   `json:"n"`
    E   string   `json:"e"`
    X5c []string `json:"x5c"`
}

// Auth0Middleware validates Auth0 JWT tokens
func Auth0Middleware(config Auth0Config) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            // Extract token
            authHeader := r.Header.Get("Authorization")
            if authHeader == "" {
                http.Error(w, "Missing authorization header", http.StatusUnauthorized)
                return
            }
            
            parts := strings.Split(authHeader, " ")
            if len(parts) != 2 || parts[0] != "Bearer" {
                http.Error(w, "Invalid authorization header", http.StatusUnauthorized)
                return
            }
            
            tokenString := parts[1]
            
            // Parse token without validation to get kid
            token, _, err := new(jwt.Parser).ParseUnverified(tokenString, jwt.MapClaims{})
            if err != nil {
                http.Error(w, "Invalid token", http.StatusUnauthorized)
                return
            }
            
            // Get kid from token header
            kid, ok := token.Header["kid"].(string)
            if !ok {
                http.Error(w, "Invalid token header", http.StatusUnauthorized)
                return
            }
            
            // Fetch JWKS from Auth0
            jwks, err := fetchJWKS(config.Domain)
            if err != nil {
                http.Error(w, "Error fetching JWKS", http.StatusInternalServerError)
                return
            }
            
            // Find matching key
            var jwk *JSONWebKey
            for _, key := range jwks.Keys {
                if key.Kid == kid {
                    jwk = &key
                    break
                }
            }
            
            if jwk == nil {
                http.Error(w, "Unable to find appropriate key", http.StatusUnauthorized)
                return
            }
            
            // Parse and validate token with public key
            parsedToken, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
                // Verify signing method
                if token.Method.Alg() != "RS256" {
                    return nil, errors.New("unexpected signing method")
                }
                
                // Convert JWK to public key
                cert := "-----BEGIN CERTIFICATE-----\n" + jwk.X5c[0] + "\n-----END CERTIFICATE-----"
                return jwt.ParseRSAPublicKeyFromPEM([]byte(cert))
            })
            
            if err != nil || !parsedToken.Valid {
                http.Error(w, "Invalid token", http.StatusUnauthorized)
                return
            }
            
            // Validate claims
            claims, ok := parsedToken.Claims.(jwt.MapClaims)
            if !ok {
                http.Error(w, "Invalid claims", http.StatusUnauthorized)
                return
            }
            
            // Validate audience
            if !claims.VerifyAudience(config.Audience, true) {
                http.Error(w, "Invalid audience", http.StatusUnauthorized)
                return
            }
            
            // Validate issuer
            expectedIssuer := "https://" + config.Domain + "/"
            if !claims.VerifyIssuer(expectedIssuer, true) {
                http.Error(w, "Invalid issuer", http.StatusUnauthorized)
                return
            }
            
            // Add claims to context
            ctx := context.WithValue(r.Context(), UserContextKey, claims)
            next.ServeHTTP(w, r.WithContext(ctx))
        })
    }
}

func fetchJWKS(domain string) (*JWKS, error) {
    jwksURL := "https://" + domain + "/.well-known/jwks.json"
    
    client := &http.Client{Timeout: 10 * time.Second}
    resp, err := client.Get(jwksURL)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    var jwks JWKS
    if err := json.NewDecoder(resp.Body).Decode(&jwks); err != nil {
        return nil, err
    }
    
    return &jwks, nil
}

// Usage
func main() {
    auth0Config := Auth0Config{
        Domain:   "your-tenant.auth0.com",
        Audience: "https://your-api.com",
    }
    
    r := mux.NewRouter()
    
    api := r.PathPrefix("/api").Subrouter()
    api.Use(Auth0Middleware(auth0Config))
    api.HandleFunc("/protected", protectedHandler).Methods("GET")
    
    http.ListenAndServe(":8080", r)
}
```

---

## OAuth2 Middleware (Google, GitHub)

```go
package middleware

import (
    "context"
    "encoding/json"
    "net/http"
    
    "golang.org/x/oauth2"
    "golang.org/x/oauth2/google"
    "golang.org/x/oauth2/github"
)

type OAuthConfig struct {
    ClientID     string
    ClientSecret string
    RedirectURL  string
    Scopes       []string
}

// Google OAuth
func GoogleOAuthConfig(config OAuthConfig) *oauth2.Config {
    return &oauth2.Config{
        ClientID:     config.ClientID,
        ClientSecret: config.ClientSecret,
        RedirectURL:  config.RedirectURL,
        Scopes:       config.Scopes,
        Endpoint:     google.Endpoint,
    }
}

// GitHub OAuth
func GitHubOAuthConfig(config OAuthConfig) *oauth2.Config {
    return &oauth2.Config{
        ClientID:     config.ClientID,
        ClientSecret: config.ClientSecret,
        RedirectURL:  config.RedirectURL,
        Scopes:       config.Scopes,
        Endpoint:     github.Endpoint,
    }
}

// OAuth handlers
func OAuthLoginHandler(oauthConfig *oauth2.Config) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        // Generate random state for CSRF protection
        state := generateRandomState() // Implement this
        
        // Store state in session/cookie
        http.SetCookie(w, &http.Cookie{
            Name:     "oauth_state",
            Value:    state,
            MaxAge:   300, // 5 minutes
            HttpOnly: true,
            Secure:   true,
            SameSite: http.SameSiteLaxMode,
        })
        
        // Redirect to OAuth provider
        url := oauthConfig.AuthCodeURL(state)
        http.Redirect(w, r, url, http.StatusTemporaryRedirect)
    }
}

func OAuthCallbackHandler(oauthConfig *oauth2.Config) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        // Validate state (CSRF protection)
        stateCookie, err := r.Cookie("oauth_state")
        if err != nil {
            http.Error(w, "State cookie not found", http.StatusBadRequest)
            return
        }
        
        state := r.URL.Query().Get("state")
        if state != stateCookie.Value {
            http.Error(w, "Invalid state parameter", http.StatusBadRequest)
            return
        }
        
        // Exchange code for token
        code := r.URL.Query().Get("code")
        token, err := oauthConfig.Exchange(context.Background(), code)
        if err != nil {
            http.Error(w, "Failed to exchange token", http.StatusInternalServerError)
            return
        }
        
        // Fetch user info
        client := oauthConfig.Client(context.Background(), token)
        resp, err := client.Get("https://www.googleapis.com/oauth2/v2/userinfo") // For Google
        if err != nil {
            http.Error(w, "Failed to get user info", http.StatusInternalServerError)
            return
        }
        defer resp.Body.Close()
        
        var userInfo map[string]interface{}
        if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
            http.Error(w, "Failed to decode user info", http.StatusInternalServerError)
            return
        }
        
        // Create session or JWT token
        // Store user info in database
        // Redirect to application
        
        json.NewEncoder(w).Encode(userInfo)
    }
}

// Usage
func main() {
    googleConfig := GoogleOAuthConfig(OAuthConfig{
        ClientID:     "your-client-id",
        ClientSecret: "your-client-secret",
        RedirectURL:  "http://localhost:8080/auth/google/callback",
        Scopes:       []string{"email", "profile"},
    })
    
    r := mux.NewRouter()
    r.HandleFunc("/auth/google", OAuthLoginHandler(googleConfig)).Methods("GET")
    r.HandleFunc("/auth/google/callback", OAuthCallbackHandler(googleConfig)).Methods("GET")
    
    http.ListenAndServe(":8080", r)
}
```

---

## CORS Middleware

```go
package middleware

import (
    "net/http"
)

type CORSConfig struct {
    AllowedOrigins   []string
    AllowedMethods   []string
    AllowedHeaders   []string
    ExposedHeaders   []string
    AllowCredentials bool
    MaxAge           int
}

func DefaultCORSConfig() CORSConfig {
    return CORSConfig{
        AllowedOrigins:   []string{"*"},
        AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
        AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
        ExposedHeaders:   []string{"Link"},
        AllowCredentials: false,
        MaxAge:           300,
    }
}

func CORSMiddleware(config CORSConfig) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            origin := r.Header.Get("Origin")
            
            // Check if origin is allowed
            allowed := false
            for _, allowedOrigin := range config.AllowedOrigins {
                if allowedOrigin == "*" || allowedOrigin == origin {
                    allowed = true
                    break
                }
            }
            
            if !allowed {
                next.ServeHTTP(w, r)
                return
            }
            
            // Set CORS headers
            if origin != "" {
                w.Header().Set("Access-Control-Allow-Origin", origin)
            } else if len(config.AllowedOrigins) == 1 {
                w.Header().Set("Access-Control-Allow-Origin", config.AllowedOrigins[0])
            }
            
            w.Header().Set("Access-Control-Allow-Methods", strings.Join(config.AllowedMethods, ", "))
            w.Header().Set("Access-Control-Allow-Headers", strings.Join(config.AllowedHeaders, ", "))
            w.Header().Set("Access-Control-Expose-Headers", strings.Join(config.ExposedHeaders, ", "))
            
            if config.AllowCredentials {
                w.Header().Set("Access-Control-Allow-Credentials", "true")
            }
            
            if config.MaxAge > 0 {
                w.Header().Set("Access-Control-Max-Age", fmt.Sprintf("%d", config.MaxAge))
            }
            
            // Handle preflight requests
            if r.Method == "OPTIONS" {
                w.WriteHeader(http.StatusNoContent)
                return
            }
            
            next.ServeHTTP(w, r)
        })
    }
}

// Usage
func main() {
    corsConfig := CORSConfig{
        AllowedOrigins:   []string{"http://localhost:3000", "https://app.example.com"},
        AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE"},
        AllowedHeaders:   []string{"Authorization", "Content-Type"},
        AllowCredentials: true,
        MaxAge:           3600,
    }
    
    r := mux.NewRouter()
    r.Use(CORSMiddleware(corsConfig))
    
    r.HandleFunc("/api/data", dataHandler).Methods("GET", "POST")
    
    http.ListenAndServe(":8080", r)
}
```

---

## General Middleware Pattern

```go
package middleware

import (
    "log"
    "net/http"
    "time"
)

// Middleware type
type Middleware func(http.Handler) http.Handler

// Chain multiple middleware
func Chain(middlewares ...Middleware) Middleware {
    return func(final http.Handler) http.Handler {
        for i := len(middlewares) - 1; i >= 0; i-- {
            final = middlewares[i](final)
        }
        return final
    }
}

// Logging middleware
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Call next handler
        next.ServeHTTP(w, r)
        
        log.Printf("%s %s %s", r.Method, r.RequestURI, time.Since(start))
    })
}

// Recovery middleware
func RecoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("Panic: %v", err)
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        
        next.ServeHTTP(w, r)
    })
}

// Usage: Chain multiple middleware
func main() {
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Write([]byte("Hello, World!"))
    })
    
    // Apply middleware in order
    wrapped := Chain(
        RecoveryMiddleware,
        LoggingMiddleware,
        CORSMiddleware(DefaultCORSConfig()),
        JWTMiddleware(jwtSecret),
    )(handler)
    
    http.ListenAndServe(":8080", wrapped)
}
```

---

## Notes

**Security Checklist:**
- ✅ Always validate JWT signatures
- ✅ Use HTTPS in production
- ✅ Set secure cookie flags (HttpOnly, Secure, SameSite)
- ✅ Implement rate limiting on auth endpoints
- ✅ Use short-lived access tokens (15 min) + refresh tokens
- ✅ Validate token expiration
- ✅ Implement CSRF protection for cookies
- ✅ Whitelist CORS origins (don't use * in production)
- ✅ Log authentication failures
- ✅ Implement account lockout after failed attempts

**Common Pitfalls:**
- ❌ Storing JWT secret in code (use environment variables)
- ❌ Not validating token expiration
- ❌ Using weak signing algorithms
- ❌ Exposing sensitive data in JWT payload
- ❌ Not implementing token refresh
- ❌ Allowing CORS * with credentials

---