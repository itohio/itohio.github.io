---
title: "Go Modules & Workspaces"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "go", "golang", "modules", "workspaces"]
---


Go modules and workspaces for dependency management. Essential commands for Go project management.

---

## Initialization

```bash
# Initialize new module
go mod init example.com/myproject
go mod init github.com/username/repo

# Initialize in existing directory
cd myproject
go mod init example.com/myproject
```

**go.mod file:**
```go
module example.com/myproject

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
)

require (
    // Indirect dependencies
    github.com/bytedance/sonic v1.9.1 // indirect
    github.com/chenzhuoyu/base64x v0.0.0-20221115062448-fe3a3abad311 // indirect
)
```

---

## Adding Dependencies

```bash
# Add dependency (automatically adds to go.mod)
go get github.com/gin-gonic/gin

# Add specific version
go get github.com/gin-gonic/gin@v1.9.1

# Add latest version
go get github.com/gin-gonic/gin@latest

# Add specific commit
go get github.com/gin-gonic/gin@abc1234

# Add branch
go get github.com/gin-gonic/gin@master

# Add and install
go get -u github.com/gin-gonic/gin  # Update to latest minor/patch
go get -u=patch github.com/gin-gonic/gin  # Update to latest patch only
```

---

## Removing Dependencies

```bash
# Remove unused dependencies
go mod tidy

# Remove specific dependency (edit go.mod, then tidy)
# Remove from go.mod manually, then:
go mod tidy
```

---

## Updating Dependencies

```bash
# Update all dependencies to latest minor/patch
go get -u ./...

# Update all dependencies to latest patch only
go get -u=patch ./...

# Update specific package
go get -u github.com/gin-gonic/gin

# View available updates
go list -u -m all

# Update to specific version
go get github.com/gin-gonic/gin@v1.10.0
```

---

## Listing Dependencies

```bash
# List all dependencies
go list -m all

# List direct dependencies only
go list -m -f '{{if not .Indirect}}{{.Path}}{{end}}' all

# List outdated dependencies
go list -u -m all

# Show dependency graph
go mod graph

# Show why a package is needed
go mod why github.com/gin-gonic/gin

# Show detailed info
go list -m -json github.com/gin-gonic/gin
```

---

## Vendoring

```bash
# Create vendor directory
go mod vendor

# Build using vendor
go build -mod=vendor

# Verify vendor directory matches go.mod
go mod verify
```

**Use Cases:**
- Offline builds
- Ensure reproducible builds
- Corporate environments with restricted internet

---

## Cleaning Up

```bash
# Remove unused dependencies and add missing ones
go mod tidy

# Verify dependencies
go mod verify

# Download dependencies to module cache
go mod download

# Clean module cache
go clean -modcache
```

---

## Replace Directive

**Local Development:**
```go
// go.mod
module example.com/myproject

go 1.21

require github.com/myorg/shared v1.0.0

// Replace with local version
replace github.com/myorg/shared => ../shared
```

**Fork or Mirror:**
```go
// Replace with fork
replace github.com/original/repo => github.com/myusername/repo v1.2.3

// Replace with local path
replace github.com/original/repo => /path/to/local/repo
```

```bash
# Add replace directive
go mod edit -replace github.com/original/repo=../local/repo

# Remove replace directive
go mod edit -dropreplace github.com/original/repo
```

---

## Go Workspaces (Go 1.18+)

### Creating Workspace

```bash
# Initialize workspace
go work init

# Add modules to workspace
go work use ./module1
go work use ./module2

# Add all modules in directory
go work use ./...
```

**go.work file:**
```go
go 1.21

use (
    ./api
    ./shared
    ./worker
)

// Optional: replace directives
replace github.com/myorg/shared => ./shared
```

### Workspace Structure

```
myproject/
├── go.work           # Workspace file
├── api/
│   ├── go.mod
│   ├── go.sum
│   └── main.go
├── shared/
│   ├── go.mod
│   ├── go.sum
│   └── lib.go
└── worker/
    ├── go.mod
    ├── go.sum
    └── main.go
```

### Workspace Commands

```bash
# Sync workspace
go work sync

# Edit workspace
go work edit -use ./newmodule
go work edit -dropuse ./oldmodule

# Run commands in workspace context
go build ./...
go test ./...

# Disable workspace (use go.mod instead)
go build -workfile=off
```

---

## Private Modules

### Setup for Private Repos

```bash
# Configure Git for private repos
git config --global url."git@github.com:".insteadOf "https://github.com/"

# Set GOPRIVATE environment variable
export GOPRIVATE=github.com/myorg/*,gitlab.com/mycompany/*

# Or in go env
go env -w GOPRIVATE=github.com/myorg/*

# Disable checksum database for private modules
go env -w GONOSUMDB=github.com/myorg/*
```

### Using Private Modules

```bash
# Ensure SSH key is configured
ssh -T git@github.com

# Install private module
go get github.com/myorg/private-repo
```

**.netrc for HTTPS (alternative):**
```
machine github.com
login username
password ghp_xxxxxxxxxxxxx
```

---

## Module Proxy

```bash
# View current proxy settings
go env GOPROXY

# Set proxy
go env -w GOPROXY=https://proxy.golang.org,direct

# Disable proxy (direct only)
go env -w GOPROXY=direct

# Use multiple proxies
go env -w GOPROXY=https://proxy1.com,https://proxy2.com,direct

# Bypass proxy for specific modules
go env -w GOPRIVATE=github.com/myorg/*
```

---

## Common go.mod Directives

```go
module example.com/myproject

// Go version
go 1.21

// Direct dependencies
require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
)

// Indirect dependencies (automatically managed)
require (
    github.com/bytedance/sonic v1.9.1 // indirect
)

// Replace dependencies
replace (
    github.com/old/repo => github.com/new/repo v1.2.3
    github.com/local/repo => ../local/repo
)

// Exclude specific versions
exclude github.com/broken/package v1.2.3

// Retract published versions (in your own module)
retract (
    v1.0.0 // Published accidentally
    [v1.1.0, v1.2.0] // Range of versions
)
```

---

## Troubleshooting

### Clear Module Cache

```bash
# Clear entire module cache
go clean -modcache

# Clear build cache
go clean -cache

# Clear all caches
go clean -cache -modcache -testcache
```

### Fix Checksum Mismatch

```bash
# Remove go.sum and regenerate
rm go.sum
go mod tidy

# Or update specific module
go get github.com/package/name@latest
go mod tidy
```

### Resolve Version Conflicts

```bash
# View dependency graph
go mod graph | grep package-name

# See why package is required
go mod why github.com/package/name

# Force specific version
go get github.com/package/name@v1.2.3
go mod tidy
```

### Private Repo Access Issues

```bash
# Check GOPRIVATE
go env GOPRIVATE

# Test SSH access
ssh -T git@github.com

# Use HTTPS with token
git config --global url."https://username:token@github.com/".insteadOf "https://github.com/"

# Or use .netrc file
```

---

## Best Practices

### 1. Always Run go mod tidy

```bash
# After adding/removing imports
go mod tidy

# Before committing
go mod tidy
git add go.mod go.sum
git commit -m "Update dependencies"
```

### 2. Commit go.sum

```bash
# ✅ Always commit go.sum
git add go.mod go.sum
git commit

# ❌ Don't ignore go.sum
# .gitignore should NOT contain go.sum
```

### 3. Use Semantic Versioning

```bash
# Tag releases properly
git tag v1.0.0
git push origin v1.0.0

# Follow semver
v1.0.0  # Initial release
v1.0.1  # Patch (bug fixes)
v1.1.0  # Minor (new features, backward compatible)
v2.0.0  # Major (breaking changes)
```

### 4. Pin Major Versions in Imports

```go
// For v2+ modules, include version in import path
import "github.com/user/repo/v2"

// go.mod
module github.com/user/repo/v2

go 1.21
```

### 5. Use Workspace for Monorepos

```bash
# Instead of replace directives, use workspaces
go work init
go work use ./module1 ./module2 ./module3

# Benefits:
# - Cleaner go.mod files
# - Easier to work across modules
# - No need to remember to remove replace directives
```

---

## Example Project Structure

### Single Module

```
myproject/
├── go.mod
├── go.sum
├── main.go
├── internal/
│   ├── handler/
│   │   └── handler.go
│   └── service/
│       └── service.go
└── pkg/
    └── utils/
        └── utils.go
```

### Multi-Module Workspace

```
myproject/
├── go.work
├── api/
│   ├── go.mod
│   ├── go.sum
│   ├── main.go
│   └── internal/
├── shared/
│   ├── go.mod
│   ├── go.sum
│   └── pkg/
│       ├── models/
│       └── utils/
└── worker/
    ├── go.mod
    ├── go.sum
    └── main.go
```

---

## Quick Reference

```bash
# Initialize
go mod init <module-path>

# Add/Update dependencies
go get <package>[@version]
go get -u ./...

# Clean up
go mod tidy

# Vendor
go mod vendor

# Workspace
go work init
go work use ./module

# Info
go list -m all
go mod graph
go mod why <package>

# Cache
go clean -modcache

# Private repos
go env -w GOPRIVATE=<pattern>
```

---

## Create and Publish Go Module

### Module Structure

```
mymodule/
├── go.mod
├── go.sum
├── README.md
├── LICENSE
├── .gitignore
├── cmd/
│   └── myapp/
│       └── main.go
├── pkg/
│   └── mylib/
│       ├── mylib.go
│       └── mylib_test.go
└── internal/
    └── helper/
        └── helper.go
```

### Initialize Module

```bash
# Create module
go mod init github.com/username/mymodule

# Or with version
go mod init github.com/username/mymodule/v2
```

### go.mod

```go
module github.com/username/mymodule

go 1.21

require (
    github.com/pkg/errors v0.9.1
    golang.org/x/sync v0.5.0
)

require (
    // Indirect dependencies
    github.com/stretchr/testify v1.8.4 // indirect
)
```

### Version Your Module

```bash
# Tag version
git tag v1.0.0
git push origin v1.0.0

# Major version (v2+) requires module path change
# go.mod:
module github.com/username/mymodule/v2

# Tag
git tag v2.0.0
git push origin v2.0.0
```

### Publish to GitHub

```bash
# Push to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/username/mymodule.git
git push -u origin main

# Tag version
git tag v1.0.0
git push origin v1.0.0

# Module is now available!
# Users can install with:
# go get github.com/username/mymodule@v1.0.0
```

### Semantic Versioning

```bash
# Patch release (bug fixes)
git tag v1.0.1
git push origin v1.0.1

# Minor release (new features, backward compatible)
git tag v1.1.0
git push origin v1.1.0

# Major release (breaking changes)
# Update go.mod first:
module github.com/username/mymodule/v2

git tag v2.0.0
git push origin v2.0.0
```

### Pre-release Versions

```bash
# Alpha
git tag v1.0.0-alpha.1
git push origin v1.0.0-alpha.1

# Beta
git tag v1.0.0-beta.1
git push origin v1.0.0-beta.1

# Release candidate
git tag v1.0.0-rc.1
git push origin v1.0.0-rc.1

# Install pre-release
go get github.com/username/mymodule@v1.0.0-beta.1
```

### Retract Versions

```go
// go.mod
module github.com/username/mymodule

go 1.21

// Retract broken versions
retract (
    v1.0.1 // Published accidentally
    [v1.0.5, v1.0.7] // Security vulnerability
)
```

### Private Go Modules

#### Using GOPRIVATE

```bash
# Configure private repos
go env -w GOPRIVATE=github.com/myorg/*

# Or multiple patterns
go env -w GOPRIVATE=github.com/myorg/*,gitlab.com/mycompany/*

# With authentication
git config --global url."https://username:token@github.com/".insteadOf "https://github.com/"
```

#### Using Go Proxy

```bash
# Athens (self-hosted Go proxy)
docker run -p 3000:3000 gomods/athens:latest

# Configure Go to use Athens
go env -w GOPROXY=http://localhost:3000,direct

# Or with fallback
go env -w GOPROXY=http://localhost:3000,https://proxy.golang.org,direct
```

#### Using Artifactory

```bash
# Configure Artifactory as Go proxy
go env -w GOPROXY=https://artifactory.example.com/api/go/go-virtual

# With authentication
# Create ~/.netrc:
machine artifactory.example.com
login username
password token
```

### Module Proxy (pkg.go.dev)

```bash
# After publishing to GitHub, request indexing
# Visit: https://pkg.go.dev/github.com/username/mymodule@v1.0.0

# Or use proxy directly
GOPROXY=https://proxy.golang.org go get github.com/username/mymodule@v1.0.0

# Module will appear on pkg.go.dev automatically
```

### Best Practices

```bash
# 1. Use semantic versioning
git tag v1.0.0

# 2. Write good documentation
# Add examples in _test.go files

# 3. Use go.mod replace for local development
replace github.com/username/mymodule => ../mymodule

# 4. Keep dependencies minimal
go mod tidy

# 5. Test before releasing
go test ./...

# 6. Use retract for broken versions
# Add to go.mod:
retract v1.0.1

# 7. Major version in module path (v2+)
module github.com/username/mymodule/v2
```

### Example: Publishable Library

```go
// mylib.go
package mylib

import "fmt"

// Version of the library
const Version = "1.0.0"

// Greet returns a greeting message
func Greet(name string) string {
    return fmt.Sprintf("Hello, %s!", name)
}

// Add adds two integers
func Add(a, b int) int {
    return a + b
}
```

```go
// mylib_test.go
package mylib_test

import (
    "testing"
    "github.com/username/mymodule/pkg/mylib"
)

func TestGreet(t *testing.T) {
    got := mylib.Greet("World")
    want := "Hello, World!"
    if got != want {
        t.Errorf("Greet() = %q, want %q", got, want)
    }
}

func ExampleGreet() {
    fmt.Println(mylib.Greet("World"))
    // Output: Hello, World!
}
```

### Publish Checklist

- [ ] Code is tested (`go test ./...`)
- [ ] Documentation is complete
- [ ] Examples are provided
- [ ] LICENSE file exists
- [ ] README.md is comprehensive
- [ ] go.mod is clean (`go mod tidy`)
- [ ] Version is tagged (`git tag v1.0.0`)
- [ ] Pushed to GitHub (`git push origin v1.0.0`)
- [ ] Verify on pkg.go.dev

---