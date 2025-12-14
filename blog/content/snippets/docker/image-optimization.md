---
title: "Docker Image Optimization"
date: 2024-12-12
draft: false
category: "docker"
tags: ["docker", "optimization", "alpine", "multi-stage", "go"]
---

Techniques for building small, efficient Docker images with focus on Alpine, scratch, and Go binaries.

---

## Multi-Stage Builds

### Basic Multi-Stage Build

```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o myapp

# Final stage
FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/myapp .
CMD ["./myapp"]
```

**Benefits**:
- Build dependencies not included in final image
- Smaller final image size
- Faster deployment

---

## Alpine-Based Images

### Using Alpine

```dockerfile
# Alpine is ~5MB vs Ubuntu ~70MB
FROM alpine:3.19

# Install packages
RUN apk add --no-cache \
    ca-certificates \
    tzdata

WORKDIR /app
COPY myapp .

CMD ["./myapp"]
```

### Alpine with Build Tools

```dockerfile
FROM alpine:3.19

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    go

# Build your app
WORKDIR /app
COPY . .
RUN go build -o myapp

# Clean up build dependencies (if not using multi-stage)
RUN apk del gcc musl-dev go

CMD ["./myapp"]
```

---

## Scratch Images (Minimal)

### Go Binary on Scratch

```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .

# Build statically linked binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o myapp .

# Final stage - scratch (empty image)
FROM scratch
COPY --from=builder /app/myapp /myapp

# Copy CA certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy timezone data if needed
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

ENTRYPOINT ["/myapp"]
```

**Scratch Image**:
- Literally empty - no OS, no shell
- Final image = your binary + dependencies
- Smallest possible size (~5-20MB for Go apps)
- No shell for debugging (use multi-stage with debug variant)

---

## Go-Specific Optimizations

### Static Binary Build

```dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build flags for minimal binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -a \
    -installsuffix cgo \
    -ldflags="-w -s" \
    -o myapp .

FROM scratch
COPY --from=builder /app/myapp /myapp
ENTRYPOINT ["/myapp"]
```

**Build Flags**:
- `CGO_ENABLED=0`: Disable CGO (pure Go, static linking)
- `-a`: Force rebuild of packages
- `-installsuffix cgo`: Add suffix to package directory
- `-ldflags="-w -s"`: Strip debug info and symbol table
  - `-w`: Disable DWARF generation
  - `-s`: Disable symbol table

### With UPX Compression

```dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .

# Build binary
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o myapp .

# Compress with UPX
RUN apk add --no-cache upx
RUN upx --best --lzma myapp

FROM scratch
COPY --from=builder /app/myapp /myapp
ENTRYPOINT ["/myapp"]
```

**UPX Compression**:
- Can reduce binary size by 50-70%
- Slight startup time increase (decompression)
- Trade-off: size vs startup speed

---

## musl vs glibc

### Understanding musl

Alpine uses **musl libc** instead of **glibc** (used by Debian/Ubuntu).

**musl characteristics**:
- Smaller (~1MB vs ~6MB for glibc)
- Simpler, more standards-compliant
- Slightly different behavior in some edge cases
- Better for static linking

### Pure Go (No CGO) - No libc Needed

```dockerfile
# Best approach for Go
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY . .

# CGO_ENABLED=0 means no C dependencies, no libc needed
RUN CGO_ENABLED=0 go build -o myapp .

FROM scratch
COPY --from=builder /app/myapp /myapp
ENTRYPOINT ["/myapp"]
```

### With CGO (Needs musl)

```dockerfile
FROM golang:1.21-alpine AS builder

# Install musl-dev for CGO
RUN apk add --no-cache gcc musl-dev

WORKDIR /app
COPY . .

# Build with CGO (dynamically linked to musl)
RUN go build -o myapp .

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/myapp /myapp
ENTRYPOINT ["/myapp"]
```

**When to use CGO**:
- Need C libraries (e.g., SQLite with `go-sqlite3`)
- Performance-critical C code
- Legacy C code integration

**When to avoid CGO**:
- Pure Go is faster to compile
- Easier cross-compilation
- Smaller binaries
- Better portability

---

## Distroless Images

### Google Distroless

```dockerfile
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o myapp .

# Distroless: minimal runtime, no shell, no package manager
FROM gcr.io/distroless/static-debian12
COPY --from=builder /app/myapp /myapp
ENTRYPOINT ["/myapp"]
```

**Distroless variants**:
- `static-debian12`: For static binaries (like Go with CGO_ENABLED=0)
- `base-debian12`: Includes glibc (for CGO)
- `cc-debian12`: Includes libgcc (for C++ dependencies)

**Benefits**:
- Smaller than Alpine in some cases
- More secure (fewer packages = smaller attack surface)
- No shell (can't exec into container)

---

## Layer Optimization

### Bad: Many Layers

```dockerfile
FROM alpine:3.19
RUN apk add --no-cache ca-certificates
RUN apk add --no-cache tzdata
RUN apk add --no-cache curl
RUN apk add --no-cache wget
COPY myapp /app/
WORKDIR /app
```

### Good: Fewer Layers

```dockerfile
FROM alpine:3.19

# Combine RUN commands
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    curl \
    wget

WORKDIR /app
COPY myapp .
```

### Best: Multi-Stage with Minimal Layers

```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o myapp .

FROM scratch
COPY --from=builder /app/myapp /myapp
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
ENTRYPOINT ["/myapp"]
```

---

## .dockerignore

### Essential .dockerignore

```
# Git
.git
.gitignore

# Build artifacts
*.exe
*.dll
*.so
*.dylib
bin/
obj/
target/

# Dependencies
node_modules/
vendor/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Documentation
README.md
LICENSE
*.md

# Tests
*_test.go
test/
tests/

# CI/CD
.github/
.gitlab-ci.yml
Jenkinsfile

# Docker
Dockerfile*
docker-compose*.yml
.dockerignore
```

**Benefits**:
- Faster builds (less context to send)
- Smaller build context
- Avoid accidentally copying secrets

---

## Cache Optimization

### Leverage Build Cache

```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app

# Copy dependency files first (cached if unchanged)
COPY go.mod go.sum ./
RUN go mod download

# Copy source code (invalidates cache only if code changes)
COPY . .
RUN CGO_ENABLED=0 go build -o myapp .

FROM scratch
COPY --from=builder /app/myapp /myapp
ENTRYPOINT ["/myapp"]
```

**Key**: Copy dependency files before source code so `go mod download` is cached.

### BuildKit Cache Mounts

```dockerfile
# syntax=docker/dockerfile:1
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./

# Use cache mount for go modules
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -o myapp .

FROM scratch
COPY --from=builder /app/myapp /myapp
ENTRYPOINT ["/myapp"]
```

**Enable BuildKit**:
```bash
export DOCKER_BUILDKIT=1
docker build -t myapp .
```

---

## Security Hardening

### Non-Root User

```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o myapp .

FROM alpine:3.19

# Create non-root user
RUN addgroup -g 1000 appgroup && \
    adduser -D -u 1000 -G appgroup appuser

WORKDIR /app
COPY --from=builder /app/myapp .

# Change ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

ENTRYPOINT ["./myapp"]
```

### Read-Only Filesystem

```dockerfile
FROM scratch
COPY myapp /myapp
USER 65534:65534
ENTRYPOINT ["/myapp"]
```

Run with:
```bash
docker run --read-only --tmpfs /tmp myapp
```

---

## Size Comparison

### Example: Go HTTP Server

```dockerfile
# 1. Full Golang image: ~800MB
FROM golang:1.21
WORKDIR /app
COPY . .
RUN go build -o myapp .
CMD ["./myapp"]

# 2. Alpine: ~15MB
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o myapp .

FROM alpine:3.19
COPY --from=builder /app/myapp /myapp
CMD ["./myapp"]

# 3. Scratch: ~8MB
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o myapp .

FROM scratch
COPY --from=builder /app/myapp /myapp
ENTRYPOINT ["/myapp"]

# 4. Scratch + UPX: ~3MB
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o myapp .
RUN apk add --no-cache upx && upx --best --lzma myapp

FROM scratch
COPY --from=builder /app/myapp /myapp
ENTRYPOINT ["/myapp"]
```

---

## Complete Example: Optimized Go App

```dockerfile
# syntax=docker/dockerfile:1

# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /app

# Cache dependencies
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

# Build
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s -X main.version=${VERSION}" \
    -o myapp .

# Optional: Compress with UPX
# RUN apk add --no-cache upx && upx --best --lzma myapp

# Final stage
FROM scratch

# Copy CA certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Copy binary
COPY --from=builder /app/myapp /myapp

# Non-root user
USER 65534:65534

# Metadata
LABEL org.opencontainers.image.source="https://github.com/user/repo"
LABEL org.opencontainers.image.description="My optimized Go app"

ENTRYPOINT ["/myapp"]
```

Build:
```bash
export DOCKER_BUILDKIT=1
docker build --build-arg VERSION=1.0.0 -t myapp:1.0.0 .
```

---

## Quick Reference

| Base Image | Size | Use Case |
|------------|------|----------|
| `scratch` | ~0MB | Static binaries (Go with CGO_ENABLED=0) |
| `alpine:3.19` | ~5MB | Need shell, basic tools |
| `gcr.io/distroless/static` | ~2MB | Static binaries, more secure than Alpine |
| `gcr.io/distroless/base` | ~20MB | Dynamic binaries (CGO) |
| `debian:bookworm-slim` | ~70MB | Need glibc, more packages |
| `ubuntu:22.04` | ~80MB | Maximum compatibility |

---

## Tips

- **Use multi-stage builds** - Keep build deps out of final image
- **CGO_ENABLED=0** - For pure Go, enables scratch/distroless
- **-ldflags="-w -s"** - Strip debug info (smaller binary)
- **Alpine for flexibility** - Small with package manager
- **Scratch for minimal** - Smallest possible (Go, Rust)
- **Distroless for security** - No shell, minimal attack surface
- **Cache go mod download** - Faster rebuilds
- **Order matters** - Copy dependencies before source
- **.dockerignore** - Exclude unnecessary files
- **Non-root user** - Security best practice
- **UPX compression** - Optional, 50-70% smaller (trade-off: startup time)

---

## Common Gotchas

### musl vs glibc

```dockerfile
# ❌ Bad: Build on Debian, run on Alpine (glibc vs musl)
FROM golang:1.21 AS builder
RUN go build -o myapp .

FROM alpine:3.19
COPY --from=builder /go/myapp /myapp
CMD ["./myapp"]
# Error: not found (looking for glibc)

# ✅ Good: Match libc or use CGO_ENABLED=0
FROM golang:1.21-alpine AS builder
RUN CGO_ENABLED=0 go build -o myapp .

FROM alpine:3.19
COPY --from=builder /go/myapp /myapp
CMD ["./myapp"]
```

### Missing CA Certificates

```dockerfile
# ❌ Bad: HTTPS calls will fail
FROM scratch
COPY myapp /myapp
CMD ["/myapp"]

# ✅ Good: Copy CA certs
FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY myapp /myapp
CMD ["/myapp"]
```

### Timezone Issues

```dockerfile
# ❌ Bad: No timezone data
FROM scratch
COPY myapp /myapp
CMD ["/myapp"]
# time.LoadLocation will fail

# ✅ Good: Copy timezone data
FROM scratch
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY myapp /myapp
CMD ["/myapp"]
```

