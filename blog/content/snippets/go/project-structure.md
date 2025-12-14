---
title: "Go Project Structure"
date: 2024-12-12T20:20:00Z
draft: false
description: "Standard Go project layout and organization"
type: "snippet"
tags: ["go", "golang", "project-structure", "best-practices", "go-knowhow"]
category: "go"
---



Standard Go project layout following community conventions. Organize your Go projects for maintainability, testability, and clarity.

## Use Case

Use this structure when you need to:
- Start a new Go project
- Organize a growing codebase
- Follow Go community standards
- Prepare for open source release

## Standard Layout

```
myproject/
├── cmd/                    # Main applications
│   └── myapp/
│       └── main.go
├── internal/               # Private application code
│   ├── app/
│   ├── pkg/
│   └── config/
├── pkg/                    # Public library code
│   ├── api/
│   └── utils/
├── api/                    # API definitions (OpenAPI, Protocol Buffers)
├── web/                    # Web assets
├── configs/                # Configuration files
├── scripts/                # Build and deployment scripts
├── test/                   # Additional test data and helpers
├── docs/                   # Documentation
├── examples/               # Example code
├── tools/                  # Supporting tools
├── vendor/                 # Vendored dependencies (optional)
├── go.mod                  # Module definition
├── go.sum                  # Dependency checksums
├── Makefile                # Build automation
└── README.md
```

## Explanation

- `cmd/` - Main entry points, one subdirectory per executable
- `internal/` - Private code, cannot be imported by other projects
- `pkg/` - Public library code, can be imported by others
- `api/` - API contracts (protobuf, OpenAPI specs)
- `configs/` - Configuration file templates
- `test/` - Additional test files and test data

## Examples

### Example 1: Simple CLI Tool

```
mycli/
├── cmd/
│   └── mycli/
│       └── main.go          # Entry point
├── internal/
│   ├── command/             # Command implementations
│   │   ├── init.go
│   │   ├── run.go
│   │   └── status.go
│   └── config/
│       └── config.go        # Configuration handling
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

**main.go:**
```go
package main

import (
    "fmt"
    "os"
    
    "github.com/user/mycli/internal/command"
)

func main() {
    if err := command.Execute(); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
}
```

### Example 2: Web Service

```
myservice/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── handler/             # HTTP handlers
│   │   ├── user.go
│   │   └── auth.go
│   ├── service/             # Business logic
│   │   └── user_service.go
│   ├── repository/          # Data access
│   │   └── user_repo.go
│   └── model/               # Domain models
│       └── user.go
├── pkg/
│   └── client/              # Public API client
│       └── client.go
├── api/
│   └── openapi.yaml         # API specification
├── configs/
│   └── config.yaml
├── go.mod
└── README.md
```

**main.go:**
```go
package main

import (
    "log"
    "net/http"
    
    "github.com/user/myservice/internal/handler"
    "github.com/user/myservice/internal/repository"
    "github.com/user/myservice/internal/service"
)

func main() {
    // Initialize dependencies
    repo := repository.NewUserRepository()
    svc := service.NewUserService(repo)
    h := handler.NewHandler(svc)
    
    // Setup routes
    http.HandleFunc("/users", h.HandleUsers)
    
    // Start server
    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### Example 3: Library with Examples

```
mylib/
├── pkg/
│   └── mylib/
│       ├── core.go
│       ├── core_test.go
│       ├── utils.go
│       └── utils_test.go
├── examples/
│   ├── basic/
│   │   └── main.go
│   └── advanced/
│       └── main.go
├── docs/
│   ├── getting-started.md
│   └── api.md
├── go.mod
├── go.sum
└── README.md
```

### Example 4: Makefile

```makefile
.PHONY: build test clean run

# Build binary
build:
	go build -o bin/myapp cmd/myapp/main.go

# Run tests
test:
	go test -v ./...

# Run tests with coverage
test-coverage:
	go test -v -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out

# Run linter
lint:
	golangci-lint run

# Format code
fmt:
	go fmt ./...

# Tidy dependencies
tidy:
	go mod tidy

# Run application
run:
	go run cmd/myapp/main.go

# Clean build artifacts
clean:
	rm -rf bin/
	rm -f coverage.out

# Install dependencies
deps:
	go mod download

# Build for multiple platforms
build-all:
	GOOS=linux GOARCH=amd64 go build -o bin/myapp-linux-amd64 cmd/myapp/main.go
	GOOS=darwin GOARCH=amd64 go build -o bin/myapp-darwin-amd64 cmd/myapp/main.go
	GOOS=windows GOARCH=amd64 go build -o bin/myapp-windows-amd64.exe cmd/myapp/main.go
```

## Notes

- Use `internal/` to prevent external imports of private code
- Keep `pkg/` for genuinely reusable code
- One `main.go` per executable in `cmd/`
- Follow Go naming conventions (lowercase packages)
- Use `go mod` for dependency management

## Gotchas/Warnings

- ⚠️ **internal/**: Cannot be imported from outside the project
- ⚠️ **pkg/**: Only put stable, public APIs here
- ⚠️ **Flat structure**: Small projects can stay flat - don't over-engineer
- ⚠️ **Vendor**: Only vendor if you have specific requirements