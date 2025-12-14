---
title: "Buf Commands & Project Structure"
date: 2024-12-12
draft: false
category: "protobuf"
tags: ["protobuf-knowhow", "protobuf", "buf", "grpc", "api"]
---


Buf commands, project structure, and code generation patterns for Protocol Buffers. Based on real-world project structure.

---

## Installation

```bash
# Install buf
# macOS
brew install bufbuild/buf/buf

# Linux
curl -sSL "https://github.com/bufbuild/buf/releases/latest/download/buf-$(uname -s)-$(uname -m)" -o /usr/local/bin/buf
chmod +x /usr/local/bin/buf

# Windows (Scoop)
scoop install buf

# Or use Go
go install github.com/bufbuild/buf/cmd/buf@latest

# Verify installation
buf --version
```

---

## Project Structure

```
proto/
├── buf.yaml              # Buf configuration
├── buf.gen.yaml          # Code generation config
├── buf.lock              # Dependency lock file
└── types/
    ├── core/
    │   ├── header.proto
    │   ├── intents.proto
    │   └── interests.proto
    ├── p2p/
    │   ├── addrbook.proto
    │   └── peering.proto
    └── api/
        ├── v1/
        │   ├── user.proto
        │   └── auth.proto
        └── v2/
            └── user.proto
```

---

## Configuration Files

### buf.yaml

```yaml
version: v1

# Dependencies from Buf Schema Registry
deps:
  - buf.build/protocolbuffers/wellknowntypes
  - buf.build/grpc-ecosystem/grpc-gateway

# Breaking change detection
breaking:
  use:
    - FILE  # Check for breaking changes at file level

# Linting rules
lint:
  use:
    - DEFAULT              # Default lint rules
    - COMMENTS             # Require comments
    - FILE_LOWER_SNAKE_CASE  # File naming convention
  except:
    - COMMENT_FIELD        # Don't require field comments
    - PACKAGE_VERSION_SUFFIX  # Allow packages without version suffix
  
  # Ignore specific files
  ignore:
    - types/test
```

### buf.gen.yaml

```yaml
version: v1

# Managed mode - automatic package management
managed:
  enabled: true
  go_package_prefix:
    default: github.com/myorg/myproject/
    except:
      - buf.build/protocolbuffers/wellknowntypes

# Code generation plugins
plugins:
  # Go
  - plugin: go
    out: ../
    opt: paths=source_relative
  
  # gRPC Go
  - plugin: go-grpc
    out: ../
    opt:
      - paths=source_relative
      - require_unimplemented_servers=false
  
  # gRPC Gateway
  - plugin: grpc-gateway
    out: ../
    opt:
      - paths=source_relative
      - generate_unbound_methods=true
  
  # OpenAPI/Swagger
  - plugin: openapiv2
    out: ../docs
    opt:
      - allow_merge=true
      - merge_file_name=api
  
  # TypeScript
  - plugin: es
    out: ../web/src/gen
    opt:
      - target=ts
  
  # Python
  - plugin: python
    out: ../python/gen
  
  # Rust
  - plugin: rust
    out: ../rust/gen
```

---

## Basic Commands

```bash
# Initialize new buf project
buf mod init

# Update dependencies
buf mod update

# Lint proto files
buf lint

# Check for breaking changes
buf breaking --against '.git#branch=main'

# Generate code
buf generate

# Format proto files
buf format -w

# Build
buf build

# Export to image
buf build -o image.bin

# Export to JSON
buf build -o image.json
```

---

## Linting

```bash
# Lint all files
buf lint

# Lint specific directory
buf lint types/core

# Lint with config
buf lint --config buf.yaml

# List lint rules
buf config ls-lint-rules

# Check specific rules
buf lint --error-format=json
```

---

## Breaking Change Detection

```bash
# Check against main branch
buf breaking --against '.git#branch=main'

# Check against specific commit
buf breaking --against '.git#commit=abc123'

# Check against remote
buf breaking --against 'https://github.com/user/repo.git#branch=main'

# Check against local directory
buf breaking --against '../old-proto'

# Check against Buf Schema Registry
buf breaking --against 'buf.build/myorg/myrepo'

# Exclude specific files
buf breaking --against '.git#branch=main' --exclude-path types/test
```

---

## Code Generation

```bash
# Generate from local files
buf generate

# Generate from specific directory
buf generate types/core

# Generate with specific config
buf generate --template buf.gen.yaml

# Generate from remote
buf generate buf.build/myorg/myrepo

# Generate from git
buf generate 'https://github.com/user/repo.git#branch=main,subdir=proto'
```

---

## Example Proto Files

### types/core/header.proto

```protobuf
syntax = "proto3";

package types.core;

option go_package = "github.com/myorg/myproject/types/core";

// Type enum defines message types
enum Type {
  UNSPECIFIED_TYPE = 0;
  MESSAGE = 1;
  INTENT = 2;
  INTENTS = 3;
  INTEREST = 4;
  INTERESTS = 5;
  NOTIFY_INTENT = 6;
  RESULT = 7;
  PING = 8;
  PONG = 9;
  HANDSHAKE = 50;
  PEERS = 51;
  ADDRBOOK = 52;
  USER_TYPES = 1000;
}

// Header is the message header
message Header {
  uint64 receive_timestamp = 1; // Overwritten upon reception
  uint64 timestamp = 2;
  Type type = 3;
  bool want_result = 4;
  bytes signature = 5;
  string route = 6;
}

// Result is the optional reply from remote peer
message Result {
  uint64 nonce = 1;
  uint64 error = 2;
  string description = 3;
}

// Ping message for latency measurement
message Ping {
  bytes payload = 1;
}

// Pong response for latency measurement
message Pong {
  uint64 receive_timestamp = 1; // Local timestamp when Ping received
  uint64 ping_timestamp = 2;    // Remote timestamp when Ping sent
  bytes payload = 3;
}
```

### types/api/v1/user.proto

```protobuf
syntax = "proto3";

package types.api.v1;

option go_package = "github.com/myorg/myproject/types/api/v1";

import "google/protobuf/timestamp.proto";
import "google/api/annotations.proto";

// User service
service UserService {
  // Get user by ID
  rpc GetUser(GetUserRequest) returns (GetUserResponse) {
    option (google.api.http) = {
      get: "/v1/users/{id}"
    };
  }
  
  // List users
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse) {
    option (google.api.http) = {
      get: "/v1/users"
    };
  }
  
  // Create user
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse) {
    option (google.api.http) = {
      post: "/v1/users"
      body: "*"
    };
  }
  
  // Update user
  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse) {
    option (google.api.http) = {
      put: "/v1/users/{id}"
      body: "*"
    };
  }
  
  // Delete user
  rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse) {
    option (google.api.http) = {
      delete: "/v1/users/{id}"
    };
  }
  
  // Stream user updates
  rpc StreamUsers(StreamUsersRequest) returns (stream User) {}
}

// User message
message User {
  string id = 1;
  string email = 2;
  string name = 3;
  google.protobuf.Timestamp created_at = 4;
  google.protobuf.Timestamp updated_at = 5;
  repeated string roles = 6;
  map<string, string> metadata = 7;
}

// Request/Response messages
message GetUserRequest {
  string id = 1;
}

message GetUserResponse {
  User user = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
  string filter = 3;
}

message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
  int32 total_count = 3;
}

message CreateUserRequest {
  string email = 1;
  string name = 2;
  repeated string roles = 3;
}

message CreateUserResponse {
  User user = 1;
}

message UpdateUserRequest {
  string id = 1;
  string name = 2;
  repeated string roles = 3;
}

message UpdateUserResponse {
  User user = 1;
}

message DeleteUserRequest {
  string id = 1;
}

message DeleteUserResponse {
  bool success = 1;
}

message StreamUsersRequest {
  string filter = 1;
}
```

---

## Buf Schema Registry

```bash
# Login to Buf Schema Registry
buf registry login

# Create repository
buf registry repository create buf.build/myorg/myrepo

# Push to registry
buf push

# Pull from registry
buf export buf.build/myorg/myrepo -o ./proto

# List repositories
buf registry repository list

# Delete repository
buf registry repository delete buf.build/myorg/myrepo
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Buf

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - uses: bufbuild/buf-setup-action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Lint
        run: buf lint
        working-directory: proto
      
      - name: Breaking change detection
        if: github.event_name == 'pull_request'
        run: buf breaking --against '.git#branch=main'
        working-directory: proto
      
      - name: Generate
        run: buf generate
        working-directory: proto
      
      - name: Push to Buf Schema Registry
        if: github.ref == 'refs/heads/main'
        run: buf push
        working-directory: proto
        env:
          BUF_TOKEN: ${{ secrets.BUF_TOKEN }}
```

---

## Makefile Integration

```makefile
.PHONY: proto-lint proto-gen proto-breaking proto-format proto-clean

# Lint proto files
proto-lint:
	cd proto && buf lint

# Generate code
proto-gen:
	cd proto && buf generate

# Check breaking changes
proto-breaking:
	cd proto && buf breaking --against '.git#branch=main'

# Format proto files
proto-format:
	cd proto && buf format -w

# Clean generated files
proto-clean:
	rm -rf types/*/types/*.pb.go
	rm -rf types/*/types/*_grpc.pb.go

# Update dependencies
proto-update:
	cd proto && buf mod update

# Full workflow
proto: proto-lint proto-breaking proto-gen
```

---

## Best Practices

### Naming Conventions

```protobuf
// ✅ Good: Clear, descriptive names
message UserProfile {
  string user_id = 1;
  string display_name = 2;
}

// ❌ Bad: Unclear abbreviations
message UsrProf {
  string uid = 1;
  string dn = 2;
}
```

### Field Numbers

```protobuf
// ✅ Good: Reserve deleted fields
message User {
  reserved 2, 3;  // Removed fields
  reserved "old_field", "deprecated_field";
  
  string id = 1;
  string name = 4;  // New field
}

// ❌ Bad: Reusing field numbers
message User {
  string id = 1;
  string name = 2;  // Was email before, now name
}
```

### Enums

```protobuf
// ✅ Good: Zero value is UNSPECIFIED
enum Status {
  STATUS_UNSPECIFIED = 0;
  STATUS_ACTIVE = 1;
  STATUS_INACTIVE = 2;
}

// ❌ Bad: Zero value has meaning
enum Status {
  ACTIVE = 0;  // Zero should be unspecified
  INACTIVE = 1;
}
```

### Versioning

```protobuf
// ✅ Good: Version in package name
package myapp.api.v1;

// ✅ Good: Version in service name for breaking changes
service UserServiceV2 {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
}
```

---

## Troubleshooting

```bash
# Clear buf cache
rm -rf ~/.cache/buf

# Verbose output
buf generate --debug

# Check dependencies
buf mod ls-deps

# Validate configuration
buf config ls-lint-rules
buf config ls-breaking-rules

# Check proto syntax
protoc --proto_path=. --descriptor_set_out=/dev/null types/**/*.proto
```

---