---
title: "Google Protobuf Packages"
date: 2024-12-12
draft: false
category: "protobuf"
tags: ["protobuf", "google", "api", "grpc"]
---



Useful Google protobuf packages for APIs, gRPC, and common patterns.

## googleapis

Common Google API definitions.

### Installation

```bash
# Using buf
buf dep update

# Add to buf.yaml
deps:
  - buf.build/googleapis/googleapis
```

### google.type

Common types for APIs.

```protobuf
import "google/type/date.proto";
import "google/type/datetime.proto";
import "google/type/dayofweek.proto";
import "google/type/money.proto";
import "google/type/postal_address.proto";
import "google/type/latlng.proto";
import "google/type/color.proto";

message Event {
    string name = 1;
    google.type.Date start_date = 2;
    google.type.DateTime start_time = 3;
    google.type.DayOfWeek day = 4;
}

message Product {
    string name = 1;
    google.type.Money price = 2;
}

message Location {
    string name = 1;
    google.type.LatLng coordinates = 2;
    google.type.PostalAddress address = 3;
}
```

#### Date

```protobuf
message Date {
    int32 year = 1;   // Year (e.g., 2024)
    int32 month = 2;  // Month (1-12)
    int32 day = 3;    // Day (1-31)
}
```

#### Money

```protobuf
message Money {
    string currency_code = 1;  // ISO 4217 (e.g., "USD")
    int64 units = 2;           // Whole units
    int32 nanos = 3;           // Fractional units (nano precision)
}
```

**Example:**
```go
price := &money.Money{
    CurrencyCode: "USD",
    Units: 19,
    Nanos: 990000000,  // $19.99
}
```

#### LatLng

```protobuf
message LatLng {
    double latitude = 1;   // -90 to +90
    double longitude = 2;  // -180 to +180
}
```

---

## google.rpc

Standard RPC error handling.

### Status

```protobuf
import "google/rpc/status.proto";
import "google/rpc/error_details.proto";

message Response {
    google.rpc.Status status = 1;
    bytes data = 2;
}
```

### Error Details

```protobuf
import "google/rpc/error_details.proto";

// Common error details
google.rpc.RetryInfo
google.rpc.DebugInfo
google.rpc.QuotaFailure
google.rpc.PreconditionFailure
google.rpc.BadRequest
google.rpc.RequestInfo
google.rpc.ResourceInfo
google.rpc.Help
google.rpc.LocalizedMessage
```

**Usage (Go):**

```go
import (
    "google.golang.org/genproto/googleapis/rpc/errdetails"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// Create error with details
st := status.New(codes.InvalidArgument, "invalid email format")
br := &errdetails.BadRequest{
    FieldViolations: []*errdetails.BadRequest_FieldViolation{
        {
            Field: "email",
            Description: "must be a valid email address",
        },
    },
}
st, err := st.WithDetails(br)
return st.Err()
```

---

## google.api

API annotations for REST/gRPC transcoding.

### HTTP Annotations

```protobuf
import "google/api/annotations.proto";
import "google/api/http.proto";

service UserService {
    rpc GetUser(GetUserRequest) returns (User) {
        option (google.api.http) = {
            get: "/v1/users/{user_id}"
        };
    }
    
    rpc CreateUser(CreateUserRequest) returns (User) {
        option (google.api.http) = {
            post: "/v1/users"
            body: "*"
        };
    }
    
    rpc UpdateUser(UpdateUserRequest) returns (User) {
        option (google.api.http) = {
            patch: "/v1/users/{user.id}"
            body: "user"
        };
    }
    
    rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty) {
        option (google.api.http) = {
            delete: "/v1/users/{user_id}"
        };
    }
    
    rpc ListUsers(ListUsersRequest) returns (ListUsersResponse) {
        option (google.api.http) = {
            get: "/v1/users"
        };
    }
}
```

### Field Behavior

```protobuf
import "google/api/field_behavior.proto";

message CreateUserRequest {
    string name = 1 [(google.api.field_behavior) = REQUIRED];
    string email = 2 [(google.api.field_behavior) = REQUIRED];
    string phone = 3 [(google.api.field_behavior) = OPTIONAL];
    string id = 4 [(google.api.field_behavior) = OUTPUT_ONLY];
}
```

**Field Behaviors:**
- `REQUIRED` - Must be provided
- `OPTIONAL` - May be provided
- `OUTPUT_ONLY` - Read-only, set by server
- `INPUT_ONLY` - Write-only, not returned
- `IMMUTABLE` - Cannot be changed after creation

### Resource

```protobuf
import "google/api/resource.proto";

message User {
    option (google.api.resource) = {
        type: "example.com/User"
        pattern: "users/{user}"
    };
    
    string name = 1;
}

message GetUserRequest {
    string name = 1 [(google.api.resource_reference) = {
        type: "example.com/User"
    }];
}
```

---

## google.longrunning

Long-running operations.

```protobuf
import "google/longrunning/operations.proto";

service BatchService {
    rpc ProcessBatch(ProcessBatchRequest) returns (google.longrunning.Operation) {
        option (google.api.http) = {
            post: "/v1/batches:process"
            body: "*"
        };
        option (google.longrunning.operation_info) = {
            response_type: "ProcessBatchResponse"
            metadata_type: "ProcessBatchMetadata"
        };
    }
}

message ProcessBatchMetadata {
    int32 total_items = 1;
    int32 processed_items = 2;
    google.protobuf.Timestamp start_time = 3;
}

message ProcessBatchResponse {
    int32 successful = 1;
    int32 failed = 2;
    repeated string errors = 3;
}
```

**Usage (Go):**

```go
import (
    "google.golang.org/genproto/googleapis/longrunning"
)

// Start operation
op, err := client.ProcessBatch(ctx, req)

// Poll for completion
for {
    if op.Done {
        if err := op.GetError(); err != nil {
            // handle error
        }
        resp := &ProcessBatchResponse{}
        op.GetResponse().UnmarshalTo(resp)
        break
    }
    time.Sleep(time.Second)
    op, _ = opsClient.GetOperation(ctx, &longrunning.GetOperationRequest{
        Name: op.Name,
    })
}
```

---

## buf.build Packages

### buf.validate

Validation rules for fields.

```protobuf
import "buf/validate/validate.proto";

message User {
    string email = 1 [(buf.validate.field).string = {
        email: true
        min_len: 5
        max_len: 100
    }];
    
    int32 age = 2 [(buf.validate.field).int32 = {
        gte: 0
        lte: 150
    }];
    
    string username = 3 [(buf.validate.field).string = {
        pattern: "^[a-z0-9_]{3,20}$"
    }];
    
    repeated string tags = 4 [(buf.validate.field).repeated = {
        min_items: 1
        max_items: 10
        items: {
            string: {
                min_len: 1
                max_len: 50
            }
        }
    }];
}
```

### envoyproxy/protoc-gen-validate

Alternative validation (older, still widely used).

```protobuf
import "validate/validate.proto";

message User {
    string email = 1 [(validate.rules).string.email = true];
    int32 age = 2 [(validate.rules).int32 = {gte: 0, lte: 150}];
    string uuid = 3 [(validate.rules).string.uuid = true];
}
```

---

## grpc-ecosystem

### grpc-gateway

REST to gRPC transcoding.

```protobuf
import "protoc-gen-openapiv2/options/annotations.proto";

option (grpc.gateway.protoc_gen_openapiv2.options.openapiv2_swagger) = {
    info: {
        title: "User API";
        version: "1.0";
        description: "User management API";
    };
    schemes: HTTPS;
    consumes: "application/json";
    produces: "application/json";
};

service UserService {
    rpc GetUser(GetUserRequest) returns (User) {
        option (google.api.http) = {
            get: "/v1/users/{user_id}"
        };
        option (grpc.gateway.protoc_gen_openapiv2.options.openapiv2_operation) = {
            summary: "Get user by ID"
            description: "Returns a single user"
            tags: "Users"
        };
    }
}
```

---

## Quick Reference

| Package | Import | Use Case |
|---------|--------|----------|
| `google.type` | `google/type/*.proto` | Common types (Date, Money, LatLng) |
| `google.rpc` | `google/rpc/*.proto` | Error handling, status codes |
| `google.api` | `google/api/*.proto` | REST annotations, field behavior |
| `google.longrunning` | `google/longrunning/*.proto` | Long-running operations |
| `buf.validate` | `buf/validate/validate.proto` | Field validation rules |
| `grpc-gateway` | `protoc-gen-openapiv2/options/*.proto` | OpenAPI/Swagger docs |

---

## buf.yaml Example

```yaml
version: v1
deps:
  - buf.build/googleapis/googleapis
  - buf.build/bufbuild/protovalidate
  - buf.build/grpc-ecosystem/grpc-gateway
lint:
  use:
    - DEFAULT
breaking:
  use:
    - FILE
```

---

## Best Practices

1. **Use `google.type.Money`** - Don't use float/double for money
2. **Use `google.type.Date`** - Separate date from time when appropriate
3. **Use `google.api.http`** - Enable REST/gRPC transcoding
4. **Use `google.api.field_behavior`** - Document field requirements
5. **Use `google.rpc.Status`** - Standard error responses
6. **Use validation** - buf.validate or protoc-gen-validate
7. **Use `google.longrunning.Operation`** - For async operations

---

## Common Imports

```protobuf
syntax = "proto3";

package myapp.v1;

// Well-known types
import "google/protobuf/timestamp.proto";
import "google/protobuf/duration.proto";
import "google/protobuf/empty.proto";
import "google/protobuf/field_mask.proto";

// Google types
import "google/type/date.proto";
import "google/type/money.proto";
import "google/type/latlng.proto";

// API annotations
import "google/api/annotations.proto";
import "google/api/field_behavior.proto";
import "google/api/resource.proto";

// Validation
import "buf/validate/validate.proto";
```

