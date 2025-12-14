---
title: "Protobuf Well-Known Types"
date: 2024-12-12
draft: false
category: "protobuf"
tags: ["protobuf", "types", "google", "any", "timestamp"]
---



Google's standard protobuf library includes many useful well-known types.

## Import Path

```protobuf
import "google/protobuf/any.proto";
import "google/protobuf/timestamp.proto";
import "google/protobuf/duration.proto";
import "google/protobuf/struct.proto";
import "google/protobuf/wrappers.proto";
import "google/protobuf/empty.proto";
```

---

## Any Type

Store any message type dynamically.

### Definition

```protobuf
syntax = "proto3";

import "google/protobuf/any.proto";

message Container {
    google.protobuf.Any payload = 1;
}
```

### Usage (Go)

```go
import (
    "google.golang.org/protobuf/types/known/anypb"
)

// Pack a message into Any
userMsg := &User{Name: "Alice", Age: 30}
anyMsg, err := anypb.New(userMsg)
if err != nil {
    // handle error
}

container := &Container{Payload: anyMsg}

// Unpack from Any
var user User
if err := container.Payload.UnmarshalTo(&user); err != nil {
    // handle error
}

// Check type before unpacking
if container.Payload.MessageIs(&User{}) {
    var user User
    container.Payload.UnmarshalTo(&user)
}
```

### Usage (Python)

```python
from google.protobuf import any_pb2

# Pack
user_msg = User(name="Alice", age=30)
any_msg = any_pb2.Any()
any_msg.Pack(user_msg)

container = Container(payload=any_msg)

# Unpack
user = User()
if container.payload.Is(User.DESCRIPTOR):
    container.payload.Unpack(user)
```

---

## Timestamp

Represents a point in time.

### Definition

```protobuf
import "google/protobuf/timestamp.proto";

message Event {
    string name = 1;
    google.protobuf.Timestamp created_at = 2;
    google.protobuf.Timestamp updated_at = 3;
}
```

### Usage (Go)

```go
import (
    "time"
    "google.golang.org/protobuf/types/known/timestamppb"
)

// Create from time.Time
now := time.Now()
ts := timestamppb.New(now)

event := &Event{
    Name: "UserLogin",
    CreatedAt: ts,
}

// Convert to time.Time
t := event.CreatedAt.AsTime()

// Check validity
if event.CreatedAt.IsValid() {
    // use timestamp
}
```

### Usage (Python)

```python
from google.protobuf.timestamp_pb2 import Timestamp
from datetime import datetime

# Create from datetime
ts = Timestamp()
ts.FromDatetime(datetime.now())

event = Event(name="UserLogin", created_at=ts)

# Convert to datetime
dt = event.created_at.ToDatetime()
```

---

## Duration

Represents a time span.

### Definition

```protobuf
import "google/protobuf/duration.proto";

message Task {
    string name = 1;
    google.protobuf.Duration timeout = 2;
    google.protobuf.Duration elapsed = 3;
}
```

### Usage (Go)

```go
import (
    "time"
    "google.golang.org/protobuf/types/known/durationpb"
)

// Create from time.Duration
timeout := 30 * time.Second
dur := durationpb.New(timeout)

task := &Task{
    Name: "ProcessData",
    Timeout: dur,
}

// Convert to time.Duration
d := task.Timeout.AsDuration()

// Check validity
if task.Timeout.IsValid() {
    // use duration
}
```

---

## Struct, Value, ListValue

Dynamic JSON-like structures.

### Definition

```protobuf
import "google/protobuf/struct.proto";

message Config {
    google.protobuf.Struct settings = 1;
    google.protobuf.Value dynamic_value = 2;
    google.protobuf.ListValue items = 3;
}
```

### Usage (Go)

```go
import (
    "google.golang.org/protobuf/types/known/structpb"
)

// Create Struct from map
settings := map[string]interface{}{
    "debug": true,
    "timeout": 30,
    "hosts": []string{"localhost", "example.com"},
}

structSettings, err := structpb.NewStruct(settings)
if err != nil {
    // handle error
}

config := &Config{Settings: structSettings}

// Convert back to map
m := config.Settings.AsMap()

// Create Value
val, _ := structpb.NewValue("hello")
numberVal, _ := structpb.NewValue(42)
boolVal, _ := structpb.NewValue(true)

// Create ListValue
list, _ := structpb.NewList([]interface{}{"a", "b", "c"})
```

---

## Wrappers

Nullable primitive types.

### Definition

```protobuf
import "google/protobuf/wrappers.proto";

message User {
    string name = 1;
    google.protobuf.Int32Value age = 2;          // nullable int32
    google.protobuf.StringValue email = 3;       // nullable string
    google.protobuf.BoolValue verified = 4;      // nullable bool
    google.protobuf.DoubleValue balance = 5;     // nullable double
}
```

### Available Wrappers

- `google.protobuf.DoubleValue`
- `google.protobuf.FloatValue`
- `google.protobuf.Int64Value`
- `google.protobuf.UInt64Value`
- `google.protobuf.Int32Value`
- `google.protobuf.UInt32Value`
- `google.protobuf.BoolValue`
- `google.protobuf.StringValue`
- `google.protobuf.BytesValue`

### Usage (Go)

```go
import (
    "google.golang.org/protobuf/types/known/wrapperspb"
)

// Create with value
age := wrapperspb.Int32(30)
email := wrapperspb.String("user@example.com")

user := &User{
    Name: "Alice",
    Age: age,
    Email: email,
}

// Check if set (nil = not set)
if user.Age != nil {
    fmt.Println("Age:", user.Age.Value)
}

// Set to nil (unset)
user.Email = nil
```

---

## Empty

Represents an empty message (useful for RPCs).

### Definition

```protobuf
import "google/protobuf/empty.proto";

service UserService {
    rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty);
    rpc Ping(google.protobuf.Empty) returns (google.protobuf.Empty);
}
```

### Usage (Go)

```go
import (
    "google.golang.org/protobuf/types/known/emptypb"
)

// Return empty
func (s *server) DeleteUser(ctx context.Context, req *DeleteUserRequest) (*emptypb.Empty, error) {
    // delete user logic
    return &emptypb.Empty{}, nil
}

// Call with empty
resp, err := client.Ping(ctx, &emptypb.Empty{})
```

---

## FieldMask

Specify which fields to update/return.

### Definition

```protobuf
import "google/protobuf/field_mask.proto";

message UpdateUserRequest {
    User user = 1;
    google.protobuf.FieldMask update_mask = 2;
}
```

### Usage (Go)

```go
import (
    "google.golang.org/protobuf/types/known/fieldmaskpb"
)

// Create field mask
mask, err := fieldmaskpb.New(&User{}, "name", "email")
if err != nil {
    // handle error
}

req := &UpdateUserRequest{
    User: &User{
        Name: "Alice",
        Email: "alice@example.com",
    },
    UpdateMask: mask,
}

// Check if field is in mask
if mask.IsValid(&User{}) {
    // mask is valid for User type
}

// Get paths
paths := mask.GetPaths() // ["name", "email"]
```

---

## Quick Reference

| Type | Import | Use Case |
|------|--------|----------|
| `Any` | `google/protobuf/any.proto` | Store any message type |
| `Timestamp` | `google/protobuf/timestamp.proto` | Point in time |
| `Duration` | `google/protobuf/duration.proto` | Time span |
| `Struct` | `google/protobuf/struct.proto` | Dynamic JSON-like data |
| `Value` | `google/protobuf/struct.proto` | Dynamic single value |
| `ListValue` | `google/protobuf/struct.proto` | Dynamic array |
| `Empty` | `google/protobuf/empty.proto` | Empty response |
| `FieldMask` | `google/protobuf/field_mask.proto` | Partial updates |
| `*Value` (wrappers) | `google/protobuf/wrappers.proto` | Nullable primitives |

---

## Best Practices

1. **Use `Timestamp` for dates** - Don't use int64 for timestamps
2. **Use `Duration` for time spans** - Don't use int64 for durations
3. **Use wrappers for optional primitives** - Distinguish between zero and unset
4. **Use `Any` sparingly** - Type safety is lost, prefer oneof when possible
5. **Use `Empty` for void RPCs** - Standard way to represent no data
6. **Use `FieldMask` for updates** - Partial updates in REST/gRPC APIs
7. **Use `Struct` for dynamic data** - When schema is unknown at compile time

---

## Common Patterns

### Optional Fields (Proto3)

```protobuf
message User {
    string name = 1;
    google.protobuf.Int32Value age = 2;  // optional, can be null
}
```

### Polymorphic Messages

```protobuf
message Event {
    string id = 1;
    google.protobuf.Timestamp timestamp = 2;
    google.protobuf.Any payload = 3;  // Can be any event type
}
```

### Partial Updates

```protobuf
message UpdateRequest {
    User user = 1;
    google.protobuf.FieldMask update_mask = 2;
}
```

### Dynamic Configuration

```protobuf
message Config {
    google.protobuf.Struct settings = 1;  // Arbitrary JSON-like config
}
```

