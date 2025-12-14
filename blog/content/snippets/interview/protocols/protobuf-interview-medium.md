---
title: "Protocol Buffers Interview Questions - Medium"
date: 2025-12-14
tags: ["protobuf", "protocol-buffers", "interview", "medium", "advanced", "grpc", "schema-evolution"]
---

Medium-level Protocol Buffers interview questions covering advanced features, schema evolution, and best practices.

## Q1: How do you handle schema evolution and backward compatibility?

**Answer**:

**Backward Compatibility Rules**:

1. **Never change field numbers**:
```protobuf
// ❌ BAD: Changing field number breaks compatibility
message User {
  string email = 3;  // Was field 2, now field 3 - BREAKS!
}

// ✅ GOOD: Keep field numbers
message User {
  string name = 1;
  int32 age = 2;
  string email = 3;  // New field, new number
}
```

2. **Don't remove fields** (mark as reserved):
```protobuf
message User {
  string name = 1;
  // int32 age = 2;  // ❌ DON'T DELETE
  
  // ✅ GOOD: Reserve the number
  reserved 2;
  reserved "age";
  
  string email = 3;
}
```

3. **Add new fields at the end**:
```protobuf
message User {
  string name = 1;      // Existing
  int32 age = 2;        // Existing
  string email = 3;     // Existing
  string phone = 4;     // ✅ New field
  bool verified = 5;    // ✅ New field
}
```

4. **Change types carefully**:
```protobuf
// ❌ BAD: Changing type breaks compatibility
message User {
  int32 age = 2;  // Was string, now int32
}

// ✅ GOOD: Use new field
message User {
  string age_str = 2;  // Old field (deprecated)
  int32 age = 6;       // New field
}
```

**Versioning Strategy**:
```protobuf
message User {
  // Version 1 fields
  string name = 1;
  int32 age = 2;
  
  // Version 2: Added
  string email = 3;
  
  // Version 3: Added
  repeated string tags = 4;
  
  // Version 4: Deprecated old field, added new
  reserved 2;  // Old age field
  int64 age_v2 = 5;  // New age field (int64)
}
```

**Documentation**: [Updating A Message Type](https://protobuf.dev/programming-guides/proto3/#updating)

---

## Q2: How do you use `Any` type for dynamic messages?

**Answer**:

**`Any` Type**:
```protobuf
import "google/protobuf/any.proto";

message Event {
  int64 timestamp = 1;
  string event_type = 2;
  google.protobuf.Any payload = 3;  // Can hold any message type
}
```

**Usage**:
```python
# Python
from google.protobuf import any_pb2
import user_pb2
import order_pb2

# Create event with User payload
user = user_pb2.User(name="John", age=30)
event = event_pb2.Event()
event.timestamp = 1234567890
event.event_type = "user_created"
event.payload.Pack(user)  # Pack message into Any

# Unpack
if event.payload.Is(user_pb2.User.DESCRIPTOR):
    unpacked_user = user_pb2.User()
    event.payload.Unpack(unpacked_user)
    print(unpacked_user.name)
```

```go
// Go
import (
    "google.golang.org/protobuf/types/known/anypb"
    pb "path/to/generated"
)

// Pack
user := &pb.User{Name: "John", Age: 30}
any, _ := anypb.New(user)

event := &pb.Event{
    Timestamp: 1234567890,
    EventType: "user_created",
    Payload:   any,
}

// Unpack
if event.Payload.MessageIs(&pb.User{}) {
    user := &pb.User{}
    event.Payload.UnmarshalTo(user)
    fmt.Println(user.Name)
}
```

**Use Cases**:
- Plugin systems
- Event sourcing
- Generic message handlers
- Extensible APIs

**Documentation**: [Any](https://protobuf.dev/programming-guides/proto3/#any)

---

## Q3: How do you use `Well-Known Types`?

**Answer**:

**Common Well-Known Types**:

1. **Timestamp**:
```protobuf
import "google/protobuf/timestamp.proto";

message Order {
  int64 id = 1;
  google.protobuf.Timestamp created_at = 2;
  google.protobuf.Timestamp updated_at = 3;
}
```

```python
# Python
from google.protobuf.timestamp_pb2 import Timestamp
from datetime import datetime

order = order_pb2.Order()
order.id = 123

# Set timestamp
now = datetime.now()
timestamp = Timestamp()
timestamp.FromDatetime(now)
order.created_at.CopyFrom(timestamp)

# Get timestamp
dt = order.created_at.ToDatetime()
```

2. **Duration**:
```protobuf
import "google/protobuf/duration.proto";

message Task {
  string name = 1;
  google.protobuf.Duration estimated_time = 2;
}
```

3. **Struct** (JSON-like):
```protobuf
import "google/protobuf/struct.proto";

message Config {
  google.protobuf.Struct settings = 1;
}
```

```python
# Python
from google.protobuf.struct_pb2 import Struct

config = config_pb2.Config()
config.settings["key1"] = "value1"
config.settings["key2"] = 123
config.settings["key3"] = True
```

4. **Empty**:
```protobuf
import "google/protobuf/empty.proto";

service UserService {
  rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty);
}
```

**Documentation**: [Well-Known Types](https://protobuf.dev/reference/protobuf/google.protobuf/)

---

## Q4: How do you define and use services for gRPC?

**Answer**:

**Service Definition**:
```protobuf
syntax = "proto3";

package user;

import "google/protobuf/empty.proto";

service UserService {
  // Unary RPC
  rpc GetUser(GetUserRequest) returns (User);
  
  // Server streaming
  rpc ListUsers(ListUsersRequest) returns (stream User);
  
  // Client streaming
  rpc CreateUsers(stream CreateUserRequest) returns (CreateUsersResponse);
  
  // Bidirectional streaming
  rpc ChatUsers(stream ChatMessage) returns (stream ChatMessage);
}

message GetUserRequest {
  int64 user_id = 1;
}

message ListUsersRequest {
  int32 page = 1;
  int32 page_size = 2;
}

message CreateUserRequest {
  string name = 1;
  string email = 2;
}

message CreateUsersResponse {
  repeated int64 user_ids = 1;
}

message ChatMessage {
  int64 user_id = 1;
  string message = 2;
}
```

**Server Implementation (Go)**:
```go
type server struct {
    pb.UnimplementedUserServiceServer
}

func (s *server) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    // Fetch user
    user := &pb.User{
        Id:    req.UserId,
        Name:  "John Doe",
        Email: "john@example.com",
    }
    return user, nil
}

func (s *server) ListUsers(req *pb.ListUsersRequest, stream pb.UserService_ListUsersServer) error {
    // Stream users
    for i := 0; i < 10; i++ {
        user := &pb.User{
            Id:    int64(i),
            Name:  fmt.Sprintf("User %d", i),
            Email: fmt.Sprintf("user%d@example.com", i),
        }
        stream.Send(user)
    }
    return nil
}
```

**Client Usage (Go)**:
```go
conn, _ := grpc.Dial("localhost:50051", grpc.WithInsecure())
client := pb.NewUserServiceClient(conn)

// Unary
user, _ := client.GetUser(context.Background(), &pb.GetUserRequest{UserId: 123})

// Streaming
stream, _ := client.ListUsers(context.Background(), &pb.ListUsersRequest{Page: 1})
for {
    user, err := stream.Recv()
    if err == io.EOF {
        break
    }
    fmt.Println(user)
}
```

**Documentation**: [gRPC Services](https://grpc.io/docs/what-is-grpc/core-concepts/)

---

## Q5: How do you handle errors in gRPC services?

**Answer**:

**gRPC Status Codes**:
```go
import (
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

func (s *server) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    if req.UserId <= 0 {
        return nil, status.Error(codes.InvalidArgument, "user_id must be positive")
    }
    
    user, err := s.db.GetUser(req.UserId)
    if err == ErrNotFound {
        return nil, status.Error(codes.NotFound, "user not found")
    }
    if err != nil {
        return nil, status.Error(codes.Internal, "internal error")
    }
    
    return user, nil
}
```

**Status Codes**:
- `OK`: Success
- `INVALID_ARGUMENT`: Invalid request
- `NOT_FOUND`: Resource not found
- `ALREADY_EXISTS`: Resource already exists
- `PERMISSION_DENIED`: Permission denied
- `UNAUTHENTICATED`: Authentication required
- `RESOURCE_EXHAUSTED`: Rate limiting
- `FAILED_PRECONDITION`: Precondition failed
- `ABORTED`: Operation aborted
- `OUT_OF_RANGE`: Out of valid range
- `UNIMPLEMENTED`: Not implemented
- `INTERNAL`: Internal error
- `UNAVAILABLE`: Service unavailable
- `DATA_LOSS`: Data loss

**Error Details**:
```go
import (
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
    "google.golang.org/genproto/googleapis/rpc/errdetails"
)

st := status.New(codes.InvalidArgument, "invalid user_id")
br := &errdetails.BadRequest{
    FieldViolations: []*errdetails.BadRequest_FieldViolation{
        {
            Field:       "user_id",
            Description: "must be positive",
        },
    },
}
st, _ = st.WithDetails(br)
return nil, st.Err()
```

**Documentation**: [gRPC Status Codes](https://grpc.io/docs/guides/error/)

---

## Q6: How do you use options and custom options?

**Answer**:

**Standard Options**:
```protobuf
import "google/protobuf/descriptor.proto";

message User {
  string name = 1 [(google.protobuf.field_options) = {
    deprecated: true
  }];
  
  string email = 2;
  
  int32 age = 3 [(validate.rules).int32 = {
    gte: 0,
    lte: 150
  }];
}
```

**Service Options**:
```protobuf
service UserService {
  option (google.api.default_host) = "api.example.com";
  
  rpc GetUser(GetUserRequest) returns (User) {
    option (google.api.http) = {
      get: "/v1/users/{user_id}"
    };
  }
}
```

**Custom Options**:
```protobuf
import "google/protobuf/descriptor.proto";

// Define custom option
extend google.protobuf.FieldOptions {
  string my_custom_option = 50000;
  int32 my_number_option = 50001;
}

message User {
  string name = 1 [(my_custom_option) = "custom_value"];
  int32 age = 2 [(my_number_option) = 42];
}
```

**Reading Options**:
```go
// Go
import "google.golang.org/protobuf/reflect/protoreflect"

field := desc.Fields().ByName("name")
opt := field.Options().(*descriptorpb.FieldOptions)
customOpt := proto.GetExtension(opt, my_custom_option).(string)
```

**Documentation**: [Custom Options](https://protobuf.dev/programming-guides/proto3/#customoptions)

---

## Q7: How do you optimize Protocol Buffer performance?

**Answer**:

**1. Use Appropriate Field Types**:
```protobuf
// ❌ BAD: Using int64 for small numbers
message Order {
  int64 quantity = 1;  // Usually small, wastes space
}

// ✅ GOOD: Use int32 for small numbers
message Order {
  int32 quantity = 1;  // More efficient
}
```

**2. Use Packed Repeated Fields**:
```protobuf
// proto3: Repeated scalar fields are packed by default
message Data {
  repeated int32 values = 1;  // Automatically packed
}

// proto2: Explicitly pack
message Data {
  repeated int32 values = 1 [packed=true];
}
```

**3. Use `sint32`/`sint64` for Negative Numbers**:
```protobuf
// ❌ BAD: int32 for negative numbers
message Temperature {
  int32 celsius = 1;  // Negative values encoded inefficiently
}

// ✅ GOOD: sint32 for negative numbers
message Temperature {
  sint32 celsius = 1;  // ZigZag encoding for negatives
}
```

**4. Avoid Unnecessary Nesting**:
```protobuf
// ❌ BAD: Deep nesting
message A {
  message B {
    message C {
      string value = 1;
    }
    C c = 1;
  }
  B b = 1;
}

// ✅ GOOD: Flatten when possible
message A {
  string value = 1;  // Direct access
}
```

**5. Use `bytes` for Large Binary Data**:
```protobuf
message Image {
  bytes data = 1;  // Efficient for binary
  string format = 2;
}
```

**6. Reuse Message Instances**:
```go
// ❌ BAD: Creating new instances
for i := 0; i < 1000; i++ {
    user := &pb.User{}  // New allocation each time
    // ...
}

// ✅ GOOD: Reuse instances
user := &pb.User{}
for i := 0; i < 1000; i++ {
    user.Reset()  // Reuse
    // ...
}
```

**Documentation**: [Encoding](https://protobuf.dev/programming-guides/encoding/)

---

## Q8: How do you handle large messages and streaming?

**Answer**:

**Problem**: Large messages can cause memory issues.

**Solution 1: Streaming**:
```protobuf
service DataService {
  // Stream large dataset
  rpc GetLargeDataset(GetDatasetRequest) returns (stream DataChunk);
  
  // Stream upload
  rpc UploadLargeFile(stream FileChunk) returns (UploadResponse);
}
```

**Solution 2: Pagination**:
```protobuf
message ListUsersRequest {
  int32 page = 1;
  int32 page_size = 2;
}

message ListUsersResponse {
  repeated User users = 1;
  int32 total_count = 2;
  int32 page = 3;
  bool has_next = 4;
}
```

**Solution 3: Chunking**:
```protobuf
message LargeData {
  int64 total_size = 1;
  repeated DataChunk chunks = 2;
}

message DataChunk {
  int32 chunk_index = 1;
  bytes data = 2;
  bool is_last = 3;
}
```

**Streaming Implementation**:
```go
func (s *server) GetLargeDataset(req *pb.GetDatasetRequest, stream pb.DataService_GetLargeDatasetServer) error {
    data := s.getLargeData()
    chunkSize := 1024 * 1024  // 1MB chunks
    
    for i := 0; i < len(data); i += chunkSize {
        end := i + chunkSize
        if end > len(data) {
            end = len(data)
        }
        
        chunk := &pb.DataChunk{
            ChunkIndex: int32(i / chunkSize),
            Data:       data[i:end],
            IsLast:     end == len(data),
        }
        
        if err := stream.Send(chunk); err != nil {
            return err
        }
    }
    
    return nil
}
```

**Documentation**: [gRPC Streaming](https://grpc.io/docs/what-is-grpc/core-concepts/#streaming-rpc)

---

## Q9: How do you validate Protocol Buffer messages?

**Answer**:

**Using `protoc-gen-validate`**:
```protobuf
import "validate/validate.proto";

message User {
  string email = 1 [(validate.rules).string.email = true];
  int32 age = 2 [(validate.rules).int32 = {
    gte: 0,
    lte: 150
  }];
  string phone = 3 [(validate.rules).string.pattern = "^\\+?[1-9]\\d{1,14}$"];
  repeated string tags = 4 [(validate.rules).repeated = {
    min_items: 1,
    max_items: 10
  }];
}
```

**Validation Rules**:
- **String**: `min_len`, `max_len`, `pattern`, `email`, `uri`
- **Numbers**: `const`, `lt`, `lte`, `gt`, `gte`, `in`, `not_in`
- **Repeated**: `min_items`, `max_items`, `unique`
- **Maps**: `min_pairs`, `max_pairs`
- **Nested**: Validate nested messages

**Usage**:
```go
import "github.com/envoyproxy/protoc-gen-validate/validate"

user := &pb.User{
    Email: "invalid-email",
    Age:   200,
}

if err := user.Validate(); err != nil {
    // Handle validation error
    fmt.Println(err)
}
```

**Custom Validation**:
```go
func (u *User) Validate() error {
    if u.Age < 0 || u.Age > 150 {
        return fmt.Errorf("age must be between 0 and 150")
    }
    if !strings.Contains(u.Email, "@") {
        return fmt.Errorf("invalid email")
    }
    return nil
}
```

**Documentation**: [protoc-gen-validate](https://github.com/bufbuild/protoc-gen-validate)

---

## Q10: How do you use Protocol Buffers with different serialization formats?

**Answer**:

**1. Binary (Default)**:
```go
// Most efficient
data, _ := proto.Marshal(user)
user := &pb.User{}
proto.Unmarshal(data, user)
```

**2. JSON**:
```go
import "google.golang.org/protobuf/encoding/protojson"

// To JSON
jsonData, _ := protojson.Marshal(user)

// From JSON
user := &pb.User{}
protojson.Unmarshal(jsonData, user)
```

**3. Text Format**:
```go
import "google.golang.org/protobuf/encoding/prototext"

// To text
textData, _ := prototext.Marshal(user)

// From text
user := &pb.User{}
prototext.Unmarshal(textData, user)
```

**4. Wire Format** (for debugging):
```go
import "google.golang.org/protobuf/encoding/protowire"

// Inspect wire format
data, _ := proto.Marshal(user)
fmt.Println(protowire.FormatBytes(data))
```

**Performance Comparison**:
- **Binary**: Fastest, smallest
- **JSON**: Human-readable, larger
- **Text**: Human-readable, largest
- **Wire**: Debugging only

**Documentation**: [Encoding Formats](https://protobuf.dev/programming-guides/encoding/)

---

## Q11: How do you handle versioning and multiple proto files?

**Answer**:

**Using buf for Dependency Management**:

`buf` simplifies managing multiple proto files and dependencies:

**buf.yaml Configuration**:
```yaml
version: v1
name: buf.build/your-org/your-repo
deps:
  - buf.build/googleapis/googleapis
  - buf.build/your-org/common-proto
modules:
  - path: proto
lint:
  use:
    - DEFAULT
breaking:
  use:
    - FILE
```

**Import Strategy**:
```protobuf
// common.proto
syntax = "proto3";
package common;

message Timestamp {
  int64 seconds = 1;
  int32 nanos = 2;
}

// user.proto
syntax = "proto3";
package user;

import "common.proto";

message User {
  int64 id = 1;
  string name = 2;
  common.Timestamp created_at = 3;
}
```

**Versioning in Package Names**:
```protobuf
// v1/user.proto
syntax = "proto3";
package user.v1;

message User {
  int64 id = 1;
  string name = 2;
}

// v2/user.proto
syntax = "proto3";
package user.v2;

import "user/v1/user.proto";

message User {
  int64 id = 1;
  string name = 2;
  string email = 3;  // New field
}
```

**With buf**:
```bash
# Update dependencies
buf mod update

# Generate code (handles all imports automatically)
buf generate

# Check for breaking changes between versions
buf breaking --against 'buf.build/your-org/your-repo:main'
```

**Legacy: Using protoc with Import Paths**:
```bash
# Compile with import paths
protoc \
  --proto_path=. \
  --proto_path=./third_party \
  --go_out=. \
  user.proto
```

**Benefits of buf for Versioning**:
- Automatic dependency resolution
- Breaking change detection
- Centralized dependency management
- Works with Buf Schema Registry

**Documentation**: 
- [buf Dependencies](https://buf.build/docs/bsr/dependencies)
- [Importing Definitions](https://protobuf.dev/programming-guides/proto3/#other)

---

## Q12: How do you use Protocol Buffers with REST APIs?

**Answer**:

**gRPC-Gateway** (gRPC to REST):
```protobuf
import "google/api/annotations.proto";

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
      patch: "/v1/users/{user_id}"
      body: "user"
    };
  }
}
```

**JSON over HTTP**:
```go
import (
    "google.golang.org/protobuf/encoding/protojson"
    "net/http"
)

func GetUserHandler(w http.ResponseWriter, r *http.Request) {
    userID := r.URL.Query().Get("user_id")
    
    user := &pb.User{
        Id:   parseUserID(userID),
        Name: "John Doe",
    }
    
    // Convert to JSON
    jsonData, _ := protojson.Marshal(user)
    
    w.Header().Set("Content-Type", "application/json")
    w.Write(jsonData)
}
```

**Documentation**: [gRPC-Gateway](https://grpc-ecosystem.github.io/grpc-gateway/)

---

