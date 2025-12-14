---
title: "Bazel Build System"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "bazel", "build-system", "monorepo"]
---


Bazel - Fast, scalable, multi-language build system from Google.

---

## Installation

```bash
# Linux (via Bazelisk - recommended)
npm install -g @bazel/bazelisk

# Or download binary
wget https://github.com/bazelbuild/bazel/releases/download/6.4.0/bazel-6.4.0-linux-x86_64
chmod +x bazel-6.4.0-linux-x86_64
sudo mv bazel-6.4.0-linux-x86_64 /usr/local/bin/bazel

# macOS
brew install bazel

# Windows (via Chocolatey)
choco install bazel

# Verify
bazel version
```

---

## Basic Commands

```bash
# Build target
bazel build //path/to:target

# Build all targets
bazel build //...

# Run target
bazel run //path/to:target

# Test target
bazel test //path/to:test

# Test all
bazel test //...

# Clean build
bazel clean

# Deep clean
bazel clean --expunge

# Query dependencies
bazel query 'deps(//path/to:target)'

# Show build graph
bazel query --output=graph //path/to:target
```

---

## Workspace Setup

### WORKSPACE File

```python
# WORKSPACE
workspace(name = "my_project")

# Load rules
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Go rules
http_archive(
    name = "io_bazel_rules_go",
    sha256 = "...",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.41.0/rules_go-v0.41.0.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.41.0/rules_go-v0.41.0.zip",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()
go_register_toolchains(version = "1.21.0")
```

---

## BUILD Files

### Go Example

```python
# BUILD.bazel
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
    name = "mylib",
    srcs = ["lib.go"],
    importpath = "github.com/user/project/mylib",
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_pkg_errors//:errors",
    ],
)

go_binary(
    name = "myapp",
    srcs = ["main.go"],
    deps = [":mylib"],
)

go_test(
    name = "mylib_test",
    srcs = ["lib_test.go"],
    embed = [":mylib"],
)
```

### Python Example

```python
# BUILD.bazel
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "mylib",
    srcs = ["mylib.py"],
    deps = [
        "@pypi//requests",
    ],
)

py_binary(
    name = "myapp",
    srcs = ["main.py"],
    deps = [":mylib"],
)

py_test(
    name = "mylib_test",
    srcs = ["mylib_test.py"],
    deps = [":mylib"],
)
```

---

## Dependencies

### External Dependencies (Go)

```python
# WORKSPACE
load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies", "go_repository")

go_repository(
    name = "com_github_pkg_errors",
    importpath = "github.com/pkg/errors",
    sum = "h1:...",
    version = "v0.9.1",
)
```

### External Dependencies (Python)

```python
# WORKSPACE
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pypi",
    requirements_lock = "//:requirements.txt",
)

load("@pypi//:requirements.bzl", "install_deps")
install_deps()
```

---

## Gazelle (Go Dependency Management)

```bash
# Install Gazelle
# Add to WORKSPACE:
http_archive(
    name = "bazel_gazelle",
    sha256 = "...",
    urls = ["https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.33.0/bazel-gazelle-v0.33.0.tar.gz"],
)

# Generate BUILD files
bazel run //:gazelle

# Update dependencies
bazel run //:gazelle -- update-repos -from_file=go.mod
```

---

## Build Configuration

### .bazelrc

```bash
# .bazelrc
# Build settings
build --jobs=8
build --verbose_failures

# C++ settings
build --cxxopt=-std=c++17
build --host_cxxopt=-std=c++17

# Test settings
test --test_output=errors
test --test_summary=detailed

# Remote cache
build --remote_cache=https://cache.example.com

# Platforms
build:linux --platforms=@io_bazel_rules_go//go/toolchain:linux_amd64
build:macos --platforms=@io_bazel_rules_go//go/toolchain:darwin_amd64
```

---

## Query Commands

```bash
# Find all targets
bazel query //...

# Find dependencies
bazel query 'deps(//path/to:target)'

# Find reverse dependencies
bazel query 'rdeps(//..., //path/to:target)'

# Find tests
bazel query 'tests(//...)'

# Find by kind
bazel query 'kind("go_binary", //...)'

# Output as graph
bazel query --output=graph //path/to:target | dot -Tpng > graph.png
```

---

## Remote Caching

```bash
# Setup remote cache
bazel build --remote_cache=grpc://cache.example.com:9092 //...

# With authentication
bazel build \
  --remote_cache=grpcs://cache.example.com:443 \
  --google_default_credentials \
  //...

# Local disk cache
bazel build --disk_cache=/tmp/bazel-cache //...
```

---

## Remote Execution

```bash
# Build with remote execution
bazel build \
  --remote_executor=grpc://executor.example.com:8980 \
  --remote_cache=grpc://cache.example.com:9092 \
  //...
```

---

## Docker Integration

```python
# BUILD.bazel
load("@io_bazel_rules_docker//container:container.bzl", "container_image")

container_image(
    name = "myapp_image",
    base = "@alpine_linux_amd64//image",
    entrypoint = ["/myapp"],
    files = [":myapp"],
)
```

```bash
# Build Docker image
bazel build //path/to:myapp_image

# Load into Docker
bazel run //path/to:myapp_image
```

---

## Troubleshooting

```bash
# Clean build
bazel clean

# Deep clean (removes all caches)
bazel clean --expunge

# Verbose output
bazel build --verbose_failures //...

# Show commands
bazel build --subcommands //...

# Explain build
bazel build --explain=explain.txt //...

# Profile build
bazel build --profile=profile.json //...

# Analyze profile
bazel analyze-profile profile.json
```

---

## Best Practices

1. **Use Bazelisk** - Automatically manages Bazel versions
2. **Keep BUILD files simple** - One target per file when possible
3. **Use visibility** - Control target access
4. **Enable remote caching** - Speed up builds
5. **Use Gazelle** - Auto-generate BUILD files for Go
6. **Version dependencies** - Pin versions in WORKSPACE
7. **Test incrementally** - `bazel test //...` regularly

---

## Common Patterns

### Monorepo Structure

```
my-project/
├── WORKSPACE
├── .bazelrc
├── BUILD.bazel
├── go.mod
├── backend/
│   ├── BUILD.bazel
│   ├── main.go
│   └── lib/
│       ├── BUILD.bazel
│       └── lib.go
├── frontend/
│   ├── BUILD.bazel
│   └── src/
└── tools/
    └── BUILD.bazel
```

### Root BUILD.bazel

```python
# BUILD.bazel
load("@bazel_gazelle//:def.bzl", "gazelle")

# Generate BUILD files
gazelle(name = "gazelle")

# Update dependencies
gazelle(
    name = "gazelle-update-repos",
    args = [
        "-from_file=go.mod",
        "-to_macro=deps.bzl%go_dependencies",
        "-prune",
    ],
    command = "update-repos",
)
```

---