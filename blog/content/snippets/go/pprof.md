---
title: "Go pprof Profiling"
date: 2024-12-12
draft: false
category: "go"
tags: ["go-knowhow", "profiling", "pprof", "performance", "optimization"]
---


Go pprof commands and practices for analyzing CPU, memory, goroutine, and block profiles.

---

## Enable Profiling

### HTTP Server

```go
package main

import (
    "log"
    "net/http"
    _ "net/http/pprof"  // Import for side effects
)

func main() {
    // Your application code here
    
    // Start pprof server
    go func() {
        log.Println("pprof server starting on :6060")
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()
    
    // Your main application logic
    select {}
}
```

### Programmatic Profiling

```go
package main

import (
    "os"
    "runtime"
    "runtime/pprof"
)

func main() {
    // CPU profiling
    f, _ := os.Create("cpu.prof")
    defer f.Close()
    pprof.StartCPUProfile(f)
    defer pprof.StopCPUProfile()
    
    // Your code here
    doWork()
    
    // Memory profiling
    mf, _ := os.Create("mem.prof")
    defer mf.Close()
    runtime.GC() // Get up-to-date statistics
    pprof.WriteHeapProfile(mf)
}
```

---

## CPU Profiling

### Capture CPU Profile

```bash
# Via HTTP endpoint (30 seconds)
curl http://localhost:6060/debug/pprof/profile?seconds=30 > cpu.prof

# Via test
go test -cpuprofile=cpu.prof -bench=.

# Via runtime/pprof (programmatic)
# See code example above
```

### Analyze CPU Profile

```bash
# Interactive mode
go tool pprof cpu.prof

# Commands in interactive mode:
# top       - Show top functions by CPU time
# top10     - Show top 10 functions
# list main - Show source code for main package
# web       - Open graphviz visualization (requires graphviz)
# pdf       - Generate PDF report
# svg       - Generate SVG visualization
# exit      - Exit

# Direct commands
go tool pprof -top cpu.prof
go tool pprof -list=main cpu.prof
go tool pprof -web cpu.prof

# Compare two profiles
go tool pprof -base=old.prof new.prof

# Generate flame graph
go tool pprof -http=:8080 cpu.prof
```

---

## Memory Profiling

### Capture Memory Profile

```bash
# Heap profile via HTTP
curl http://localhost:6060/debug/pprof/heap > heap.prof

# Allocs profile (all allocations, not just in-use)
curl http://localhost:6060/debug/pprof/allocs > allocs.prof

# Via test
go test -memprofile=mem.prof -bench=.

# Via benchmark with allocation stats
go test -benchmem -bench=.
```

### Analyze Memory Profile

```bash
# Interactive mode
go tool pprof heap.prof

# Show allocations
go tool pprof -alloc_space heap.prof

# Show in-use memory
go tool pprof -inuse_space heap.prof

# Top allocators
go tool pprof -top heap.prof

# List specific function
go tool pprof -list=functionName heap.prof

# Web visualization
go tool pprof -http=:8080 heap.prof

# Compare profiles
go tool pprof -base=old_heap.prof new_heap.prof
```

---

## Goroutine Profiling

### Capture Goroutine Profile

```bash
# Via HTTP
curl http://localhost:6060/debug/pprof/goroutine > goroutine.prof

# Full goroutine stack dump
curl http://localhost:6060/debug/pprof/goroutine?debug=2 > goroutine.txt
```

### Analyze Goroutines

```bash
# Interactive
go tool pprof goroutine.prof

# Direct view
go tool pprof -top goroutine.prof

# Web view
go tool pprof -http=:8080 goroutine.prof

# Text dump (human-readable)
cat goroutine.txt
```

---

## Block Profiling

### Enable Block Profiling

```go
package main

import (
    "runtime"
)

func main() {
    // Enable block profiling
    runtime.SetBlockProfileRate(1)  // 1 = record every blocking event
    
    // Your code
}
```

### Capture and Analyze

```bash
# Via HTTP
curl http://localhost:6060/debug/pprof/block > block.prof

# Analyze
go tool pprof block.prof
go tool pprof -top block.prof
go tool pprof -http=:8080 block.prof
```

---

## Mutex Profiling

### Enable Mutex Profiling

```go
package main

import (
    "runtime"
)

func main() {
    // Enable mutex profiling
    runtime.SetMutexProfileFraction(1)  // Sample 1 in N mutex events
    
    // Your code
}
```

### Capture and Analyze

```bash
# Via HTTP
curl http://localhost:6060/debug/pprof/mutex > mutex.prof

# Analyze
go tool pprof mutex.prof
go tool pprof -http=:8080 mutex.prof
```

---

## Trace Profiling

### Capture Trace

```go
package main

import (
    "os"
    "runtime/trace"
)

func main() {
    f, _ := os.Create("trace.out")
    defer f.Close()
    
    trace.Start(f)
    defer trace.Stop()
    
    // Your code
    doWork()
}
```

```bash
# Via HTTP (5 seconds)
curl http://localhost:6060/debug/pprof/trace?seconds=5 > trace.out

# Via test
go test -trace=trace.out
```

### Analyze Trace

```bash
# Open trace viewer
go tool trace trace.out

# This opens a web browser with:
# - View trace: Timeline of goroutines
# - Goroutine analysis: Goroutine execution stats
# - Network blocking profile
# - Synchronization blocking profile
# - Syscall blocking profile
# - Scheduler latency profile
```

---

## Benchmarking with Profiling

```bash
# CPU profile
go test -bench=. -cpuprofile=cpu.prof

# Memory profile
go test -bench=. -memprofile=mem.prof

# Both
go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof

# With allocation stats
go test -bench=. -benchmem

# Specific benchmark
go test -bench=BenchmarkMyFunc -cpuprofile=cpu.prof

# Run longer for more accurate results
go test -bench=. -benchtime=10s -cpuprofile=cpu.prof
```

---

## pprof Commands Reference

### Top-level Commands

```bash
# top [N]         - Show top N entries (default 10)
# list <regex>    - Show source code for functions matching regex
# web             - Open graphviz visualization
# weblist <regex> - Show annotated source in browser
# pdf             - Generate PDF report
# svg             - Generate SVG
# png             - Generate PNG
# gif             - Generate GIF
# dot             - Generate DOT file
# tree            - Show call tree
# peek <regex>    - Show callers and callees
# disasm <regex>  - Show disassembly
# exit/quit       - Exit pprof
```

### Options

```bash
# -flat           - Sort by flat (self) time
# -cum            - Sort by cumulative time
# -alloc_space    - Show allocation space
# -alloc_objects  - Show allocation count
# -inuse_space    - Show in-use space
# -inuse_objects  - Show in-use object count
# -base <file>    - Compare against base profile
# -http=:8080     - Start web interface
```

---

## Continuous Profiling

### Automated Collection

```go
package main

import (
    "fmt"
    "os"
    "runtime/pprof"
    "time"
)

func collectProfiles() {
    ticker := time.NewTicker(5 * time.Minute)
    defer ticker.Stop()
    
    for range ticker.C {
        timestamp := time.Now().Format("20060102-150405")
        
        // CPU profile
        cpuFile, _ := os.Create(fmt.Sprintf("cpu-%s.prof", timestamp))
        pprof.StartCPUProfile(cpuFile)
        time.Sleep(30 * time.Second)
        pprof.StopCPUProfile()
        cpuFile.Close()
        
        // Heap profile
        heapFile, _ := os.Create(fmt.Sprintf("heap-%s.prof", timestamp))
        pprof.WriteHeapProfile(heapFile)
        heapFile.Close()
    }
}

func main() {
    go collectProfiles()
    
    // Your application
}
```

---

## Best Practices

1. **Profile in production-like environment** - Dev machines may not reflect production
2. **Run long enough** - At least 30 seconds for CPU profiling
3. **Use benchmarks** - For repeatable profiling
4. **Compare profiles** - Use `-base` to see improvements
5. **Focus on hot paths** - Optimize top functions first
6. **Profile before optimizing** - Don't guess, measure
7. **Check allocations** - Use `-benchmem` to see allocation impact
8. **Use trace for concurrency** - Better than pprof for goroutine issues

---

## Example Workflow

```bash
# 1. Write benchmark
cat > main_test.go <<EOF
package main

import "testing"

func BenchmarkMyFunc(b *testing.B) {
    for i := 0; i < b.N; i++ {
        MyFunc()
    }
}
EOF

# 2. Run benchmark with profiling
go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof -benchmem

# 3. Analyze CPU profile
go tool pprof -http=:8080 cpu.prof

# 4. Analyze memory profile
go tool pprof -http=:8081 mem.prof

# 5. Make optimizations

# 6. Run again and compare
go test -bench=. -cpuprofile=cpu_new.prof
go tool pprof -base=cpu.prof cpu_new.prof
```

---