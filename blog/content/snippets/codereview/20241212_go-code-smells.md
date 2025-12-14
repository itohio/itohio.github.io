---
title: "Go Code Smells"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "go", "golang", "code-smells"]
---


Common code smells in Go and how to fix them.

---

## Ignoring Errors

```go
// ❌ Bad
result, _ := doSomething()

// ✅ Good
result, err := doSomething()
if err != nil {
    return fmt.Errorf("failed: %w", err)
}
```

---

## Not Using defer

```go
// ❌ Bad
file, err := os.Open("file.txt")
if err != nil {
    return err
}
data, err := io.ReadAll(file)
file.Close()
return data, err

// ✅ Good
file, err := os.Open("file.txt")
if err != nil {
    return err
}
defer file.Close()
data, err := io.ReadAll(file)
return data, err
```

---

## Goroutine Leaks

```go
// ❌ Bad: Goroutine never stops
func process() {
    ch := make(chan int)
    go func() {
        for {
            select {
            case v := <-ch:
                process(v)
            }
        }
    }()
}

// ✅ Good: Use context
func process(ctx context.Context) {
    ch := make(chan int)
    go func() {
        for {
            select {
            case v := <-ch:
                process(v)
            case <-ctx.Done():
                return
            }
        }
    }()
}
```

---

## Pointer to Loop Variable

```go
// ❌ Bad
var results []*Item
for _, item := range items {
    results = append(results, &item)
}

// ✅ Good
var results []*Item
for i := range items {
    results = append(results, &items[i])
}
```

---

## Naked Returns

```go
// ❌ Bad
func calculate(x, y int) (result int, err error) {
    // ... 50 lines ...
    result = x + y
    return
}

// ✅ Good
func calculate(x, y int) (int, error) {
    // ... code ...
    result := x + y
    return result, nil
}
```

---