---
title: "Rust Code Smells"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "rust", "code-smells"]
---


Common code smells in Rust and how to fix them.

---

## Unwrap/Expect Abuse

```rust
// ❌ Bad
let value = some_option.unwrap();
let result = some_result.expect("failed");

// ✅ Good
let value = some_option.ok_or(Error::MissingValue)?;
let result = some_result?;
```

---

## Clone Instead of Borrow

```rust
// ❌ Bad
fn process(data: Vec<String>) {
    for item in data.clone() {
        println!("{}", item);
    }
}

// ✅ Good
fn process(data: &[String]) {
    for item in data {
        println!("{}", item);
    }
}
```

---

## Not Using Iterators

```rust
// ❌ Bad
let mut result = Vec::new();
for i in 0..items.len() {
    if items[i] > 10 {
        result.push(items[i] * 2);
    }
}

// ✅ Good
let result: Vec<_> = items.iter()
    .filter(|&&x| x > 10)
    .map(|&x| x * 2)
    .collect();
```

---

## Manual String Building

```rust
// ❌ Bad
let mut s = String::new();
s.push_str("Hello");
s.push_str(", ");
s.push_str("world");

// ✅ Good
let s = format!("Hello, {}", "world");
```

---

## Not Using Match

```rust
// ❌ Bad
if result.is_ok() {
    let value = result.unwrap();
    process(value);
} else {
    handle_error(result.unwrap_err());
}

// ✅ Good
match result {
    Ok(value) => process(value),
    Err(e) => handle_error(e),
}
```

---