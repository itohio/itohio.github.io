---
title: "Rust Secure Coding"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "rust", "security"]
---


Secure coding practices for Rust applications.

---

## SQL Injection Prevention

```rust
// ❌ Vulnerable
let username = req.param("username");
let query = format!("SELECT * FROM users WHERE username = '{}'", username);
conn.query(&query);

// ✅ Secure (using sqlx)
let username = req.param("username");
let user = sqlx::query_as!(User, "SELECT * FROM users WHERE username = $1", username)
    .fetch_one(&pool)
    .await?;
```

---

## Command Injection Prevention

```rust
// ❌ Vulnerable
use std::process::Command;
let filename = req.param("file");
let output = Command::new("sh")
    .arg("-c")
    .arg(format!("cat {}", filename))
    .output()?;

// ✅ Secure
use std::process::Command;
let filename = req.param("file");
if !filename.chars().all(|c| c.is_alphanumeric() || c == '.' || c == '_') {
    return Err(Error::InvalidFilename);
}
let output = Command::new("cat")
    .arg(filename)
    .output()?;
```

---

## Secure Password Hashing

```rust
// ❌ Insecure
use md5::{Md5, Digest};
let hash = Md5::digest(password.as_bytes());

// ✅ Secure
use argon2::{
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2
};

let salt = SaltString::generate(&mut OsRng);
let argon2 = Argon2::default();
let password_hash = argon2.hash_password(password.as_bytes(), &salt)?
    .to_string();

// Verify
let parsed_hash = PasswordHash::new(&password_hash)?;
argon2.verify_password(password.as_bytes(), &parsed_hash)?;
```

---

## Secure Random Generation

```rust
// ❌ Insecure
use rand::Rng;
let mut rng = rand::thread_rng();
let token: u32 = rng.gen();

// ✅ Secure
use rand::rngs::OsRng;
use rand::RngCore;

let mut token = [0u8; 32];
OsRng.fill_bytes(&mut token);
```

---

## Unsafe Code Review

```rust
// ❌ Dangerous
unsafe {
    let ptr = some_value as *const i32;
    *ptr  // Potential UB
}

// ✅ Better: Minimize unsafe
// Only use unsafe when absolutely necessary
// Document safety invariants
/// # Safety
/// `ptr` must be valid and aligned
unsafe fn read_value(ptr: *const i32) -> i32 {
    ptr.read()
}
```

---