---
title: "Hardware Random Number Generation"
date: 2024-12-12
draft: false
category: "hardware"
tags: ["hardware-knowhow", "random", "security", "entropy", "rng"]
---


Hardware random number generation using /dev/random, /dev/urandom, and hardware RNG sources.

---

## Linux Random Devices

### /dev/random vs /dev/urandom

```bash
# /dev/random - Blocks when entropy pool is depleted
# Use for: Long-term cryptographic keys

# /dev/urandom - Never blocks, uses CSPRNG when entropy low
# Use for: Most cryptographic operations (recommended)

# Read random bytes
head -c 32 /dev/urandom | base64

# Generate random number
od -An -N4 -tu4 < /dev/urandom

# Generate random hex
xxd -l 16 -p /dev/urandom
```

### Check Entropy

```bash
# Available entropy
cat /proc/sys/kernel/random/entropy_avail

# Should be > 1000 for good randomness
# Typical range: 128 - 4096

# Watch entropy
watch -n 1 cat /proc/sys/kernel/random/entropy_avail

# Entropy pool size
cat /proc/sys/kernel/random/poolsize
```

---

## Hardware RNG (HRNG)

### Check for Hardware RNG

```bash
# List hardware RNG devices
ls -l /dev/hwrng

# Check if rng-tools is installed
which rngd

# Install rng-tools
sudo apt install rng-tools

# Check RNG status
sudo rngd -l

# Start rng daemon
sudo systemctl start rng-tools
sudo systemctl enable rng-tools

# Verify it's feeding entropy
cat /proc/sys/kernel/random/entropy_avail
# Should be consistently high (3000-4000)
```

### CPU-based RNG (RDRAND/RDSEED)

```bash
# Check if CPU supports RDRAND
grep -o 'rdrand' /proc/cpuinfo

# Check if CPU supports RDSEED
grep -o 'rdseed' /proc/cpuinfo

# Use CPU RNG with rng-tools
sudo rngd -r /dev/hwrng
```

---

## Generate Random Data

### Random Bytes

```bash
# 32 random bytes
head -c 32 /dev/urandom

# Random bytes as hex
xxd -l 32 -p /dev/urandom

# Random bytes as base64
head -c 32 /dev/urandom | base64

# Random bytes to file
dd if=/dev/urandom of=random.bin bs=1M count=10
```

### Random Numbers

```bash
# Random number (0-99)
shuf -i 0-99 -n 1

# Random number using /dev/urandom
echo $(($(od -An -N4 -tu4 < /dev/urandom) % 100))

# Multiple random numbers
shuf -i 1-100 -n 10

# Random float (0-1)
echo "scale=10; $(od -An -N4 -tu4 < /dev/urandom) / 4294967295" | bc -l
```

### Random Strings

```bash
# Random alphanumeric string (32 chars)
tr -dc A-Za-z0-9 </dev/urandom | head -c 32

# Random password (with special chars)
tr -dc 'A-Za-z0-9!@#$%^&*' </dev/urandom | head -c 20

# Random UUID
cat /proc/sys/kernel/random/uuid

# Or use uuidgen
uuidgen
```

---

## Python

```python
import os
import secrets

# Cryptographically secure random bytes
random_bytes = os.urandom(32)
print(random_bytes.hex())

# Using secrets module (Python 3.6+)
# Token (URL-safe)
token = secrets.token_urlsafe(32)
print(token)

# Token (hex)
token_hex = secrets.token_hex(16)
print(token_hex)

# Random number
random_num = secrets.randbelow(100)  # 0-99
print(random_num)

# Random choice
items = ['apple', 'banana', 'cherry']
choice = secrets.choice(items)
print(choice)

# Generate password
import string
alphabet = string.ascii_letters + string.digits + string.punctuation
password = ''.join(secrets.choice(alphabet) for i in range(20))
print(password)
```

---

## Go

```go
package main

import (
    "crypto/rand"
    "encoding/base64"
    "encoding/hex"
    "fmt"
    "math/big"
)

func main() {
    // Random bytes
    bytes := make([]byte, 32)
    _, err := rand.Read(bytes)
    if err != nil {
        panic(err)
    }
    
    // As hex
    fmt.Println("Hex:", hex.EncodeToString(bytes))
    
    // As base64
    fmt.Println("Base64:", base64.StdEncoding.EncodeToString(bytes))
    
    // Random number (0-99)
    n, err := rand.Int(rand.Reader, big.NewInt(100))
    if err != nil {
        panic(err)
    }
    fmt.Println("Random number:", n)
    
    // Random string
    const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    b := make([]byte, 32)
    for i := range b {
        num, _ := rand.Int(rand.Reader, big.NewInt(int64(len(letters))))
        b[i] = letters[num.Int64()]
    }
    fmt.Println("Random string:", string(b))
}
```

---

## Rust

```rust
use rand::{Rng, thread_rng};
use rand::distributions::Alphanumeric;

fn main() {
    let mut rng = thread_rng();
    
    // Random number
    let n: u32 = rng.gen();
    println!("Random u32: {}", n);
    
    // Random number in range
    let n: u32 = rng.gen_range(0..100);
    println!("Random 0-99: {}", n);
    
    // Random bytes
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    println!("Random bytes: {:?}", bytes);
    
    // Random string
    let s: String = thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();
    println!("Random string: {}", s);
}
```

---

## OpenSSL

```bash
# Random bytes
openssl rand 32

# Random bytes as hex
openssl rand -hex 32

# Random bytes as base64
openssl rand -base64 32

# Random password
openssl rand -base64 20 | tr -d "=+/" | cut -c1-20
```

---

## Testing Randomness

### NIST Statistical Test Suite

```bash
# Install sts (NIST Statistical Test Suite)
# Download from: https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software

# Generate test data
head -c 1000000 /dev/urandom > random.bin

# Run tests
./assess 1000000

# Good randomness should pass most tests
```

### Simple Chi-Square Test

```bash
# Generate random numbers and check distribution
for i in {1..10000}; do 
    echo $(($(od -An -N1 -tu1 < /dev/urandom) % 10))
done | sort | uniq -c

# Each digit should appear ~1000 times
```

---

## Increase Entropy (if needed)

### haveged

```bash
# Install haveged (entropy daemon)
sudo apt install haveged

# Start service
sudo systemctl start haveged
sudo systemctl enable haveged

# Check entropy (should be consistently high)
cat /proc/sys/kernel/random/entropy_avail
```

### rng-tools with RDRAND

```bash
# Use CPU's RDRAND instruction
sudo rngd -r /dev/hwrng

# Or configure in /etc/default/rng-tools
HRNGDEVICE=/dev/hwrng
```

---

## Windows

```powershell
# PowerShell random number
Get-Random -Minimum 0 -Maximum 100

# Random bytes (using .NET)
$bytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RNGCryptoServiceProvider]::new()
$rng.GetBytes($bytes)
[Convert]::ToBase64String($bytes)

# Random GUID
[guid]::NewGuid()

# Random password
Add-Type -AssemblyName System.Web
[System.Web.Security.Membership]::GeneratePassword(20, 5)
```

---

## Best Practices

1. **Use /dev/urandom** for most applications (not /dev/random)
2. **Use secrets module** in Python (not random module)
3. **Use crypto/rand** in Go (not math/rand)
4. **Check entropy** on headless servers
5. **Use hardware RNG** when available
6. **Never seed with time** for cryptographic purposes
7. **Use OS-provided RNG** (don't roll your own)

---