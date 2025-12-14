---
title: "Key Derivation Functions"
date: 2024-12-12
draft: false
category: "crypto"
tags: ["cryptography", "kdf", "password-hashing", "crypto-knowhow", "mathematics"]
---


Key Derivation Functions (KDFs) for password hashing and key derivation.

---

## Password-Based KDFs

### PBKDF2 (Password-Based Key Derivation Function 2)

$$
DK = PBKDF2(password, salt, iterations, dkLen)
$$

**Algorithm**:

$$
\begin{aligned}
T_i &= F(password, salt, iterations, i) \\
F(password, salt, c, i) &= U_1 \oplus U_2 \oplus \cdots \oplus U_c
\end{aligned}
$$

Where:

$$
\begin{aligned}
U_1 &= PRF(password, salt \| INT(i)) \\
U_j &= PRF(password, U_{j-1})
\end{aligned}
$$

**Iterations**: Minimum 100,000 for PBKDF2-HMAC-SHA256

### Argon2 (Winner of Password Hashing Competition)

$$
H = Argon2(password, salt, t, m, p)
$$

**Parameters**:
- $t$: Time cost (iterations)
- $m$: Memory cost (KB)
- $p$: Parallelism degree

**Variants**:
- **Argon2d**: Data-dependent (GPU-resistant)
- **Argon2i**: Data-independent (side-channel resistant)
- **Argon2id**: Hybrid (recommended)

**Memory-hard**: Requires large memory, making GPU/ASIC attacks expensive.

### scrypt

$$
DK = scrypt(password, salt, N, r, p, dkLen)
$$

**Parameters**:
- $N$: CPU/memory cost (power of 2)
- $r$: Block size
- $p$: Parallelization

**Memory required**: $\approx 128 \cdot N \cdot r$ bytes

---

## Python Implementation

### Argon2 (Recommended)

```python
from argon2 import PasswordHasher

ph = PasswordHasher(
    time_cost=2,        # iterations
    memory_cost=65536,  # 64 MB
    parallelism=4,      # threads
    hash_len=32,        # output length
    salt_len=16         # salt length
)

# Hash password
password = "user_password"
hash = ph.hash(password)

# Verify password
try:
    ph.verify(hash, password)
    print("✅ Password correct!")
except:
    print("❌ Password incorrect!")

# Check if rehash needed (params changed)
if ph.check_needs_rehash(hash):
    hash = ph.hash(password)
```

### PBKDF2

```python
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import os

password = b"user_password"
salt = os.urandom(16)

kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=100000,
)

key = kdf.derive(password)

# Verify
kdf2 = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=100000,
)

try:
    kdf2.verify(password, key)
    print("✅ Password correct!")
except:
    print("❌ Password incorrect!")
```

### scrypt

```python
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

password = b"user_password"
salt = os.urandom(16)

kdf = Scrypt(
    salt=salt,
    length=32,
    n=2**14,  # CPU/memory cost
    r=8,      # block size
    p=1,      # parallelization
)

key = kdf.derive(password)
```

---

## Extract-Expand KDFs

### HKDF (HMAC-based KDF)

**Extract**:

$$
PRK = HKDF\text{-}Extract(salt, IKM)
$$

**Expand**:

$$
OKM = HKDF\text{-}Expand(PRK, info, L)
$$

**Use case**: Derive multiple keys from shared secret (e.g., after ECDH)

```python
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

# Shared secret from ECDH
shared_secret = b"..."

# Derive encryption and MAC keys
hkdf = HKDF(
    algorithm=hashes.SHA256(),
    length=64,  # 32 for enc + 32 for MAC
    salt=None,
    info=b'handshake data',
)

key_material = hkdf.derive(shared_secret)
enc_key = key_material[:32]
mac_key = key_material[32:]
```

---

## Comparison

| Algorithm | Type | Memory-Hard | Speed | Use Case |
|-----------|------|-------------|-------|----------|
| **Argon2id** | Password | Yes | Slow | Password hashing (best) |
| **scrypt** | Password | Yes | Slow | Password hashing |
| **PBKDF2** | Password | No | Slow | Legacy, still acceptable |
| **bcrypt** | Password | No | Slow | Password hashing |
| **HKDF** | Extract-expand | No | Fast | Key derivation from shared secret |

---

## Security Recommendations

### Password Hashing

1. **Use Argon2id** (or scrypt if unavailable)
2. **Minimum parameters**:
   - Argon2: `time_cost=2, memory_cost=65536 (64MB), parallelism=4`
   - scrypt: `N=2^14, r=8, p=1`
   - PBKDF2: `iterations=100000+`
3. **Always use salt** (random, unique per password)
4. **Salt length**: 16+ bytes

### Key Derivation

1. **Use HKDF** for deriving keys from shared secrets
2. **Include context** in `info` parameter
3. **Separate keys** for different purposes

---