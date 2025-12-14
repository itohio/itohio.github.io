---
title: "Digital Signatures"
date: 2024-12-12
draft: false
category: "crypto"
tags: ["cryptography", "signatures", "ecdsa", "eddsa", "crypto-knowhow", "mathematics"]
---


Digital signature algorithms with mathematical foundations.

---

## Mathematical Definition

A digital signature scheme consists of:

$$
\begin{aligned}
(pk, sk) &\leftarrow KeyGen() \\
\sigma &= Sign(sk, M) \\
\{0,1\} &= Verify(pk, M, \sigma)
\end{aligned}
$$

**Properties**:
- **Authentication**: Proves message from holder of $sk$
- **Integrity**: Detects tampering
- **Non-repudiation**: Signer can't deny signing

---

## RSA Signatures (RSA-PSS)

### Signing

$$
\sigma = (H(M))^d \mod n
$$

### Verification

$$
H(M) \stackrel{?}{=} \sigma^e \mod n
$$

Where $H$ is a hash function (SHA-256).

---

## ECDSA (Elliptic Curve DSA)

### Signing

1. Choose random $k \in [1, n-1]$
2. Compute $R = k \cdot G = (x_R, y_R)$
3. Compute $r = x_R \mod n$
4. Compute $s = k^{-1}(H(M) + r \cdot sk) \mod n$

$$
\sigma = (r, s)
$$

### Verification

1. Compute $w = s^{-1} \mod n$
2. Compute $u_1 = H(M) \cdot w \mod n$
3. Compute $u_2 = r \cdot w \mod n$
4. Compute $R' = u_1 \cdot G + u_2 \cdot pk$
5. Verify $r \stackrel{?}{=} x_{R'} \mod n$

### Why It Works

$$
\begin{aligned}
R' &= u_1 \cdot G + u_2 \cdot pk \\
&= (H(M) \cdot s^{-1}) \cdot G + (r \cdot s^{-1}) \cdot (sk \cdot G) \\
&= s^{-1}(H(M) + r \cdot sk) \cdot G \\
&= k \cdot G = R
\end{aligned}
$$

---

## EdDSA (Ed25519)

### Signing

1. Compute $r = H(hash\_prefix, M)$
2. Compute $R = r \cdot G$
3. Compute $h = H(R, pk, M)$
4. Compute $s = (r + h \cdot sk) \mod \ell$

$$
\sigma = (R, s)
$$

### Verification

$$
s \cdot G \stackrel{?}{=} R + H(R, pk, M) \cdot pk
$$

**Advantages over ECDSA**:
- Deterministic (no random $k$)
- Faster
- Smaller signatures
- Side-channel resistant

---

## Python Implementation

### Ed25519 (Recommended)

```python
from cryptography.hazmat.primitives.asymmetric import ed25519

# Generate key pair
private_key = ed25519.Ed25519PrivateKey.generate()
public_key = private_key.public_key()

# Sign
message = b"Important message"
signature = private_key.sign(message)

# Verify
try:
    public_key.verify(signature, message)
    print("✅ Signature valid!")
except:
    print("❌ Signature invalid!")
```

### ECDSA

```python
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

# Generate key pair
private_key = ec.generate_private_key(ec.SECP256K1())
public_key = private_key.public_key()

# Sign
signature = private_key.sign(
    message,
    ec.ECDSA(hashes.SHA256())
)

# Verify
try:
    public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
    print("✅ Signature valid!")
except:
    print("❌ Signature invalid!")
```

---

## Schnorr Signatures (Bitcoin Taproot)

### Signing

1. Choose random $k$
2. Compute $R = k \cdot G$
3. Compute $e = H(R \| pk \| M)$
4. Compute $s = k + e \cdot sk$

$$
\sigma = (R, s)
$$

### Verification

$$
s \cdot G \stackrel{?}{=} R + H(R \| pk \| M) \cdot pk
$$

**Advantages**:
- Simple and elegant
- Supports key aggregation (MuSig)
- Smaller multi-signatures

---

## Comparison

| Algorithm | Signature Size | Speed | Security | Use Case |
|-----------|----------------|-------|----------|----------|
| **Ed25519** | 64 bytes | Very fast | High | Modern, SSH |
| **ECDSA** | ~72 bytes | Fast | High | Bitcoin, Ethereum |
| **RSA-PSS** | 256-512 bytes | Slow | High | Legacy, TLS |
| **Schnorr** | 64 bytes | Fast | High | Bitcoin Taproot |

---