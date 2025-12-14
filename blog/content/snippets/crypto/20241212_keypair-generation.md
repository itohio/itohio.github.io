---
title: "Generate Public/Private Key Pairs"
date: 2024-12-12
draft: false
category: "crypto"
tags: ["cryptography", "ssh", "rsa", "ed25519", "ecc", "crypto-knowhow"]
---


Generate public/private key pairs on Linux for various cryptographic purposes.

---

## SSH Keys (Ed25519 - Recommended)

```bash
# Generate Ed25519 key pair
ssh-keygen -t ed25519 -C "your_email@example.com"

# Specify output file
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_custom -C "comment"

# Without passphrase (not recommended)
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519

# View public key
cat ~/.ssh/id_ed25519.pub

# View fingerprint
ssh-keygen -lf ~/.ssh/id_ed25519.pub
```

---

## SSH Keys (RSA)

```bash
# Generate RSA key pair (4096 bits)
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Generate RSA key pair (2048 bits - minimum)
ssh-keygen -t rsa -b 2048 -C "your_email@example.com"

# Specify output file
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_custom

# Convert to PEM format
ssh-keygen -p -m PEM -f ~/.ssh/id_rsa
```

---

## SSH Keys (ECDSA)

```bash
# Generate ECDSA key pair (256-bit)
ssh-keygen -t ecdsa -b 256 -C "your_email@example.com"

# Generate ECDSA key pair (384-bit)
ssh-keygen -t ecdsa -b 384 -C "your_email@example.com"

# Generate ECDSA key pair (521-bit)
ssh-keygen -t ecdsa -b 521 -C "your_email@example.com"
```

---

## OpenSSL RSA Keys

```bash
# Generate private key (2048-bit)
openssl genrsa -out private.pem 2048

# Generate private key (4096-bit)
openssl genrsa -out private.pem 4096

# Generate encrypted private key (AES-256)
openssl genrsa -aes256 -out private.pem 4096

# Extract public key
openssl rsa -in private.pem -pubout -out public.pem

# View private key
openssl rsa -in private.pem -text -noout

# View public key
openssl rsa -pubin -in public.pem -text -noout

# Remove passphrase from private key
openssl rsa -in private.pem -out private_nopass.pem
```

---

## OpenSSL ECC Keys

```bash
# List available curves
openssl ecparam -list_curves

# Generate ECC private key (secp256r1 / prime256v1)
openssl ecparam -name prime256v1 -genkey -noout -out private.pem

# Generate ECC private key (secp384r1)
openssl ecparam -name secp384r1 -genkey -noout -out private.pem

# Generate ECC private key (secp521r1)
openssl ecparam -name secp521r1 -genkey -noout -out private.pem

# Generate encrypted ECC private key
openssl ecparam -name prime256v1 -genkey | openssl ec -aes256 -out private.pem

# Extract public key
openssl ec -in private.pem -pubout -out public.pem

# View private key
openssl ec -in private.pem -text -noout

# View public key
openssl ec -pubin -in public.pem -text -noout
```

---

## OpenSSL Ed25519 Keys

```bash
# Generate Ed25519 private key
openssl genpkey -algorithm Ed25519 -out private.pem

# Extract public key
openssl pkey -in private.pem -pubout -out public.pem

# View private key
openssl pkey -in private.pem -text -noout

# View public key
openssl pkey -pubin -in public.pem -text -noout
```

---

## GPG/PGP Keys

```bash
# Generate GPG key (interactive)
gpg --full-generate-key

# Generate GPG key (batch mode)
gpg --batch --generate-key <<EOF
Key-Type: RSA
Key-Length: 4096
Subkey-Type: RSA
Subkey-Length: 4096
Name-Real: Your Name
Name-Email: your_email@example.com
Expire-Date: 2y
Passphrase: your_passphrase
%commit
EOF

# Generate Ed25519 key
gpg --quick-generate-key "Your Name <email@example.com>" ed25519 default 2y

# List keys
gpg --list-keys
gpg --list-secret-keys

# Export public key
gpg --export --armor your_email@example.com > public.asc

# Export private key
gpg --export-secret-keys --armor your_email@example.com > private.asc

# Export key to file
gpg --output public.gpg --export your_email@example.com
gpg --output private.gpg --export-secret-keys your_email@example.com
```

---

## Age Keys (Modern Alternative)

```bash
# Install age
sudo apt install age  # Debian/Ubuntu
brew install age      # macOS

# Generate key pair
age-keygen -o key.txt

# Output:
# Public key: age1ql3z7hjy54pw3hyww5ayyfg7zqgvc7w3j2elw8zmrj2kg5sfn9aqmcac8p
# (private key stored in key.txt)

# View public key
age-keygen -y key.txt
```

---

## Key Formats

### PEM Format (Privacy-Enhanced Mail)

```bash
# RSA private key (PKCS#1)
-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----

# RSA private key (PKCS#8)
-----BEGIN PRIVATE KEY-----
...
-----END PRIVATE KEY-----

# Public key
-----BEGIN PUBLIC KEY-----
...
-----END PUBLIC KEY-----
```

### Convert Between Formats

```bash
# PKCS#1 to PKCS#8
openssl pkcs8 -topk8 -inform PEM -outform PEM -nocrypt \
  -in private_pkcs1.pem -out private_pkcs8.pem

# PKCS#8 to PKCS#1
openssl rsa -in private_pkcs8.pem -out private_pkcs1.pem

# PEM to DER
openssl rsa -in private.pem -outform DER -out private.der

# DER to PEM
openssl rsa -in private.der -inform DER -out private.pem

# SSH to PEM
ssh-keygen -p -m PEM -f ~/.ssh/id_rsa
```

---

## Key Permissions

```bash
# Set correct permissions for private keys
chmod 600 ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_rsa
chmod 600 private.pem

# Set correct permissions for public keys
chmod 644 ~/.ssh/id_ed25519.pub
chmod 644 ~/.ssh/id_rsa.pub
chmod 644 public.pem

# SSH directory permissions
chmod 700 ~/.ssh
```

---

## Key Information

```bash
# SSH key fingerprint
ssh-keygen -lf ~/.ssh/id_ed25519.pub
ssh-keygen -lf ~/.ssh/id_rsa.pub

# SSH key randomart
ssh-keygen -lvf ~/.ssh/id_ed25519.pub

# OpenSSL key info
openssl rsa -in private.pem -text -noout
openssl ec -in private.pem -text -noout
openssl pkey -in private.pem -text -noout

# GPG key fingerprint
gpg --fingerprint your_email@example.com
```

---

## Best Practices

1. **Key Type Selection**:
   - ✅ Ed25519 (modern, fast, secure)
   - ✅ RSA 4096-bit (widely supported)
   - ⚠️ ECDSA (patent concerns)
   - ❌ RSA 2048-bit (minimum, not recommended)
   - ❌ DSA (deprecated)

2. **Passphrase**:
   - Always use a strong passphrase
   - Use ssh-agent to avoid repeated entry
   - Consider using a password manager

3. **Key Storage**:
   - Keep private keys secure (chmod 600)
   - Never share private keys
   - Backup keys securely
   - Use different keys for different purposes

4. **Key Rotation**:
   - Rotate keys periodically (annually)
   - Revoke compromised keys immediately
   - Keep old keys for decryption only

---

## Python Example

```python
from cryptography.hazmat.primitives.asymmetric import rsa, ed25519, ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Generate RSA key pair
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=4096,
    backend=default_backend()
)

# Serialize private key
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

# Serialize public key
public_key = private_key.public_key()
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# Generate Ed25519 key pair
ed_private_key = ed25519.Ed25519PrivateKey.generate()
ed_public_key = ed_private_key.public_key()

# Generate ECC key pair
ec_private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
ec_public_key = ec_private_key.public_key()
```

---