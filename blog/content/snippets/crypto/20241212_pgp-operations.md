---
title: "PGP Signature Operations"
date: 2024-12-12
draft: false
category: "crypto"
tags: ["cryptography", "pgp", "gpg", "signing", "verification", "crypto-knowhow"]
---


PGP/GPG signature operations for files, emails, and git commits.

---

## Generate GPG Key

```bash
# Interactive key generation
gpg --full-generate-key

# Quick generation (Ed25519)
gpg --quick-generate-key "Your Name <email@example.com>" ed25519 sign,cert 2y

# Batch generation
cat > gpg-batch <<EOF
%no-protection
Key-Type: RSA
Key-Length: 4096
Subkey-Type: RSA
Subkey-Length: 4096
Name-Real: Your Name
Name-Email: email@example.com
Expire-Date: 2y
%commit
EOF

gpg --batch --generate-key gpg-batch
```

---

## List Keys

```bash
# List public keys
gpg --list-keys
gpg -k

# List secret keys
gpg --list-secret-keys
gpg -K

# List with fingerprints
gpg --fingerprint

# List specific key
gpg --list-keys email@example.com

# Show key details
gpg --edit-key email@example.com
```

---

## Export Keys

```bash
# Export public key (ASCII armor)
gpg --export --armor email@example.com > public.asc

# Export public key (binary)
gpg --export email@example.com > public.gpg

# Export secret key (ASCII armor)
gpg --export-secret-keys --armor email@example.com > private.asc

# Export secret key (binary)
gpg --export-secret-keys email@example.com > private.gpg

# Export to keyserver
gpg --send-keys KEY_ID
gpg --keyserver keyserver.ubuntu.com --send-keys KEY_ID
```

---

## Import Keys

```bash
# Import public key
gpg --import public.asc

# Import secret key
gpg --import private.asc

# Import from keyserver
gpg --recv-keys KEY_ID
gpg --keyserver keyserver.ubuntu.com --recv-keys KEY_ID

# Search keyserver
gpg --search-keys email@example.com
```

---

## Sign Files

```bash
# Detached signature (separate .sig file)
gpg --detach-sign file.txt
# Creates file.txt.sig

# Detached signature (ASCII armor)
gpg --detach-sign --armor file.txt
# Creates file.txt.asc

# Clear-sign (message + signature in one file)
gpg --clear-sign file.txt
# Creates file.txt.asc

# Sign and encrypt
gpg --sign --encrypt --recipient recipient@example.com file.txt
# Creates file.txt.gpg

# Sign with specific key
gpg --local-user email@example.com --detach-sign file.txt
```

---

## Verify Signatures

```bash
# Verify detached signature
gpg --verify file.txt.sig file.txt
gpg --verify file.txt.asc file.txt

# Verify clear-signed file
gpg --verify file.txt.asc

# Verify and extract signed file
gpg --decrypt file.txt.gpg > file.txt

# Verify with specific keyring
gpg --keyring ./keyring.gpg --verify file.txt.sig file.txt
```

---

## Sign Text/Messages

```bash
# Sign text from stdin
echo "Hello, World!" | gpg --clear-sign

# Sign and output to file
echo "Hello, World!" | gpg --clear-sign > signed.asc

# Verify signed text
gpg --verify signed.asc

# Sign with default key
echo "Hello, World!" | gpg --sign | gpg --decrypt
```

---

## Trust Management

```bash
# Trust a key
gpg --edit-key email@example.com
# In GPG prompt:
# trust
# 5 (ultimate trust)
# quit

# Sign someone's key (web of trust)
gpg --sign-key email@example.com

# Revoke key signature
gpg --edit-key email@example.com
# revsig

# Check key trust
gpg --check-trustdb
```

---

## Key Management

```bash
# Delete public key
gpg --delete-keys email@example.com

# Delete secret key
gpg --delete-secret-keys email@example.com

# Delete both
gpg --delete-secret-and-public-keys email@example.com

# Revoke key
gpg --gen-revoke email@example.com > revoke.asc
gpg --import revoke.asc
gpg --send-keys KEY_ID

# Change passphrase
gpg --edit-key email@example.com
# passwd
# quit

# Add subkey
gpg --edit-key email@example.com
# addkey
# quit
```

---

## Git Integration

See [Setup PGP with Git](20241212_git-pgp-setup.md)

---

## Encrypt and Sign

```bash
# Encrypt and sign for recipient
gpg --encrypt --sign --recipient recipient@example.com file.txt

# Encrypt and sign (ASCII armor)
gpg --encrypt --sign --armor --recipient recipient@example.com file.txt

# Decrypt and verify
gpg --decrypt file.txt.gpg > file.txt
```

---

## Batch Operations

```bash
# Sign multiple files
for file in *.txt; do
    gpg --detach-sign --armor "$file"
done

# Verify multiple files
for file in *.txt; do
    echo "Verifying $file"
    gpg --verify "$file.asc" "$file"
done
```

---

## Python: GPG Operations

```python
import gnupg

# Initialize GPG
gpg = gnupg.GPG()

# List keys
keys = gpg.list_keys()
for key in keys:
    print(f"{key['keyid']}: {key['uids']}")

# Import key
with open('public.asc', 'r') as f:
    import_result = gpg.import_keys(f.read())
    print(f"Imported: {import_result.count}")

# Sign data
signed_data = gpg.sign("Hello, World!", keyid='YOUR_KEY_ID')
print(signed_data)

# Verify signature
verified = gpg.verify(str(signed_data))
print(f"Valid: {verified.valid}")
print(f"Fingerprint: {verified.fingerprint}")

# Encrypt and sign
encrypted = gpg.encrypt(
    "Hello, World!",
    recipients=['recipient@example.com'],
    sign='sender@example.com'
)
print(encrypted)

# Decrypt and verify
decrypted = gpg.decrypt(str(encrypted))
print(f"Decrypted: {decrypted.data.decode()}")
print(f"Valid signature: {decrypted.valid}")
```

---

## Configuration

```bash
# GPG config file
nano ~/.gnupg/gpg.conf

# Recommended settings:
# Use SHA-512 for hashing
personal-digest-preferences SHA512 SHA384 SHA256
cert-digest-algo SHA512

# Use AES-256 for encryption
personal-cipher-preferences AES256 AES192 AES

# Disable weak algorithms
disable-cipher-algo 3DES
weak-digest SHA1

# Show long key IDs
keyid-format 0xlong

# Show fingerprints
with-fingerprint

# Use key server
keyserver hkps://keys.openpgp.org
```

---

## Keyserver Operations

```bash
# Upload key
gpg --send-keys KEY_ID

# Upload to specific keyserver
gpg --keyserver keyserver.ubuntu.com --send-keys KEY_ID

# Receive key
gpg --recv-keys KEY_ID

# Refresh keys from keyserver
gpg --refresh-keys

# Search for key
gpg --search-keys email@example.com
```

---

## Troubleshooting

```bash
# Fix "No public key" error
gpg --recv-keys KEY_ID

# Fix "gpg: signing failed: Inappropriate ioctl for device"
export GPG_TTY=$(tty)
echo 'export GPG_TTY=$(tty)' >> ~/.bashrc

# Fix permission issues
chmod 700 ~/.gnupg
chmod 600 ~/.gnupg/*

# Rebuild trust database
gpg --check-trustdb
gpg --update-trustdb

# Test GPG
echo "test" | gpg --clear-sign
```

---

## Best Practices

1. **Key Generation**:
   - Use Ed25519 or RSA 4096-bit
   - Set expiration date (2 years recommended)
   - Use strong passphrase
   - Create revocation certificate

2. **Key Management**:
   - Backup private keys securely
   - Keep revocation certificate safe
   - Rotate keys periodically
   - Use subkeys for signing

3. **Signing**:
   - Always use detached signatures for files
   - Use clear-sign for text messages
   - Verify signatures before trusting

4. **Trust**:
   - Only trust keys you've verified
   - Use web of trust carefully
   - Verify fingerprints in person

---