---
title: "Setup PGP with Git (Auto-sign Commits)"
date: 2024-12-12
draft: false
category: "crypto"
tags: ["cryptography", "pgp", "gpg", "git", "signing", "crypto-knowhow"]
---


Setup GPG/PGP to automatically sign Git commits and tags.

---

## Generate GPG Key for Git

```bash
# Generate GPG key (use same email as Git)
gpg --full-generate-key

# Select:
# 1. RSA and RSA (or Ed25519)
# 4096 bits (for RSA)
# 2y expiration
# Your name
# Your Git email
# Passphrase

# Or quick generation
gpg --quick-generate-key "Your Name <git@email.com>" rsa4096 sign 2y
```

---

## List GPG Keys

```bash
# List keys
gpg --list-secret-keys --keyid-format=long

# Output:
# /home/user/.gnupg/pubring.kbx
# -------------------------------
# sec   rsa4096/ABCD1234EFGH5678 2024-01-01 [SC] [expires: 2026-01-01]
#       1234567890ABCDEF1234567890ABCDEF12345678
# uid                 [ultimate] Your Name <git@email.com>
# ssb   rsa4096/1234567890ABCDEF 2024-01-01 [E] [expires: 2026-01-01]

# The key ID is: ABCD1234EFGH5678
```

---

## Configure Git to Use GPG

```bash
# Set GPG key for Git
git config --global user.signingkey ABCD1234EFGH5678

# Enable auto-signing for commits
git config --global commit.gpgsign true

# Enable auto-signing for tags
git config --global tag.gpgsign true

# Set GPG program (if needed)
git config --global gpg.program gpg

# For GPG2
git config --global gpg.program gpg2
```

---

## Configure GPG Agent

```bash
# Add to ~/.bashrc or ~/.zshrc
export GPG_TTY=$(tty)

# For longer cache time, edit ~/.gnupg/gpg-agent.conf
default-cache-ttl 3600
max-cache-ttl 86400

# Restart GPG agent
gpgconf --kill gpg-agent
gpg-agent --daemon
```

---

## Sign Commits

```bash
# Sign a commit (if auto-sign disabled)
git commit -S -m "Signed commit message"

# Sign all commits (if auto-sign enabled)
git commit -m "This will be signed automatically"

# Verify commit signature
git log --show-signature

# Verify specific commit
git verify-commit HEAD

# Show commit with signature
git show --show-signature HEAD
```

---

## Sign Tags

```bash
# Create signed tag
git tag -s v1.0.0 -m "Version 1.0.0"

# Create signed annotated tag (if auto-sign enabled)
git tag -a v1.0.0 -m "Version 1.0.0"

# Verify tag signature
git tag -v v1.0.0

# Show tag with signature
git show v1.0.0
```

---

## GitHub Integration

```bash
# Export public key
gpg --armor --export ABCD1234EFGH5678

# Copy output and add to GitHub:
# 1. Go to GitHub Settings
# 2. SSH and GPG keys
# 3. New GPG key
# 4. Paste public key
# 5. Add GPG key

# Verify on GitHub
# Commits will show "Verified" badge
```

---

## GitLab Integration

```bash
# Export public key
gpg --armor --export ABCD1234EFGH5678

# Add to GitLab:
# 1. Go to GitLab Settings
# 2. GPG Keys
# 3. Add new key
# 4. Paste public key
# 5. Add key

# Verify on GitLab
# Commits will show "Verified" badge
```

---

## Gitea/Forgejo Integration

```bash
# Export public key
gpg --armor --export ABCD1234EFGH5678

# Add to Gitea/Forgejo:
# 1. Go to Settings
# 2. SSH / GPG Keys
# 3. Add GPG Key
# 4. Paste public key
# 5. Add Key
```

---

## Sign Previous Commits

```bash
# Sign last commit
git commit --amend --no-edit -S

# Sign multiple commits (interactive rebase)
git rebase -i HEAD~5

# In editor, change 'pick' to 'edit' for commits to sign
# Then for each commit:
git commit --amend --no-edit -S
git rebase --continue

# Force push (if already pushed)
git push --force-with-lease
```

---

## Verify Repository Commits

```bash
# Verify all commits
git log --show-signature

# Verify commits in range
git log --show-signature HEAD~10..HEAD

# Show only signed commits
git log --show-signature | grep -B 5 "Good signature"

# Show only unsigned commits
git log --pretty=format:"%H %an %s" | while read commit; do
    if ! git verify-commit $(echo $commit | cut -d' ' -f1) 2>/dev/null; then
        echo "Unsigned: $commit"
    fi
done
```

---

## Git Config File

```bash
# View Git config
cat ~/.gitconfig

# Example configuration:
[user]
    name = Your Name
    email = git@email.com
    signingkey = ABCD1234EFGH5678

[commit]
    gpgsign = true

[tag]
    gpgsign = true

[gpg]
    program = gpg
```

---

## Troubleshooting

### "gpg: signing failed: Inappropriate ioctl for device"

```bash
# Fix: Export GPG_TTY
export GPG_TTY=$(tty)

# Add to ~/.bashrc
echo 'export GPG_TTY=$(tty)' >> ~/.bashrc
```

### "gpg: signing failed: No secret key"

```bash
# Check if key exists
gpg --list-secret-keys --keyid-format=long

# Check Git config
git config user.signingkey

# Set correct key
git config --global user.signingkey YOUR_KEY_ID
```

### "gpg: signing failed: Operation cancelled"

```bash
# GPG agent issue - restart
gpgconf --kill gpg-agent
gpg-agent --daemon

# Or use pinentry-tty
echo "pinentry-program /usr/bin/pinentry-tty" >> ~/.gnupg/gpg-agent.conf
gpgconf --kill gpg-agent
```

### "error: gpg failed to sign the data"

```bash
# Test GPG
echo "test" | gpg --clear-sign

# Check GPG agent
ps aux | grep gpg-agent

# Restart GPG agent
gpgconf --kill gpg-agent
gpg-agent --daemon

# Check Git GPG program
git config --global gpg.program gpg
```

---

## Multiple Keys

```bash
# Use different keys for different repos
cd /path/to/repo
git config user.signingkey DIFFERENT_KEY_ID

# Use different keys per directory
# In ~/.gitconfig:
[includeIf "gitdir:~/work/"]
    path = ~/.gitconfig-work

# In ~/.gitconfig-work:
[user]
    email = work@company.com
    signingkey = WORK_KEY_ID
```

---

## Batch Sign Commits

```bash
#!/bin/bash
# sign-commits.sh - Sign all commits in a branch

BRANCH=${1:-main}
BASE=${2:-origin/main}

# Get list of commits
commits=$(git log --pretty=format:"%H" $BASE..$BRANCH)

# Sign each commit
for commit in $commits; do
    echo "Signing $commit"
    GIT_EDITOR=true git rebase --exec "git commit --amend --no-edit -S" $commit^
done

echo "All commits signed!"
```

---

## Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Ensure GPG_TTY is set
export GPG_TTY=$(tty)

# Check if commit will be signed
if ! git config --get commit.gpgsign | grep -q true; then
    echo "Warning: commit.gpgsign is not enabled"
    echo "Enable with: git config commit.gpgsign true"
    exit 1
fi

# Test GPG signing
if ! echo "test" | gpg --clear-sign &>/dev/null; then
    echo "Error: GPG signing failed"
    echo "Check your GPG configuration"
    exit 1
fi

exit 0
```

```bash
# Make executable
chmod +x .git/hooks/pre-commit
```

---

## CI/CD Integration

```yaml
# GitHub Actions
name: Verify Signatures

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
      
      - name: Import GPG keys
        run: |
          # Import trusted keys
          gpg --import trusted-keys.asc
      
      - name: Verify commits
        run: |
          # Verify all commits in PR
          for commit in $(git log --pretty=format:"%H" origin/main..HEAD); do
            if ! git verify-commit $commit; then
              echo "Commit $commit is not signed!"
              exit 1
            fi
          done
```

---

## Best Practices

1. **Key Management**:
   - Use separate key for Git signing
   - Set expiration date (2 years)
   - Backup private key securely
   - Create revocation certificate

2. **Signing**:
   - Enable auto-signing globally
   - Sign all commits and tags
   - Verify signatures before merging
   - Document signing policy in CONTRIBUTING.md

3. **Team Setup**:
   - Share public keys via keyserver
   - Maintain list of trusted keys
   - Verify fingerprints in person
   - Use signed tags for releases

4. **Security**:
   - Use strong passphrase
   - Configure GPG agent timeout
   - Don't share private keys
   - Revoke compromised keys immediately

---