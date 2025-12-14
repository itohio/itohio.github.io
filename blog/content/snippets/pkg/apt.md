---
title: "APT Package Manager (Debian/Ubuntu)"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "apt", "debian", "ubuntu", "linux", "package-manager"]
---


APT (Advanced Package Tool) for Debian and Ubuntu-based systems.

---

## Basic Commands

```bash
# Update package list
sudo apt update

# Upgrade all packages
sudo apt upgrade

# Full upgrade (may remove packages)
sudo apt full-upgrade
sudo apt dist-upgrade  # Older command

# Install package
sudo apt install package-name

# Install specific version
sudo apt install package-name=version

# Install multiple packages
sudo apt install package1 package2 package3

# Install without prompts
sudo apt install -y package-name

# Remove package
sudo apt remove package-name

# Remove package and config files
sudo apt purge package-name

# Remove unused dependencies
sudo apt autoremove

# Clean package cache
sudo apt clean
sudo apt autoclean
```

---

## Search and Information

```bash
# Search for package
apt search package-name
apt-cache search package-name

# Show package info
apt show package-name
apt-cache show package-name

# List installed packages
apt list --installed

# List upgradable packages
apt list --upgradable

# Show package dependencies
apt-cache depends package-name

# Show reverse dependencies
apt-cache rdepends package-name

# Check if package is installed
dpkg -l | grep package-name
apt list --installed | grep package-name
```

---

## Package Sources

### Add Repository

```bash
# Add PPA (Ubuntu)
sudo add-apt-repository ppa:user/ppa-name
sudo apt update

# Remove PPA
sudo add-apt-repository --remove ppa:user/ppa-name

# Add repository manually
echo "deb http://repository.url/ distribution component" | sudo tee /etc/apt/sources.list.d/repo.list

# Add GPG key
wget -qO - https://repository.url/key.gpg | sudo apt-key add -

# Or with newer method
wget -qO - https://repository.url/key.gpg | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/repo.gpg
```

### sources.list

```bash
# Edit sources
sudo nano /etc/apt/sources.list

# Example entry:
# deb http://archive.ubuntu.com/ubuntu/ jammy main restricted
# deb-src http://archive.ubuntu.com/ubuntu/ jammy main restricted

# List repositories
grep -r --include '*.list' '^deb ' /etc/apt/sources.list /etc/apt/sources.list.d/
```

---

## Hold/Unhold Packages

```bash
# Hold package (prevent upgrade)
sudo apt-mark hold package-name

# Unhold package
sudo apt-mark unhold package-name

# List held packages
apt-mark showhold
```

---

## Download Packages

```bash
# Download package without installing
apt download package-name

# Download source package
apt source package-name

# Download and extract source
apt source --compile package-name

# Show download URLs
apt-get --print-uris install package-name
```

---

## Fix Broken Packages

```bash
# Fix broken dependencies
sudo apt --fix-broken install
sudo apt -f install

# Reconfigure packages
sudo dpkg --configure -a

# Force reinstall
sudo apt install --reinstall package-name

# Clean and update
sudo apt clean
sudo apt update
sudo apt upgrade
```

---

## APT vs APT-GET

```bash
# Modern (apt)
sudo apt update
sudo apt install package-name
sudo apt remove package-name
sudo apt search package-name

# Traditional (apt-get)
sudo apt-get update
sudo apt-get install package-name
sudo apt-get remove package-name
apt-cache search package-name

# apt is recommended for interactive use
# apt-get is more stable for scripts
```

---

## Install from .deb File

```bash
# Install .deb package
sudo dpkg -i package.deb

# Fix dependencies after dpkg
sudo apt install -f

# Or use apt directly (recommended)
sudo apt install ./package.deb

# Remove package installed from .deb
sudo apt remove package-name
```

---

## Build from Source

```bash
# Install build dependencies
sudo apt build-dep package-name

# Or manually install common build tools
sudo apt install build-essential

# Download source
apt source package-name
cd package-name-version

# Build
dpkg-buildpackage -us -uc

# Install built package
sudo dpkg -i ../package-name_version_arch.deb
```

---

## Unattended Upgrades

```bash
# Install
sudo apt install unattended-upgrades

# Configure
sudo dpkg-reconfigure unattended-upgrades

# Edit config
sudo nano /etc/apt/apt.conf.d/50unattended-upgrades

# Test
sudo unattended-upgrades --dry-run --debug
```

---

## APT Configuration

```bash
# Configuration file
sudo nano /etc/apt/apt.conf

# Example configurations:
# APT::Get::Assume-Yes "true";
# APT::Install-Recommends "false";
# APT::Install-Suggests "false";

# Per-command config
sudo apt -o APT::Install-Recommends=false install package-name
```

---

## Cache Management

```bash
# Show cache statistics
apt-cache stats

# Cache location
ls /var/cache/apt/archives/

# Clean cache (remove downloaded packages)
sudo apt clean

# Clean old packages only
sudo apt autoclean

# Show cache size
du -sh /var/cache/apt/archives/
```

---

## Logs

```bash
# View installation history
cat /var/log/apt/history.log

# View detailed logs
cat /var/log/apt/term.log

# View dpkg log
cat /var/log/dpkg.log

# Recently installed packages
grep " install " /var/log/dpkg.log | tail -20
```

---

## Common Packages

```bash
# Development tools
sudo apt install build-essential git curl wget

# Python development
sudo apt install python3 python3-pip python3-venv

# Node.js
sudo apt install nodejs npm

# Docker
sudo apt install docker.io docker-compose

# System tools
sudo apt install htop tmux vim neovim

# Network tools
sudo apt install net-tools dnsutils curl wget
```

---

## Troubleshooting

```bash
# Lock file issues
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock*
sudo dpkg --configure -a
sudo apt update

# Hash sum mismatch
sudo rm -rf /var/lib/apt/lists/*
sudo apt clean
sudo apt update

# Broken packages
sudo apt --fix-broken install
sudo dpkg --configure -a
sudo apt autoremove

# Check for errors
sudo apt check
```

---

## Security Updates

```bash
# List security updates
sudo apt list --upgradable | grep -i security

# Install security updates only
sudo apt upgrade -s | grep -i security

# Automatic security updates
sudo apt install unattended-upgrades apt-listchanges
```

---

## Useful Aliases

```bash
# Add to ~/.bashrc
alias update='sudo apt update && sudo apt upgrade'
alias install='sudo apt install'
alias remove='sudo apt remove'
alias search='apt search'
alias clean='sudo apt autoremove && sudo apt clean'
```

---

## Create and Publish Debian Package

### Package Structure

```
mypackage-1.0/
├── DEBIAN/
│   ├── control
│   ├── postinst
│   ├── prerm
│   └── copyright
├── usr/
│   ├── bin/
│   │   └── myapp
│   ├── share/
│   │   ├── doc/
│   │   │   └── mypackage/
│   │   │       └── README.md
│   │   └── man/
│   │       └── man1/
│   │           └── myapp.1.gz
│   └── lib/
│       └── mypackage/
└── etc/
    └── mypackage/
        └── config.conf
```

### DEBIAN/control

```
Package: mypackage
Version: 1.0.0
Section: utils
Priority: optional
Architecture: amd64
Depends: libc6 (>= 2.34), python3 (>= 3.8)
Maintainer: Your Name <email@example.com>
Description: Short description
 Long description goes here.
 It can span multiple lines.
 Each line should start with a space.
Homepage: https://github.com/username/mypackage
```

### Build Package

```bash
# Create package structure
mkdir -p mypackage-1.0/DEBIAN
mkdir -p mypackage-1.0/usr/bin

# Add files
cp myapp mypackage-1.0/usr/bin/
chmod 755 mypackage-1.0/usr/bin/myapp

# Create control file
cat > mypackage-1.0/DEBIAN/control <<EOF
Package: mypackage
Version: 1.0.0
Section: utils
Priority: optional
Architecture: amd64
Maintainer: Your Name <email@example.com>
Description: My package
 Long description
EOF

# Build package
dpkg-deb --build mypackage-1.0

# This creates: mypackage-1.0.deb

# Check package
lintian mypackage-1.0.deb
```

### Post-install Script

```bash
# DEBIAN/postinst
#!/bin/bash
set -e

# Create user
if ! id myapp >/dev/null 2>&1; then
    useradd -r -s /bin/false myapp
fi

# Create directories
mkdir -p /var/log/mypackage
chown myapp:myapp /var/log/mypackage

# Enable service
systemctl enable mypackage.service
systemctl start mypackage.service

exit 0
```

```bash
# Make executable
chmod 755 DEBIAN/postinst
```

### Publish to PPA (Ubuntu)

```bash
# Install tools
sudo apt install devscripts dput

# Create source package
cd mypackage-1.0
debuild -S -sa

# Sign with GPG
debsign -k YOUR_GPG_KEY mypackage_1.0.0_source.changes

# Upload to PPA
dput ppa:username/ppa-name mypackage_1.0.0_source.changes
```

### Publish to Custom Repository

```bash
# Install reprepro
sudo apt install reprepro

# Create repository structure
mkdir -p /var/www/apt/{conf,dists,pool}

# Create conf/distributions
cat > /var/www/apt/conf/distributions <<EOF
Origin: Your Name
Label: Your Repository
Codename: jammy
Architectures: amd64 arm64
Components: main
Description: My custom repository
SignWith: YOUR_GPG_KEY
EOF

# Add package to repository
reprepro -b /var/www/apt includedeb jammy mypackage-1.0.deb

# Serve with nginx
sudo apt install nginx
# Configure nginx to serve /var/www/apt

# Users add repository:
# echo "deb https://apt.example.com jammy main" | sudo tee /etc/apt/sources.list.d/myrepo.list
# wget -qO - https://apt.example.com/key.gpg | sudo apt-key add -
# sudo apt update
# sudo apt install mypackage
```

---