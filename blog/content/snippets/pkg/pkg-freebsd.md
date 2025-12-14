---
title: "pkg Package Manager (FreeBSD)"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "pkg", "freebsd", "bsd", "package-manager"]
---


pkg - Binary package manager for FreeBSD.

---

## Basic Commands

```bash
# Update repository catalog
pkg update

# Upgrade all packages
pkg upgrade

# Install package
pkg install package-name

# Install without confirmation
pkg install -y package-name

# Remove package
pkg delete package-name

# Remove package and dependencies
pkg autoremove

# Search for package
pkg search package-name

# Show package info
pkg info package-name

# List installed packages
pkg info

# Check for updates
pkg version -l "<"
```

---

## Search and Query

```bash
# Search by name
pkg search package-name

# Search by description
pkg search -D keyword

# Search by origin
pkg search -o category/package

# Show package details
pkg info -f package-name

# Show dependencies
pkg info -d package-name

# Show reverse dependencies
pkg info -r package-name

# List files in package
pkg info -l package-name

# Find which package owns file
pkg which /path/to/file
```

---

## Install from Repository

```bash
# Install package
pkg install nginx

# Install specific version
pkg install nginx-1.24.0

# Install from specific repository
pkg install -r FreeBSD package-name

# Reinstall package
pkg install -f package-name

# Download only
pkg fetch package-name

# Install downloaded package
pkg add package-name.txz
```

---

## Build from Ports

```bash
# Update ports tree
portsnap fetch extract
# Or for updates
portsnap fetch update

# Or use Git (modern method)
git clone https://git.FreeBSD.org/ports.git /usr/ports

# Navigate to port
cd /usr/ports/category/package-name

# Show build options
make config

# Build and install
make install clean

# Build without installing
make

# Install built package
make install

# Clean build files
make clean

# Deinstall
make deinstall
```

---

## Ports Management

```bash
# Search ports
cd /usr/ports
make search name=package-name
make search key=keyword

# Show port info
cd /usr/ports/category/package
make describe

# Show dependencies
make all-depends-list

# Show build options
make showconfig

# Update all ports (using portmaster)
pkg install portmaster
portmaster -a

# Update specific port
portmaster category/package-name
```

---

## Repository Configuration

```bash
# Repository config
cat /etc/pkg/FreeBSD.conf

# Custom repository
mkdir -p /usr/local/etc/pkg/repos
cat > /usr/local/etc/pkg/repos/custom.conf <<EOF
custom: {
    url: "pkg+http://example.com/\${ABI}/latest",
    mirror_type: "srv",
    enabled: yes
}
EOF

# Disable default repo
cat > /usr/local/etc/pkg/repos/FreeBSD.conf <<EOF
FreeBSD: { enabled: no }
EOF

# List repositories
pkg -vv | grep url
```

---

## Lock/Unlock Packages

```bash
# Lock package (prevent upgrade)
pkg lock package-name

# Unlock package
pkg unlock package-name

# List locked packages
pkg lock -l

# Lock all packages
pkg lock -a

# Unlock all packages
pkg unlock -a
```

---

## Audit and Security

```bash
# Update vulnerability database
pkg audit -F

# Check for vulnerabilities
pkg audit

# Check specific package
pkg audit package-name

# Show vulnerable packages
pkg audit -r
```

---

## Clean and Maintenance

```bash
# Clean package cache
pkg clean

# Clean all cached packages
pkg clean -a

# Remove orphaned packages
pkg autoremove

# Check for consistency
pkg check -d

# Recompute checksums
pkg check -s

# Check and repair
pkg check -B

# Rebuild package database
pkg check -r
```

---

## Backup and Restore

```bash
# Backup installed packages list
pkg info -q > packages.txt

# Restore packages
cat packages.txt | xargs pkg install

# Create package repository from installed
pkg create -a -o /path/to/repo

# Backup package database
tar -czf pkg-db-backup.tar.gz /var/db/pkg/
```

---

## Statistics

```bash
# Show statistics
pkg stats

# Show repository statistics
pkg stats -r

# Show local statistics
pkg stats -l

# Disk usage
pkg info -s

# Largest packages
pkg info -s | sort -k2 -n -r | head -10
```

---

## Configuration

```bash
# Main config file
cat /usr/local/etc/pkg.conf

# Example settings:
PKG_DBDIR: "/var/db/pkg"
PKG_CACHEDIR: "/var/cache/pkg"
ASSUME_ALWAYS_YES: false
REPOS_DIR: ["/etc/pkg", "/usr/local/etc/pkg/repos"]
```

---

## Troubleshooting

```bash
# Fix broken dependencies
pkg check -d
pkg install -f

# Rebuild database
pkg check -r

# Force reinstall
pkg install -f package-name

# Verbose output
pkg -v install package-name

# Debug mode
pkg -d install package-name

# Check integrity
pkg check -s -a
```

---

## Build from Source (Ports)

### Using Poudriere (Build Farm)

```bash
# Install poudriere
pkg install poudriere

# Create jail
poudriere jail -c -j 13amd64 -v 13.2-RELEASE

# Create ports tree
poudriere ports -c

# Build packages
poudriere bulk -j 13amd64 category/package-name

# Build from list
poudriere bulk -j 13amd64 -f package-list.txt

# Serve built packages
# Configure nginx to serve /usr/local/poudriere/data/packages/
```

---

## Useful Aliases

```bash
# Add to ~/.shrc or ~/.bashrc
alias update='pkg update && pkg upgrade'
alias install='pkg install'
alias remove='pkg delete'
alias search='pkg search'
alias clean='pkg autoremove && pkg clean -a'
```

---

## Common Packages

```bash
# Development tools
pkg install git vim tmux

# Web servers
pkg install nginx apache24

# Databases
pkg install postgresql15-server mysql80-server

# Languages
pkg install python39 go rust node

# System tools
pkg install htop bash zsh
```

---

## Create and Publish FreeBSD Port

### Port Structure

```
mypackage/
├── Makefile
├── distinfo
├── pkg-descr
├── pkg-plist
└── files/
    └── patch-configure
```

### Makefile

```makefile
# Makefile
PORTNAME=       mypackage
PORTVERSION=    1.0.0
CATEGORIES=     sysutils
MASTER_SITES=   https://github.com/username/mypackage/releases/download/v${PORTVERSION}/

MAINTAINER=     email@example.com
COMMENT=        Short description

LICENSE=        MIT
LICENSE_FILE=   ${WRKSRC}/LICENSE

USES=           gmake
USE_GITHUB=     yes
GH_ACCOUNT=     username
GH_PROJECT=     mypackage
GH_TAGNAME=     v${PORTVERSION}

PLIST_FILES=    bin/mypackage \
                man/man1/mypackage.1.gz

.include <bsd.port.mk>
```

### Generate distinfo

```bash
# Generate checksums
make makesum

# Creates distinfo:
TIMESTAMP = 1702387200
SHA256 (username-mypackage-v1.0.0_GH0.tar.gz) = abc123...
SIZE (username-mypackage-v1.0.0_GH0.tar.gz) = 123456
```

### pkg-descr

```
This is a longer description of the package.
It can span multiple lines and should describe
what the package does in detail.

WWW: https://github.com/username/mypackage
```

### pkg-plist

```
bin/mypackage
man/man1/mypackage.1.gz
%%DATADIR%%/config.sample
@sample etc/mypackage/config.conf.sample
```

### Test Port

```bash
# Test build
make

# Test install
sudo make install

# Test deinstall
sudo make deinstall

# Clean
make clean

# Check with portlint
portlint -AC

# Test in clean environment
poudriere testport -j 13amd64 sysutils/mypackage
```

### Submit to FreeBSD Ports

```bash
# 1. Fork FreeBSD ports repository
# https://github.com/freebsd/freebsd-ports

# 2. Create port directory
mkdir -p sysutils/mypackage
cd sysutils/mypackage

# 3. Add files
# Create Makefile, pkg-descr, pkg-plist, distinfo

# 4. Test thoroughly
portlint -AC
poudriere testport -j 13amd64 sysutils/mypackage

# 5. Create patch
cd ../..
git add sysutils/mypackage
git commit -m "sysutils/mypackage: New port"
git format-patch HEAD^

# 6. Submit to Bugzilla
# https://bugs.freebsd.org/bugzilla/
# Category: Ports & Packages
# Attach patch file
```

### Create Custom Package Repository

```bash
# Build packages with Poudriere
sudo pkg install poudriere

# Create jail
sudo poudriere jail -c -j 13amd64 -v 13.2-RELEASE

# Create ports tree
sudo poudriere ports -c

# Build packages
sudo poudriere bulk -j 13amd64 sysutils/mypackage

# Packages are in:
# /usr/local/poudriere/data/packages/13amd64-default/

# Serve with nginx
# Configure nginx to serve /usr/local/poudriere/data/packages/

# Users configure:
# /usr/local/etc/pkg/repos/myrepo.conf
myrepo: {
    url: "https://pkg.example.com/13amd64-default",
    mirror_type: "http",
    enabled: yes
}

# Then: sudo pkg install mypackage
```

### Best Practices

```bash
# 1. Follow Porter's Handbook
# https://docs.freebsd.org/en/books/porters-handbook/

# 2. Use proper USES
USES= gmake pkgconfig

# 3. Handle options
OPTIONS_DEFINE= DOCS EXAMPLES
OPTIONS_DEFAULT= DOCS

DOCS_BUILD_DEPENDS= sphinx-build:textproc/py-sphinx

# 4. Install documentation
post-install-DOCS-on:
    ${MKDIR} ${STAGEDIR}${DOCSDIR}
    ${INSTALL_DATA} ${WRKSRC}/README.md ${STAGEDIR}${DOCSDIR}

# 5. Test with portlint
portlint -AC

# 6. Test with Poudriere
poudriere testport -j 13amd64 category/port
```

---