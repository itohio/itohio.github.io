---
title: "dpkg Package Manager"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "dpkg", "debian", "ubuntu", "linux", "package-manager"]
---


dpkg - Low-level package manager for Debian-based systems.

---

## Basic Commands

```bash
# Install package
sudo dpkg -i package.deb

# Remove package (keep config)
sudo dpkg -r package-name

# Purge package (remove config)
sudo dpkg -P package-name

# List installed packages
dpkg -l

# List specific package
dpkg -l package-name

# Show package info
dpkg -s package-name

# List files in package
dpkg -L package-name

# Find which package owns a file
dpkg -S /path/to/file

# Extract package contents
dpkg -x package.deb /output/dir

# Extract control files
dpkg -e package.deb /output/dir
```

---

## Query Packages

```bash
# List all installed packages
dpkg -l

# List with pattern
dpkg -l | grep pattern

# Show package status
dpkg -s package-name

# List files installed by package
dpkg -L package-name

# Search for file owner
dpkg -S /usr/bin/vim

# List package contents (not installed)
dpkg -c package.deb

# Show package info (not installed)
dpkg -I package.deb
```

---

## Package Status

```bash
# Check if package is installed
dpkg -l package-name | grep ^ii

# Show package version
dpkg -l package-name | awk '{print $3}'

# List configuration files
dpkg-query -W -f='${Conffiles}\n' package-name

# Verify installed files
dpkg -V package-name
```

---

## Fix Broken Packages

```bash
# Reconfigure all packages
sudo dpkg --configure -a

# Force reconfigure specific package
sudo dpkg-reconfigure package-name

# Force install (dangerous!)
sudo dpkg -i --force-all package.deb

# Fix dependencies
sudo apt install -f
```

---

## Build Packages

```bash
# Build from source directory
dpkg-buildpackage -us -uc

# Build binary only
dpkg-buildpackage -b -us -uc

# Build source only
dpkg-buildpackage -S -us -uc

# Sign package
dpkg-buildpackage -rfakeroot
```

---

## Extract and Repack

```bash
# Extract .deb
mkdir package
dpkg-deb -x package.deb package/
dpkg-deb -e package.deb package/DEBIAN

# Modify contents
# ... edit files ...

# Repack
dpkg-deb -b package new-package.deb

# Check package
lintian new-package.deb
```

---

## Package Information

```bash
# Show control file
dpkg -I package.deb

# Show package description
dpkg -s package-name | grep Description -A 10

# Show dependencies
dpkg -s package-name | grep Depends

# Show package size
dpkg -s package-name | grep Installed-Size

# Show all package fields
dpkg -s package-name
```

---

## Database Management

```bash
# Database location
ls /var/lib/dpkg/

# Status file
cat /var/lib/dpkg/status

# Available packages
cat /var/lib/dpkg/available

# Backup database
sudo cp -r /var/lib/dpkg /var/lib/dpkg.backup

# Restore database
sudo cp -r /var/lib/dpkg.backup /var/lib/dpkg
```

---

## Logs

```bash
# dpkg log
cat /var/log/dpkg.log

# Recently installed
grep " install " /var/log/dpkg.log | tail -20

# Recently removed
grep " remove " /var/log/dpkg.log | tail -20

# Recently upgraded
grep " upgrade " /var/log/dpkg.log | tail -20
```

---

## Advanced Usage

```bash
# Force options
--force-depends       # Ignore dependency problems
--force-conflicts     # Install conflicting packages
--force-overwrite     # Overwrite files from other packages
--force-downgrade     # Install older version
--force-all           # Force everything (dangerous!)

# Example
sudo dpkg -i --force-overwrite package.deb

# Dry run
sudo dpkg --dry-run -i package.deb
```

---

## dpkg-query

```bash
# List packages by pattern
dpkg-query -l '*python*'

# Show package status
dpkg-query -W -f='${Status}\n' package-name

# Show version
dpkg-query -W -f='${Version}\n' package-name

# Custom format
dpkg-query -W -f='${Package}\t${Version}\t${Status}\n'

# List all packages with versions
dpkg-query -W -f='${Package}\t${Version}\n' | column -t
```

---

## dpkg-deb

```bash
# Extract package
dpkg-deb -x package.deb output-dir

# Extract control files
dpkg-deb -e package.deb output-dir

# Show package info
dpkg-deb -I package.deb

# Show package contents
dpkg-deb -c package.deb

# Build package
dpkg-deb -b package-dir output.deb

# Check package
dpkg-deb --info package.deb
```

---

## Create Simple Package

```bash
# Create package structure
mkdir -p mypackage/DEBIAN
mkdir -p mypackage/usr/local/bin

# Create control file
cat > mypackage/DEBIAN/control <<EOF
Package: mypackage
Version: 1.0
Section: utils
Priority: optional
Architecture: all
Maintainer: Your Name <email@example.com>
Description: My custom package
 Long description here
EOF

# Add files
cp myscript.sh mypackage/usr/local/bin/
chmod +x mypackage/usr/local/bin/myscript.sh

# Build package
dpkg-deb -b mypackage mypackage_1.0_all.deb

# Install
sudo dpkg -i mypackage_1.0_all.deb
```

---

## Troubleshooting

```bash
# Lock file issues
sudo rm /var/lib/dpkg/lock
sudo rm /var/lib/dpkg/lock-frontend
sudo dpkg --configure -a

# Corrupted package database
sudo rm /var/lib/dpkg/updates/*
sudo dpkg --configure -a

# Force remove broken package
sudo dpkg --remove --force-remove-reinstreq package-name

# Reinstall package
sudo apt install --reinstall package-name
```

---

## Useful Scripts

```bash
# List largest installed packages
dpkg-query -W -f='${Installed-Size}\t${Package}\n' | sort -rn | head -20

# List packages by install date
grep " install " /var/log/dpkg.log | tail -50

# Find orphaned packages
deborphan

# Clean up old config files
dpkg -l | grep ^rc | awk '{print $2}' | xargs sudo dpkg --purge
```

---