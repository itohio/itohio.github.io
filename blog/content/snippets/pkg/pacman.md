---
title: "pacman Package Manager (Arch Linux)"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "pacman", "arch-linux", "linux", "package-manager"]
---


pacman - Package manager for Arch Linux and derivatives (Manjaro, EndeavourOS).

---

## Basic Commands

```bash
# Update package database
sudo pacman -Sy

# Upgrade all packages
sudo pacman -Syu

# Full system upgrade
sudo pacman -Syyu

# Install package
sudo pacman -S package-name

# Install multiple packages
sudo pacman -S package1 package2 package3

# Install without confirmation
sudo pacman -S --noconfirm package-name

# Remove package
sudo pacman -R package-name

# Remove package and dependencies
sudo pacman -Rs package-name

# Remove package, dependencies, and config
sudo pacman -Rns package-name

# Remove orphaned packages
sudo pacman -Rns $(pacman -Qtdq)
```

---

## Search and Query

```bash
# Search for package
pacman -Ss package-name

# Search installed packages
pacman -Qs package-name

# Show package info
pacman -Si package-name

# Show installed package info
pacman -Qi package-name

# List files in package
pacman -Ql package-name

# Find which package owns file
pacman -Qo /path/to/file

# List explicitly installed packages
pacman -Qe

# List all installed packages
pacman -Q

# List orphaned packages
pacman -Qtd
```

---

## Install from Repository

```bash
# Update and install
sudo pacman -Syu package-name

# Install specific version (from cache)
sudo pacman -U /var/cache/pacman/pkg/package-name-version.pkg.tar.zst

# Install from URL
sudo pacman -U https://archive.org/download/package.pkg.tar.zst

# Reinstall package
sudo pacman -S --overwrite '*' package-name

# Download only (don't install)
sudo pacman -Sw package-name
```

---

## Install from Source (AUR)

### Using yay (AUR Helper)

```bash
# Install yay
sudo pacman -S --needed git base-devel
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si

# Search AUR
yay -Ss package-name

# Install from AUR
yay -S package-name

# Update AUR packages
yay -Syu
```

### Manual AUR Installation

```bash
# Install build tools
sudo pacman -S --needed base-devel git

# Clone AUR package
git clone https://aur.archlinux.org/package-name.git
cd package-name

# Review PKGBUILD (important!)
cat PKGBUILD

# Build and install
makepkg -si

# Build only
makepkg

# Install built package
sudo pacman -U package-name-version.pkg.tar.zst
```

---

## Build from Source (General)

```bash
# Install development tools
sudo pacman -S base-devel

# Clone source
git clone https://github.com/user/repo.git
cd repo

# Build
./configure
make

# Install
sudo make install

# Or create package
# Create PKGBUILD
cat > PKGBUILD <<'EOF'
pkgname=mypackage
pkgver=1.0
pkgrel=1
pkgdesc="My custom package"
arch=('x86_64')
url="https://example.com"
license=('MIT')
depends=()
source=("mypackage-$pkgver.tar.gz")
sha256sums=('SKIP')

build() {
    cd "$pkgname-$pkgver"
    ./configure --prefix=/usr
    make
}

package() {
    cd "$pkgname-$pkgver"
    make DESTDIR="$pkgdir/" install
}
EOF

# Build package
makepkg -si
```

---

## Cache Management

```bash
# Cache location
ls /var/cache/pacman/pkg/

# Clean cache (keep 3 versions)
sudo paccache -r

# Clean cache (keep 1 version)
sudo paccache -rk1

# Remove all cached packages
sudo pacman -Scc

# Remove uninstalled packages from cache
sudo pacman -Sc
```

---

## Database Management

```bash
# Refresh package database
sudo pacman -Sy

# Force refresh
sudo pacman -Syy

# Update database and upgrade
sudo pacman -Syu

# Check database
sudo pacman -Dk

# Optimize database
sudo pacman-optimize
```

---

## Package Groups

```bash
# List groups
pacman -Sg

# List packages in group
pacman -Sg group-name

# Install group
sudo pacman -S group-name

# Install specific packages from group
sudo pacman -S group-name --ignore package1,package2
```

---

## Mirrors

```bash
# Edit mirror list
sudo nano /etc/pacman.d/mirrorlist

# Rank mirrors by speed
sudo pacman -S pacman-contrib
sudo cp /etc/pacman.d/mirrorlist /etc/pacman.d/mirrorlist.backup
rankmirrors -n 6 /etc/pacman.d/mirrorlist.backup > /etc/pacman.d/mirrorlist

# Or use reflector
sudo pacman -S reflector
sudo reflector --latest 20 --protocol https --sort rate --save /etc/pacman.d/mirrorlist
```

---

## Configuration

```bash
# Edit pacman config
sudo nano /etc/pacman.conf

# Enable multilib (32-bit on 64-bit)
[multilib]
Include = /etc/pacman.d/mirrorlist

# Enable color output
Color

# Enable parallel downloads
ParallelDownloads = 5

# Ignore packages
IgnorePkg = package1 package2

# Ignore group
IgnoreGroup = gnome
```

---

## Hooks

```bash
# Hook location
ls /etc/pacman.d/hooks/

# Example hook (clean cache)
sudo mkdir -p /etc/pacman.d/hooks
sudo nano /etc/pacman.d/hooks/clean-cache.hook

[Trigger]
Operation = Upgrade
Operation = Install
Operation = Remove
Type = Package
Target = *

[Action]
Description = Cleaning pacman cache...
When = PostTransaction
Exec = /usr/bin/paccache -rk2
```

---

## Troubleshooting

```bash
# Fix corrupted database
sudo rm /var/lib/pacman/db.lck
sudo pacman -Syyu

# Reinstall all packages
sudo pacman -Qnq | sudo pacman -S -

# Fix keyring issues
sudo pacman -Sy archlinux-keyring
sudo pacman-key --init
sudo pacman-key --populate archlinux

# Force overwrite files
sudo pacman -S --overwrite '*' package-name

# Downgrade package
sudo pacman -U /var/cache/pacman/pkg/package-old-version.pkg.tar.zst

# Check for errors
sudo pacman -Dk
```

---

## Logs

```bash
# View pacman log
cat /var/log/pacman.log

# Recently installed
grep "\[ALPM\] installed" /var/log/pacman.log | tail -20

# Recently upgraded
grep "\[ALPM\] upgraded" /var/log/pacman.log | tail -20

# Recently removed
grep "\[ALPM\] removed" /var/log/pacman.log | tail -20
```

---

## Useful Aliases

```bash
# Add to ~/.bashrc
alias update='sudo pacman -Syu'
alias install='sudo pacman -S'
alias remove='sudo pacman -Rns'
alias search='pacman -Ss'
alias clean='sudo pacman -Sc && sudo paccache -r'
```

---

## AUR Helpers Comparison

| Helper | Language | Features |
|--------|----------|----------|
| **yay** | Go | Fast, feature-rich, popular |
| **paru** | Rust | Modern, secure, yay alternative |
| **pikaur** | Python | User-friendly, detailed |
| **aurman** | Python | Dependency resolution |

### Install paru

```bash
sudo pacman -S --needed base-devel git
git clone https://aur.archlinux.org/paru.git
cd paru
makepkg -si
```

---

## Security

```bash
# Update keyring
sudo pacman -S archlinux-keyring

# Check package signatures
pacman -Qi package-name | grep Signatures

# Verify database
sudo pacman -Dk

# List packages without signatures
pacman -Qi | grep -E "^Name|^Signatures" | grep -B1 "None"
```

---

## Create and Publish AUR Package

### PKGBUILD Structure

```bash
# PKGBUILD
pkgname=mypackage
pkgver=1.0.0
pkgrel=1
pkgdesc="A useful package"
arch=('x86_64' 'aarch64')
url="https://github.com/username/mypackage"
license=('MIT')
depends=('glibc' 'python')
makedepends=('git' 'go')
optdepends=('postgresql: for database support')
provides=('mypackage')
conflicts=('mypackage-git')
source=("$pkgname-$pkgver.tar.gz::https://github.com/username/$pkgname/archive/v$pkgver.tar.gz")
sha256sums=('SKIP')  # Replace with actual checksum

prepare() {
    cd "$pkgname-$pkgver"
    # Apply patches, etc.
}

build() {
    cd "$pkgname-$pkgver"
    ./configure --prefix=/usr
    make
}

check() {
    cd "$pkgname-$pkgver"
    make test
}

package() {
    cd "$pkgname-$pkgver"
    make DESTDIR="$pkgdir/" install
    
    # Install license
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
    
    # Install documentation
    install -Dm644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"
}
```

### Build Package Locally

```bash
# Create PKGBUILD
nano PKGBUILD

# Generate checksums
updpkgsums

# Build package
makepkg

# Install locally
sudo pacman -U mypackage-1.0.0-1-x86_64.pkg.tar.zst

# Test package
makepkg -si
```

### Publish to AUR

```bash
# 1. Create AUR account at https://aur.archlinux.org/register

# 2. Add SSH key to AUR
# https://aur.archlinux.org/account/

# 3. Clone AUR repository
git clone ssh://aur@aur.archlinux.org/mypackage.git
cd mypackage

# 4. Add PKGBUILD and .SRCINFO
cp /path/to/PKGBUILD .

# 5. Generate .SRCINFO
makepkg --printsrcinfo > .SRCINFO

# 6. Commit and push
git add PKGBUILD .SRCINFO
git commit -m "Initial commit: mypackage 1.0.0"
git push origin master

# Package is now on AUR!
# Users can install with: yay -S mypackage
```

### Update AUR Package

```bash
# Update PKGBUILD
nano PKGBUILD
# Change pkgver and pkgrel

# Update checksums
updpkgsums

# Generate new .SRCINFO
makepkg --printsrcinfo > .SRCINFO

# Commit and push
git add PKGBUILD .SRCINFO
git commit -m "Update to version 1.1.0"
git push
```

### PKGBUILD for Git Package

```bash
# PKGBUILD
pkgname=mypackage-git
pkgver=r123.abc1234
pkgrel=1
pkgdesc="A useful package (git version)"
arch=('x86_64')
url="https://github.com/username/mypackage"
license=('MIT')
depends=('glibc')
makedepends=('git' 'go')
provides=('mypackage')
conflicts=('mypackage')
source=("git+https://github.com/username/mypackage.git")
sha256sums=('SKIP')

pkgver() {
    cd "$srcdir/mypackage"
    printf "r%s.%s" "$(git rev-list --count HEAD)" "$(git rev-parse --short HEAD)"
}

build() {
    cd "$srcdir/mypackage"
    make
}

package() {
    cd "$srcdir/mypackage"
    make DESTDIR="$pkgdir/" install
}
```

### Custom Repository

```bash
# Build packages
makepkg

# Create repository database
repo-add myrepo.db.tar.gz *.pkg.tar.zst

# Serve repository (e.g., with nginx)
# /srv/http/myrepo/
# ├── myrepo.db
# ├── myrepo.db.tar.gz
# ├── myrepo.files
# ├── myrepo.files.tar.gz
# └── mypackage-1.0.0-1-x86_64.pkg.tar.zst

# Users add to /etc/pacman.conf:
[myrepo]
Server = https://repo.example.com/myrepo
SigLevel = Optional TrustAll

# Then: sudo pacman -Sy mypackage
```

### Best Practices

```bash
# 1. Use namcap to check PKGBUILD
namcap PKGBUILD
namcap mypackage-1.0.0-1-x86_64.pkg.tar.zst

# 2. Test in clean chroot
sudo pacman -S devtools
mkarchroot /tmp/chroot base-devel
makechrootpkg -c -r /tmp/chroot

# 3. Follow Arch packaging standards
# https://wiki.archlinux.org/title/Arch_package_guidelines

# 4. Respond to comments on AUR

# 5. Keep package updated

# 6. Add .gitignore
cat > .gitignore <<EOF
*
!PKGBUILD
!.SRCINFO
!.gitignore
EOF
```

---