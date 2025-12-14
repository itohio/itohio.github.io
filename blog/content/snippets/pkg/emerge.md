---
title: "emerge Package Manager (Gentoo)"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "emerge", "portage", "gentoo", "linux", "package-manager"]
---


emerge (Portage) - Source-based package manager for Gentoo Linux.

---

## Basic Commands

```bash
# Sync package tree
emerge --sync
# Or
emaint sync -a

# Update system
emerge --update --deep --newuse @world

# Install package
emerge package-name

# Install with dependencies
emerge --ask package-name

# Remove package
emerge --deselect package-name
emerge --depclean

# Search package
emerge --search package-name
emerge -s package-name

# Search by description
emerge --searchdesc keyword
```

---

## Install from Repository

```bash
# Install package
emerge app-editors/vim

# Install with confirmation
emerge --ask www-servers/nginx

# Install specific version
emerge =app-editors/vim-9.0.0

# Install latest version
emerge app-editors/vim

# Pretend install (dry run)
emerge --pretend package-name

# Verbose output
emerge --verbose package-name

# Ask before proceeding
emerge --ask package-name
```

---

## Build from Source

### All packages in Gentoo are built from source!

```bash
# Install package (builds from source)
emerge package-name

# Show build process
emerge --verbose package-name

# Build with specific USE flags
USE="flag1 -flag2" emerge package-name

# Build with custom CFLAGS
CFLAGS="-O3 -march=native" emerge package-name

# Parallel compilation
MAKEOPTS="-j8" emerge package-name

# Or set in /etc/portage/make.conf:
MAKEOPTS="-j8"
```

---

## USE Flags

```bash
# Show USE flags for package
emerge --pretend --verbose package-name

# Show all USE flags
equery uses package-name

# Set global USE flags
# Edit /etc/portage/make.conf
USE="X gtk gnome -kde -qt5"

# Set per-package USE flags
# /etc/portage/package.use/custom
app-editors/vim python ruby
www-servers/nginx ssl http2

# Show enabled USE flags
emerge --info | grep ^USE

# List all USE flags
less /var/db/repos/gentoo/profiles/use.desc
```

---

## System Updates

```bash
# Update world (all packages)
emerge --update --deep --newuse @world

# Update with ask
emerge --ask --update --deep --newuse @world

# Shorter form
emerge -avuDN @world

# Update only changed USE flags
emerge --newuse @world

# Update kernel
emerge --ask sys-kernel/gentoo-sources
```

---

## Package Management

```bash
# List installed packages
qlist -I

# Show package info
emerge --info package-name

# Show dependencies
emerge --pretend --tree package-name

# Show reverse dependencies
equery depends package-name

# List files in package
equery files package-name

# Find which package owns file
equery belongs /path/to/file

# Check package integrity
equery check package-name
```

---

## Cleaning

```bash
# Remove unused dependencies
emerge --depclean

# Pretend depclean
emerge --pretend --depclean

# Clean build files
eclean-dist --deep

# Clean old packages
eclean-pkg --deep

# Remove obsolete packages
emerge --prune
```

---

## Configuration

### /etc/portage/make.conf

```bash
# Compiler flags
CFLAGS="-O2 -pipe -march=native"
CXXFLAGS="${CFLAGS}"

# Parallel compilation
MAKEOPTS="-j8"

# USE flags
USE="X gtk gnome -kde -qt5 systemd"

# Mirrors
GENTOO_MIRRORS="https://mirror.example.com/gentoo"

# Features
FEATURES="parallel-fetch ccache"

# Accept licenses
ACCEPT_LICENSE="*"

# Language
L10N="en"
```

---

## Overlays (Third-party Repositories)

```bash
# Install layman (overlay manager)
emerge app-portage/layman

# List available overlays
layman -L

# Add overlay
layman -a overlay-name

# Sync overlays
layman -S

# Remove overlay
layman -d overlay-name

# Or use eselect repository (modern)
emerge app-eselect/eselect-repository
eselect repository list
eselect repository enable overlay-name
emerge --sync
```

---

## Masking/Unmasking

```bash
# Unmask package
# /etc/portage/package.unmask
=app-editors/vim-9.0.0

# Mask package
# /etc/portage/package.mask
>=app-editors/vim-9.0.0

# Accept keywords (testing)
# /etc/portage/package.accept_keywords
app-editors/vim ~amd64
```

---

## Binary Packages

```bash
# Build binary package
emerge --buildpkg package-name

# Install from binary
emerge --usepkg package-name

# Install binary only (no compile)
emerge --getbinpkg package-name

# Configure binary host
# /etc/portage/make.conf
PORTAGE_BINHOST="https://binhost.example.com/packages"

# Build all as binary
FEATURES="buildpkg"
```

---

## Troubleshooting

```bash
# Fix broken dependencies
emerge --ask --oneshot --verbose sys-apps/portage
revdep-rebuild

# Rebuild @world
emerge -e @world

# Resume failed build
emerge --resume

# Skip failed package
emerge --resume --skipfirst

# Force rebuild
emerge --oneshot package-name

# Check for problems
emerge --check-news
eselect news read
```

---

## Kernel Management

```bash
# Install kernel sources
emerge sys-kernel/gentoo-sources

# List available kernels
eselect kernel list

# Set active kernel
eselect kernel set 1

# Configure kernel
cd /usr/src/linux
make menuconfig

# Compile kernel
make -j8
make modules_install
make install

# Or use genkernel
emerge sys-kernel/genkernel
genkernel all
```

---

## Useful Tools

```bash
# Install gentoolkit
emerge app-portage/gentoolkit

# equery - query packages
equery list '*'
equery depends package-name
equery files package-name

# eix - fast package search
emerge app-portage/eix
eix-update
eix package-name

# genlop - analyze emerge logs
emerge app-portage/genlop
genlop -t package-name  # Estimate time
genlop -c               # Current emerge
```

---

## Logs

```bash
# Emerge log
cat /var/log/emerge.log

# Fetch log
cat /var/log/emerge-fetch.log

# Build log location
ls /var/tmp/portage/

# Show current emerge
genlop -c

# Show emerge history
genlop -l

# Estimate time
genlop -t package-name
```

---

## Common Workflows

### Full System Update

```bash
# Sync
emerge --sync

# Update world
emerge -avuDN @world

# Clean dependencies
emerge --depclean

# Rebuild if needed
revdep-rebuild

# Clean distfiles
eclean-dist --deep
```

### Install New Package

```bash
# Search
eix package-name

# Show info
emerge --pretend --verbose package-name

# Install
emerge --ask package-name

# Check
equery files package-name
```

---

## Useful Aliases

```bash
# Add to ~/.bashrc
alias update='emerge --sync && emerge -avuDN @world'
alias install='emerge --ask'
alias remove='emerge --deselect'
alias search='eix'
alias clean='emerge --depclean && eclean-dist --deep'
```

---

## Create and Publish Gentoo Ebuild

### Ebuild Structure

```
mypackage/
├── mypackage-1.0.0.ebuild
├── mypackage-9999.ebuild  # Live ebuild (git)
├── Manifest
├── metadata.xml
└── files/
    ├── mypackage-1.0.0-fix-build.patch
    └── mypackage.initd
```

### Basic Ebuild

```bash
# mypackage-1.0.0.ebuild
# Copyright 1999-2024 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

DESCRIPTION="A useful package"
HOMEPAGE="https://github.com/username/mypackage"
SRC_URI="https://github.com/username/${PN}/archive/v${PV}.tar.gz -> ${P}.tar.gz"

LICENSE="MIT"
SLOT="0"
KEYWORDS="~amd64 ~arm64"
IUSE="doc examples +systemd"

DEPEND="
    dev-libs/glib:2
    sys-libs/zlib
"
RDEPEND="${DEPEND}"
BDEPEND="
    virtual/pkgconfig
    doc? ( dev-python/sphinx )
"

src_configure() {
    local myconf=(
        $(use_enable doc documentation)
        $(use_enable systemd)
    )
    econf "${myconf[@]}"
}

src_compile() {
    default
    use doc && emake -C docs html
}

src_install() {
    default
    
    if use doc; then
        docinto html
        dodoc -r docs/_build/html/*
    fi
    
    if use examples; then
        docinto examples
        dodoc -r examples/*
    fi
    
    newinitd "${FILESDIR}/${PN}.initd" ${PN}
}
```

### Live Ebuild (Git)

```bash
# mypackage-9999.ebuild
# Copyright 1999-2024 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

inherit git-r3

DESCRIPTION="A useful package (live version)"
HOMEPAGE="https://github.com/username/mypackage"
EGIT_REPO_URI="https://github.com/username/mypackage.git"

LICENSE="MIT"
SLOT="0"
KEYWORDS=""  # Live ebuilds don't have keywords

DEPEND="dev-libs/glib:2"
RDEPEND="${DEPEND}"
BDEPEND="virtual/pkgconfig"

src_configure() {
    econf
}

src_install() {
    default
}
```

### metadata.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE pkgmetadata SYSTEM "https://www.gentoo.org/dtd/metadata.dtd">
<pkgmetadata>
    <maintainer type="person">
        <email>email@example.com</email>
        <name>Your Name</name>
    </maintainer>
    <longdescription>
        Long description of the package.
        Can span multiple lines.
    </longdescription>
    <use>
        <flag name="doc">Build and install documentation</flag>
        <flag name="examples">Install example files</flag>
        <flag name="systemd">Enable systemd support</flag>
    </use>
    <upstream>
        <remote-id type="github">username/mypackage</remote-id>
    </upstream>
</pkgmetadata>
```

### Generate Manifest

```bash
# Generate checksums
ebuild mypackage-1.0.0.ebuild manifest

# Creates Manifest:
DIST mypackage-1.0.0.tar.gz 123456 BLAKE2B abc... SHA512 def...
```

### Test Ebuild

```bash
# Test fetch
ebuild mypackage-1.0.0.ebuild fetch

# Test unpack
ebuild mypackage-1.0.0.ebuild unpack

# Test compile
ebuild mypackage-1.0.0.ebuild compile

# Test install
ebuild mypackage-1.0.0.ebuild install

# Install to system
ebuild mypackage-1.0.0.ebuild merge

# Check with repoman
repoman full

# Check with pkgcheck
pkgcheck scan
```

### Create Local Overlay

```bash
# Create overlay directory
mkdir -p /usr/local/portage/app-misc/mypackage

# Copy ebuild files
cp mypackage-1.0.0.ebuild /usr/local/portage/app-misc/mypackage/
cp metadata.xml /usr/local/portage/app-misc/mypackage/

# Generate Manifest
cd /usr/local/portage/app-misc/mypackage
ebuild mypackage-1.0.0.ebuild manifest

# Install
emerge app-misc/mypackage
```

### Publish to GURU (Gentoo User Repository)

```bash
# 1. Request access to GURU
# https://wiki.gentoo.org/wiki/Project:GURU

# 2. Clone GURU
git clone https://anongit.gentoo.org/git/proj/guru.git

# 3. Create package directory
cd guru
mkdir -p app-misc/mypackage

# 4. Add ebuild files
cp mypackage-1.0.0.ebuild app-misc/mypackage/
cp metadata.xml app-misc/mypackage/

# 5. Generate Manifest
cd app-misc/mypackage
ebuild mypackage-1.0.0.ebuild manifest

# 6. Check with pkgcheck
pkgcheck scan

# 7. Commit and push
git add app-misc/mypackage
git commit -s -m "app-misc/mypackage: new package, add 1.0.0"
git push

# Users can install with:
# eselect repository enable guru
# emerge --sync guru
# emerge app-misc/mypackage
```

### Create Custom Overlay

```bash
# Create overlay structure
mkdir -p /var/db/repos/myoverlay/{metadata,profiles}

# Create layout.conf
cat > /var/db/repos/myoverlay/metadata/layout.conf <<EOF
masters = gentoo
auto-sync = false
EOF

# Create repo_name
echo "myoverlay" > /var/db/repos/myoverlay/profiles/repo_name

# Add to repos.conf
cat > /etc/portage/repos.conf/myoverlay.conf <<EOF
[myoverlay]
location = /var/db/repos/myoverlay
auto-sync = no
EOF

# Add packages
mkdir -p /var/db/repos/myoverlay/app-misc/mypackage
# Add ebuild files...

# Users can sync with:
# emerge --sync
# emerge app-misc/mypackage
```

### Best Practices

```bash
# 1. Follow Gentoo development guide
# https://devmanual.gentoo.org/

# 2. Use proper EAPI
EAPI=8  # Latest stable

# 3. Use eclasses
inherit autotools git-r3 systemd

# 4. Handle USE flags properly
IUSE="doc examples +default-flag"

# 5. Use proper dependencies
DEPEND="build-time dependencies"
RDEPEND="runtime dependencies"
BDEPEND="build tools"

# 6. Install files correctly
dobin myapp
doman myapp.1
dodoc README.md
doins config.conf

# 7. Check with tools
repoman full
pkgcheck scan

# 8. Sign commits
git commit -s -m "message"
```

### Advanced Ebuild Features

```bash
# Python support
inherit python-single-r1

PYTHON_COMPAT=( python3_{10..12} )

DEPEND="${PYTHON_DEPS}"
RDEPEND="${DEPEND}"

# Multiple versions
SLOT="0/1.0"  # Sub-slot for ABI

# Patches
PATCHES=(
    "${FILESDIR}/${P}-fix-build.patch"
)

# Custom phases
src_prepare() {
    default
    eautoreconf
}

pkg_postinst() {
    elog "Important information for users"
    elog "Configuration file: /etc/mypackage/config.conf"
}
```

---