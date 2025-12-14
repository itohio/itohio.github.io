---
title: "yum/dnf Package Manager (RHEL/Fedora)"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "yum", "dnf", "rhel", "fedora", "centos", "linux", "package-manager"]
---


yum (Yellowdog Updater Modified) and dnf (Dandified Yum) for RHEL, Fedora, CentOS.

---

## dnf vs yum

```bash
# dnf is the modern replacement for yum
# Commands are mostly compatible

# Fedora 22+: dnf
# RHEL 8+: dnf
# CentOS 8+: dnf
# Older systems: yum

# This guide uses dnf, but most commands work with yum
```

---

## Basic Commands

```bash
# Update package cache
sudo dnf check-update

# Upgrade all packages
sudo dnf upgrade

# Install package
sudo dnf install package-name

# Install specific version
sudo dnf install package-name-version

# Install multiple packages
sudo dnf install package1 package2 package3

# Install without confirmation
sudo dnf install -y package-name

# Remove package
sudo dnf remove package-name

# Autoremove unused dependencies
sudo dnf autoremove
```

---

## Search and Query

```bash
# Search for package
dnf search package-name

# Show package info
dnf info package-name

# List installed packages
dnf list installed

# List available packages
dnf list available

# List updates
dnf list updates

# Show package dependencies
dnf deplist package-name

# Find which package provides file
dnf provides /path/to/file
dnf whatprovides */filename
```

---

## Repositories

```bash
# List enabled repositories
dnf repolist

# List all repositories
dnf repolist --all

# Enable repository
sudo dnf config-manager --set-enabled repo-name

# Disable repository
sudo dnf config-manager --set-disabled repo-name

# Add repository
sudo dnf config-manager --add-repo https://example.com/repo.repo

# Install from specific repo
sudo dnf --enablerepo=repo-name install package-name
```

---

## EPEL Repository (RHEL/CentOS)

```bash
# Install EPEL (Extra Packages for Enterprise Linux)
# RHEL 8/CentOS 8
sudo dnf install epel-release

# RHEL 9/CentOS 9
sudo dnf install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

# Verify
dnf repolist | grep epel
```

---

## RPM Fusion (Fedora)

```bash
# Free repository
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm

# Nonfree repository
sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
```

---

## Install from RPM File

```bash
# Install .rpm package
sudo dnf install ./package.rpm

# Or using rpm directly
sudo rpm -ivh package.rpm

# Upgrade package
sudo rpm -Uvh package.rpm

# Remove package
sudo rpm -e package-name

# Query installed package
rpm -qa | grep package-name

# List files in package
rpm -ql package-name

# Show package info
rpm -qi package-name
```

---

## Build from Source

```bash
# Install development tools
sudo dnf groupinstall "Development Tools"
sudo dnf install rpm-build rpmdevtools

# Setup RPM build environment
rpmdev-setuptree

# Directory structure:
# ~/rpmbuild/
# ├── BUILD/
# ├── RPMS/
# ├── SOURCES/
# ├── SPECS/
# └── SRPMS/

# Download source
cd ~/rpmbuild/SOURCES
wget https://example.com/source.tar.gz

# Create spec file
cd ~/rpmbuild/SPECS
rpmdev-newspec package-name

# Edit spec file
nano package-name.spec

# Build RPM
rpmbuild -ba package-name.spec

# Install built RPM
sudo dnf install ~/rpmbuild/RPMS/x86_64/package-name-version.rpm
```

---

## Example SPEC File

```spec
Name:           mypackage
Version:        1.0
Release:        1%{?dist}
Summary:        My custom package

License:        MIT
URL:            https://example.com
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  gcc make
Requires:       glibc

%description
Long description of my package

%prep
%setup -q

%build
./configure --prefix=/usr
make %{?_smp_mflags}

%install
make install DESTDIR=%{buildroot}

%files
/usr/bin/myprogram
/usr/share/doc/%{name}/README

%changelog
* Thu Dec 12 2024 Your Name <email@example.com> - 1.0-1
- Initial package
```

---

## Groups

```bash
# List groups
dnf group list

# Show group info
dnf group info "Group Name"

# Install group
sudo dnf group install "Development Tools"

# Remove group
sudo dnf group remove "Development Tools"

# Common groups:
# - "Development Tools"
# - "System Tools"
# - "Security Tools"
```

---

## History

```bash
# Show transaction history
dnf history

# Show specific transaction
dnf history info <id>

# Undo transaction
sudo dnf history undo <id>

# Redo transaction
sudo dnf history redo <id>

# Rollback to transaction
sudo dnf history rollback <id>
```

---

## Cache Management

```bash
# Clean cache
sudo dnf clean all

# Clean packages only
sudo dnf clean packages

# Clean metadata
sudo dnf clean metadata

# Make cache
sudo dnf makecache

# Download packages without installing
dnf download package-name

# Download with dependencies
dnf download --resolve package-name
```

---

## Configuration

```bash
# Main config file
sudo nano /etc/dnf/dnf.conf

# Example settings:
[main]
gpgcheck=1
installonly_limit=3
clean_requirements_on_remove=True
best=True
skip_if_unavailable=True
max_parallel_downloads=10
```

---

## Security Updates

```bash
# List security updates
dnf updateinfo list security

# Install security updates only
sudo dnf upgrade --security

# Show security advisories
dnf updateinfo list --security

# Show specific advisory
dnf updateinfo info <advisory-id>
```

---

## Modules (dnf only)

```bash
# List modules
dnf module list

# Show module info
dnf module info module-name

# Install module stream
sudo dnf module install module-name:stream

# Enable module
sudo dnf module enable module-name:stream

# Disable module
sudo dnf module disable module-name

# Reset module
sudo dnf module reset module-name

# Example: Install Python 3.11
sudo dnf module install python:3.11
```

---

## Troubleshooting

```bash
# Fix broken dependencies
sudo dnf check
sudo dnf upgrade --best --allowerasing

# Rebuild RPM database
sudo rpm --rebuilddb

# Clean and update
sudo dnf clean all
sudo dnf makecache
sudo dnf upgrade

# Force reinstall
sudo dnf reinstall package-name

# Check for duplicate packages
dnf repoquery --duplicates
```

---

## Useful Aliases

```bash
# Add to ~/.bashrc
alias update='sudo dnf upgrade'
alias install='sudo dnf install'
alias remove='sudo dnf remove'
alias search='dnf search'
alias clean='sudo dnf autoremove && sudo dnf clean all'
```

---

## yum Commands (Legacy)

```bash
# Update
sudo yum update

# Install
sudo yum install package-name

# Remove
sudo yum remove package-name

# Search
yum search package-name

# Info
yum info package-name

# List
yum list installed

# Clean
sudo yum clean all

# History
yum history
```

---

## Create and Publish RPM Package

### Setup Build Environment

```bash
# Install tools
sudo dnf install rpm-build rpmdevtools rpmlint

# Setup RPM build tree
rpmdev-setuptree

# Creates:
# ~/rpmbuild/
# ├── BUILD/
# ├── RPMS/
# ├── SOURCES/
# ├── SPECS/
# └── SRPMS/
```

### Create SPEC File

```bash
# Generate template
cd ~/rpmbuild/SPECS
rpmdev-newspec mypackage

# Edit spec file
nano mypackage.spec
```

### mypackage.spec

```spec
Name:           mypackage
Version:        1.0.0
Release:        1%{?dist}
Summary:        A useful package

License:        MIT
URL:            https://github.com/username/mypackage
Source0:        https://github.com/username/%{name}/archive/v%{version}.tar.gz

BuildRequires:  gcc make
Requires:       glibc python3

%description
Long description of my package.
It can span multiple lines.

%prep
%setup -q

%build
./configure --prefix=%{_prefix}
make %{?_smp_mflags}

%install
make install DESTDIR=%{buildroot}

# Install systemd service
install -Dm644 mypackage.service %{buildroot}%{_unitdir}/mypackage.service

%post
%systemd_post mypackage.service

%preun
%systemd_preun mypackage.service

%postun
%systemd_postun_with_restart mypackage.service

%files
%license LICENSE
%doc README.md
%{_bindir}/mypackage
%{_unitdir}/mypackage.service
%config(noreplace) %{_sysconfdir}/mypackage/config.conf

%changelog
* Thu Dec 12 2024 Your Name <email@example.com> - 1.0.0-1
- Initial package
```

### Build RPM

```bash
# Download source
cd ~/rpmbuild/SOURCES
wget https://github.com/username/mypackage/archive/v1.0.0.tar.gz

# Build RPM
cd ~/rpmbuild/SPECS
rpmbuild -ba mypackage.spec

# This creates:
# ~/rpmbuild/RPMS/x86_64/mypackage-1.0.0-1.fc38.x86_64.rpm
# ~/rpmbuild/SRPMS/mypackage-1.0.0-1.fc38.src.rpm

# Check RPM
rpmlint mypackage.spec
rpmlint ~/rpmbuild/RPMS/x86_64/mypackage-1.0.0-1.fc38.x86_64.rpm

# Install locally
sudo dnf install ~/rpmbuild/RPMS/x86_64/mypackage-1.0.0-1.fc38.x86_64.rpm
```

### Publish to COPR (Fedora)

```bash
# Install copr-cli
sudo dnf install copr-cli

# Create COPR project at https://copr.fedorainfracloud.org/

# Configure API token
# Get token from https://copr.fedorainfracloud.org/api/
nano ~/.config/copr

# Build from SRPM
copr-cli build myproject ~/rpmbuild/SRPMS/mypackage-1.0.0-1.fc38.src.rpm

# Or build from Git
copr-cli buildscm myproject --clone-url https://github.com/username/mypackage.git --spec mypackage.spec

# Users can install with:
# sudo dnf copr enable username/myproject
# sudo dnf install mypackage
```

### Custom RPM Repository

```bash
# Create repository directory
mkdir -p /var/www/rpm/{RPMS,SRPMS}

# Copy RPMs
cp ~/rpmbuild/RPMS/x86_64/*.rpm /var/www/rpm/RPMS/
cp ~/rpmbuild/SRPMS/*.rpm /var/www/rpm/SRPMS/

# Create repository metadata
createrepo /var/www/rpm

# Sign packages (optional)
rpm --addsign /var/www/rpm/RPMS/*.rpm

# Export GPG public key
gpg --export --armor YOUR_KEY_ID > /var/www/rpm/RPM-GPG-KEY-myrepo

# Serve with nginx
# Configure nginx to serve /var/www/rpm

# Users add repository:
# sudo nano /etc/yum.repos.d/myrepo.repo
[myrepo]
name=My Custom Repository
baseurl=https://repo.example.com/rpm
enabled=1
gpgcheck=1
gpgkey=https://repo.example.com/rpm/RPM-GPG-KEY-myrepo

# Then: sudo dnf install mypackage
```

### Build for Multiple Distributions

```bash
# Use mock for clean builds
sudo dnf install mock

# Add user to mock group
sudo usermod -a -G mock $USER

# Build for Fedora 38
mock -r fedora-38-x86_64 ~/rpmbuild/SRPMS/mypackage-1.0.0-1.fc38.src.rpm

# Build for RHEL 9
mock -r rhel-9-x86_64 ~/rpmbuild/SRPMS/mypackage-1.0.0-1.el9.src.rpm

# Results in:
# /var/lib/mock/fedora-38-x86_64/result/
# /var/lib/mock/rhel-9-x86_64/result/
```

### Best Practices

```bash
# 1. Follow Fedora packaging guidelines
# https://docs.fedoraproject.org/en-US/packaging-guidelines/

# 2. Use macros
%{_bindir}      # /usr/bin
%{_sbindir}     # /usr/sbin
%{_libdir}      # /usr/lib64
%{_sysconfdir}  # /etc
%{_datadir}     # /usr/share

# 3. Handle config files properly
%config(noreplace) %{_sysconfdir}/mypackage/config.conf

# 4. Use systemd macros
%systemd_post mypackage.service
%systemd_preun mypackage.service
%systemd_postun_with_restart mypackage.service

# 5. Check with rpmlint
rpmlint mypackage.spec
rpmlint mypackage.rpm

# 6. Test installation
sudo dnf install mypackage.rpm
sudo dnf remove mypackage
```

---