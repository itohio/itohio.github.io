---
title: "pip Package Manager (Python)"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "pip", "python", "package-manager", "pypi"]
---


pip - Package installer for Python.

---

## Basic Commands

```bash
# Install package
pip install package-name

# Install specific version
pip install package-name==1.0.0

# Install minimum version
pip install 'package-name>=1.0.0'

# Install version range
pip install 'package-name>=1.0.0,<2.0.0'

# Install from requirements file
pip install -r requirements.txt

# Uninstall package
pip uninstall package-name

# Uninstall without confirmation
pip uninstall -y package-name

# Upgrade package
pip install --upgrade package-name
pip install -U package-name

# Upgrade pip itself
python -m pip install --upgrade pip
```

---

## List and Search

```bash
# List installed packages
pip list

# List outdated packages
pip list --outdated

# Show package info
pip show package-name

# Show package files
pip show --files package-name

# Search packages (deprecated, use https://pypi.org)
# pip search package-name  # No longer works
```

---

## Requirements Files

```bash
# Generate requirements.txt
pip freeze > requirements.txt

# Generate with versions
pip freeze --all > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Upgrade all packages in requirements
pip install -r requirements.txt --upgrade
```

### requirements.txt Format

```
# requirements.txt
# Exact version
requests==2.31.0

# Minimum version
flask>=2.0.0

# Version range
django>=4.0.0,<5.0.0

# From Git
git+https://github.com/user/repo.git@v1.0.0

# From Git branch
git+https://github.com/user/repo.git@main

# Editable install (development)
-e .
-e ./path/to/package

# Include another requirements file
-r base.txt

# Comments
# This is a comment
```

---

## Virtual Environments

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate

# Install packages in venv
pip install package-name

# Generate requirements from venv
pip freeze > requirements.txt
```

---

## Install from Source

```bash
# Install from local directory
pip install /path/to/package

# Install in editable mode (development)
pip install -e /path/to/package
pip install -e .

# Install from Git
pip install git+https://github.com/user/repo.git

# Install from specific branch
pip install git+https://github.com/user/repo.git@branch-name

# Install from specific tag
pip install git+https://github.com/user/repo.git@v1.0.0

# Install from specific commit
pip install git+https://github.com/user/repo.git@abc123

# Install from tarball
pip install https://example.com/package.tar.gz

# Install from wheel
pip install package-name.whl
```

---

## Configuration

```bash
# Show pip config
pip config list

# Set config value
pip config set global.index-url https://pypi.org/simple

# Get config value
pip config get global.index-url

# Edit config file
pip config edit

# Config file locations:
# Global: /etc/pip.conf (Linux), %APPDATA%\pip\pip.ini (Windows)
# User: ~/.config/pip/pip.conf (Linux), %APPDATA%\pip\pip.ini (Windows)
# Virtual env: $VIRTUAL_ENV/pip.conf
```

### pip.conf Example

```ini
[global]
index-url = https://pypi.org/simple
trusted-host = pypi.org
timeout = 60

[install]
no-cache-dir = true
```

---

## Alternative Package Indexes

```bash
# Use alternative index
pip install --index-url https://test.pypi.org/simple/ package-name

# Use extra index
pip install --extra-index-url https://pypi.example.com/simple package-name

# Trust host
pip install --trusted-host pypi.example.com package-name

# Use local index
pip install --index-url file:///path/to/packages package-name
```

---

## Cache Management

```bash
# Show cache location
pip cache dir

# List cached packages
pip cache list

# Show cache info
pip cache info

# Remove cache
pip cache purge

# Remove specific package cache
pip cache remove package-name

# Install without cache
pip install --no-cache-dir package-name
```

---

## Dependency Resolution

```bash
# Show dependency tree
pip install pipdeptree
pipdeptree

# Show dependencies for package
pipdeptree -p package-name

# Show reverse dependencies
pipdeptree -r -p package-name

# Check for conflicts
pip check

# Install with no dependencies
pip install --no-deps package-name
```

---

## Wheel and Build

```bash
# Install wheel
pip install wheel

# Build wheel
pip wheel .

# Build wheel for package
pip wheel package-name

# Install from wheel
pip install package-name.whl

# Download packages (no install)
pip download package-name

# Download with dependencies
pip download -r requirements.txt -d ./packages
```

---

## Create Python Package

### Project Structure

```
mypackage/
├── mypackage/
│   ├── __init__.py
│   ├── module1.py
│   └── module2.py
├── tests/
│   ├── __init__.py
│   └── test_module1.py
├── setup.py
├── setup.cfg
├── pyproject.toml
├── README.md
├── LICENSE
└── MANIFEST.in
```

### setup.py

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mypackage",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mypackage",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/mypackage/issues",
        "Documentation": "https://mypackage.readthedocs.io",
        "Source Code": "https://github.com/yourusername/mypackage",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mypackage=mypackage.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
```

### pyproject.toml (Modern)

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "0.1.0"
description = "A short description"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["example", "package"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]
dependencies = [
    "requests>=2.25.0",
    "click>=8.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
]
docs = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/mypackage"
Documentation = "https://mypackage.readthedocs.io"
Repository = "https://github.com/yourusername/mypackage"
"Bug Tracker" = "https://github.com/yourusername/mypackage/issues"

[project.scripts]
mypackage = "mypackage.cli:main"

[tool.setuptools]
packages = ["mypackage"]

[tool.setuptools.package-data]
mypackage = ["data/*.json"]
```

---

## Build Package

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# This creates:
# dist/mypackage-0.1.0.tar.gz (source distribution)
# dist/mypackage-0.1.0-py3-none-any.whl (wheel)

# Check package
twine check dist/*
```

---

## Publish to PyPI

```bash
# Install twine
pip install twine

# Create PyPI account at https://pypi.org/account/register/

# Create API token at https://pypi.org/manage/account/token/

# Configure credentials
# Create ~/.pypirc:
cat > ~/.pypirc <<EOF
[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE
EOF

# Upload to Test PyPI (recommended first)
twine upload --repository testpypi dist/*

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ mypackage

# Upload to PyPI
twine upload dist/*

# Install from PyPI
pip install mypackage
```

---

## Private Package Repository

### Using PyPI Server

```bash
# Install pypiserver
pip install pypiserver

# Create packages directory
mkdir ~/packages

# Run server
pypiserver run -p 8080 ~/packages

# Upload package
twine upload --repository-url http://localhost:8080 dist/*

# Install from private server
pip install --index-url http://localhost:8080/simple/ mypackage
```

### Using Artifactory/Nexus

```bash
# Configure pip to use private repository
pip config set global.index-url https://artifactory.example.com/api/pypi/pypi/simple

# Or use in command
pip install --index-url https://artifactory.example.com/api/pypi/pypi/simple mypackage

# Upload with twine
twine upload --repository-url https://artifactory.example.com/api/pypi/pypi dist/*
```

### Using Git Repository

```bash
# Install directly from Git
pip install git+https://github.com/user/private-repo.git

# With authentication
pip install git+https://username:token@github.com/user/private-repo.git

# In requirements.txt
git+https://github.com/user/private-repo.git@v1.0.0
```

---

## Troubleshooting

```bash
# Verbose output
pip install -v package-name

# Very verbose
pip install -vv package-name

# Debug output
pip install -vvv package-name

# Check for issues
pip check

# Fix broken installation
pip install --force-reinstall package-name

# Clear cache and reinstall
pip cache purge
pip install --no-cache-dir package-name

# SSL issues
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package-name
```

---

## Best Practices

1. **Always use virtual environments**
2. **Pin versions in production** (`==` not `>=`)
3. **Use requirements.txt** for reproducibility
4. **Separate dev/prod dependencies**
5. **Keep pip updated**
6. **Use pip-tools** for dependency management
7. **Check for vulnerabilities** (use `pip-audit`)

---