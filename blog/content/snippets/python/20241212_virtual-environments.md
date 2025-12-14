---
title: "Python Virtual Environments"
date: 2024-12-12T20:00:00Z
draft: false
description: "Managing Python virtual environments and dependencies"
type: "snippet"
tags: ["python", "venv", "pip", "dependencies", "environment", "python-knowhow"]
category: "python"
---



Virtual environments isolate Python project dependencies, preventing conflicts between projects. Essential for reproducible research and clean dependency management.

## Use Case

Use virtual environments when you need to:
- Isolate project dependencies
- Work on multiple projects with different requirements
- Ensure reproducible environments
- Avoid system-wide package conflicts

## Code

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install packages
pip install numpy pandas matplotlib

# Save dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Deactivate
deactivate
```

## Explanation

- `python -m venv venv` - Creates virtual environment in `venv/` directory
- Activation modifies PATH to use venv's Python and pip
- `pip freeze` captures exact versions of installed packages
- `requirements.txt` enables reproducible installations

## Examples

### Example 1: Research Project Setup

```bash
# Create project directory
mkdir my-research
cd my-research

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install research dependencies
pip install numpy scipy matplotlib jupyter pandas scikit-learn

# Save dependencies
pip freeze > requirements.txt

# Add to git
echo "venv/" >> .gitignore
git add requirements.txt
git commit -m "Add Python dependencies"
```

### Example 2: Reproduce Environment

```bash
# Clone repository
git clone <repo-url>
cd <repo>

# Create and activate venv
python -m venv venv
source venv/bin/activate

# Install exact dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

### Example 3: Multiple Python Versions

```bash
# Use specific Python version
python3.11 -m venv venv311
python3.9 -m venv venv39

# Activate specific version
source venv311/bin/activate
python --version  # Should show 3.11.x
```

### Example 4: Upgrade Packages

```bash
# Activate environment
source venv/bin/activate

# Upgrade single package
pip install --upgrade numpy

# Upgrade all packages (careful!)
pip list --outdated
pip install --upgrade pip setuptools wheel

# Update requirements.txt
pip freeze > requirements.txt
```

## Notes

- Always activate venv before installing packages
- Include `requirements.txt` in version control
- Add `venv/` to `.gitignore`
- Use `pip list` to see installed packages
- Consider `pip-tools` or `poetry` for advanced dependency management

## Gotchas/Warnings

- ⚠️ **Activation**: Must activate venv in each new terminal session
- ⚠️ **System packages**: Don't use `sudo pip` - use venv instead
- ⚠️ **Path issues**: Deactivate old venv before activating new one
- ⚠️ **Requirements**: `pip freeze` includes all dependencies - use `pipreqs` for minimal requirements