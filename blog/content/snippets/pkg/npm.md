---
title: "npm - Node Package Manager"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "npm", "nodejs", "javascript"]
---


Essential npm commands for Node.js package management. Quick reference for daily development tasks.

---

## Installation & Setup

```bash
# Check npm version
npm --version
npm -v

# Update npm
npm install -g npm@latest

# Initialize new project
npm init
npm init -y  # Skip prompts, use defaults

# Initialize with specific fields
npm init --scope=@myorg
```

---

## Installing Packages

```bash
# Install package (adds to dependencies)
npm install <package>
npm i <package>

# Install specific version
npm install <package>@1.2.3
npm install <package>@latest
npm install <package>@next

# Install as dev dependency
npm install --save-dev <package>
npm install -D <package>

# Install globally
npm install --global <package>
npm install -g <package>

# Install from package.json
npm install
npm i

# Install from package-lock.json (exact versions)
npm ci  # Clean install - faster, for CI/CD
```

**Common Packages:**
```bash
# TypeScript
npm i -D typescript @types/node

# React
npm i react react-dom
npm i -D @types/react @types/react-dom

# Express
npm i express
npm i -D @types/express

# Testing
npm i -D jest @types/jest
npm i -D vitest
```

---

## Uninstalling Packages

```bash
# Uninstall package
npm uninstall <package>
npm un <package>
npm remove <package>
npm rm <package>

# Uninstall dev dependency
npm uninstall --save-dev <package>
npm un -D <package>

# Uninstall globally
npm uninstall --global <package>
npm un -g <package>
```

---

## Updating Packages

```bash
# Check for outdated packages
npm outdated

# Update package to latest version
npm update <package>
npm up <package>

# Update all packages
npm update

# Update to latest (ignoring semver)
npm install <package>@latest

# Interactive update (with npm-check-updates)
npx npm-check-updates
npx ncu -u  # Update package.json
npm install  # Install updated versions
```

---

## Listing Packages

```bash
# List installed packages
npm list
npm ls

# List top-level packages only
npm list --depth=0
npm ls --depth=0

# List globally installed packages
npm list -g --depth=0

# List specific package
npm list <package>

# List outdated packages
npm outdated
```

---

## Scripts

```json
// package.json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "lint": "eslint . --ext ts,tsx",
    "format": "prettier --write \"src/**/*.{ts,tsx}\"",
    "clean": "rm -rf dist node_modules"
  }
}
```

```bash
# Run script
npm run <script>
npm run dev
npm run build

# Special scripts (no "run" needed)
npm start   # Runs "start" script
npm test    # Runs "test" script
npm stop    # Runs "stop" script

# Pass arguments to script
npm run test -- --watch
npm run build -- --mode production

# List available scripts
npm run
```

---

## Package Information

```bash
# View package info
npm info <package>
npm view <package>

# View specific field
npm info <package> version
npm info <package> versions  # All versions
npm info <package> dependencies

# View package homepage
npm home <package>

# View package repository
npm repo <package>

# View package bugs
npm bugs <package>

# View package documentation
npm docs <package>
```

---

## Publishing

```bash
# Login to npm
npm login

# Check who you're logged in as
npm whoami

# Publish package
npm publish

# Publish with tag
npm publish --tag beta

# Publish scoped package (public)
npm publish --access public

# Unpublish package (within 72 hours)
npm unpublish <package>@<version>

# Deprecate package version
npm deprecate <package>@<version> "Use version X instead"
```

---

## Configuration

```bash
# View all config
npm config list
npm config ls

# Get specific config
npm config get registry
npm config get prefix

# Set config
npm config set registry https://registry.npmjs.org/
npm config set init-author-name "Your Name"
npm config set init-license "MIT"

# Delete config
npm config delete <key>

# Edit config file
npm config edit

# Set registry for scoped packages
npm config set @myorg:registry https://npm.pkg.github.com
```

**Common Config:**
```bash
# Set default save exact versions
npm config set save-exact true

# Set default save prefix (^ or ~)
npm config set save-prefix "~"

# Set npm cache location
npm config set cache /path/to/cache

# Disable package-lock.json
npm config set package-lock false  # Not recommended
```

---

## Cache Management

```bash
# View cache location
npm config get cache

# Verify cache
npm cache verify

# Clean cache
npm cache clean --force

# View cache size
du -sh $(npm config get cache)  # Unix
```

---

## Workspaces (Monorepo)

```json
// package.json (root)
{
  "name": "my-monorepo",
  "private": true,
  "workspaces": [
    "packages/*",
    "apps/*"
  ]
}
```

```bash
# Install all workspace dependencies
npm install

# Run script in specific workspace
npm run build --workspace=packages/app1
npm run build -w packages/app1

# Run script in all workspaces
npm run build --workspaces
npm run build -ws

# Add dependency to specific workspace
npm install lodash --workspace=packages/app1
npm i lodash -w packages/app1

# List workspaces
npm ls --workspaces
```

**Workspace Structure:**
```
my-monorepo/
├── package.json
├── packages/
│   ├── app1/
│   │   └── package.json
│   └── app2/
│       └── package.json
└── apps/
    └── web/
        └── package.json
```

---

## Security

```bash
# Audit packages for vulnerabilities
npm audit

# Audit and fix automatically
npm audit fix

# Audit and fix (including breaking changes)
npm audit fix --force

# View audit report in browser
npm audit --json | npm-audit-html

# Install specific security update
npm update <package> --depth 2
```

---

## Troubleshooting

```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Rebuild native modules
npm rebuild

# Check for issues
npm doctor

# Verbose logging
npm install --verbose
npm install --loglevel verbose

# Debug mode
npm install --dd
```

---

## Version Management

```bash
# Bump version
npm version patch  # 1.0.0 -> 1.0.1
npm version minor  # 1.0.0 -> 1.1.0
npm version major  # 1.0.0 -> 2.0.0

# Bump version with tag
npm version patch -m "Bump version to %s"

# Bump prerelease version
npm version prerelease  # 1.0.0 -> 1.0.1-0
npm version prepatch    # 1.0.0 -> 1.0.1-0
npm version preminor    # 1.0.0 -> 1.1.0-0
npm version premajor    # 1.0.0 -> 2.0.0-0
```

---

## Useful npm Packages

```bash
# npx - Run packages without installing
npx create-react-app my-app
npx create-vite my-app
npx tsc --init

# npm-check-updates - Update package.json
npx npm-check-updates
npx ncu -u

# http-server - Simple HTTP server
npx http-server

# nodemon - Auto-restart on file changes
npm i -D nodemon

# concurrently - Run multiple commands
npm i -D concurrently
# "dev": "concurrently \"npm run server\" \"npm run client\""

# cross-env - Set environment variables cross-platform
npm i -D cross-env
# "build": "cross-env NODE_ENV=production webpack"
```

---

## package.json Fields

```json
{
  "name": "my-package",
  "version": "1.0.0",
  "description": "My awesome package",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "test": "vitest"
  },
  "keywords": ["awesome", "package"],
  "author": "Your Name <you@example.com>",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/user/repo.git"
  },
  "bugs": {
    "url": "https://github.com/user/repo/issues"
  },
  "homepage": "https://github.com/user/repo#readme",
  "dependencies": {
    "react": "^18.2.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0"
  },
  "peerDependencies": {
    "react": ">=16.8.0"
  },
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=9.0.0"
  },
  "files": [
    "dist",
    "README.md"
  ]
}
```

---

## Semver (Semantic Versioning)

```
MAJOR.MINOR.PATCH
  1  .  2  .  3

MAJOR: Breaking changes
MINOR: New features (backward compatible)
PATCH: Bug fixes (backward compatible)
```

**Version Ranges:**
```json
{
  "dependencies": {
    "package1": "1.2.3",      // Exact version
    "package2": "^1.2.3",     // ^1.2.3 <= version < 2.0.0
    "package3": "~1.2.3",     // ~1.2.3 <= version < 1.3.0
    "package4": ">=1.2.3",    // Greater than or equal
    "package5": "1.2.x",      // 1.2.0 <= version < 1.3.0
    "package6": "*",          // Any version (avoid!)
    "package7": "latest"      // Latest version (avoid!)
  }
}
```

**Caret (^) vs Tilde (~):**
- `^1.2.3`: Compatible with 1.2.3 (allows minor and patch updates)
- `~1.2.3`: Approximately 1.2.3 (allows patch updates only)

---

## Common Gotchas

### 1. Node Version Mismatch

```bash
# Check Node version
node --version

# Use nvm to switch versions
nvm install 18
nvm use 18

# Or use .nvmrc file
echo "18" > .nvmrc
nvm use
```

### 2. Permission Errors (Global Install)

```bash
# ❌ Don't use sudo with npm
sudo npm install -g <package>

# ✅ Fix npm permissions
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
# Add to ~/.bashrc or ~/.zshrc:
export PATH=~/.npm-global/bin:$PATH

# Or use nvm (recommended)
```

### 3. package-lock.json Conflicts

```bash
# After merging, regenerate lock file
rm package-lock.json
npm install

# Or use npm ci in CI/CD
npm ci  # Fails if package.json and lock file don't match
```

### 4. Peer Dependency Warnings

```bash
# Install peer dependencies manually
npm install <peer-dependency>

# Or use --legacy-peer-deps flag
npm install --legacy-peer-deps

# Or use --force (not recommended)
npm install --force
```

---

## Create and Publish Package

### Package Structure

```
my-package/
├── src/
│   └── index.js
├── test/
│   └── index.test.js
├── package.json
├── README.md
├── LICENSE
├── .gitignore
└── .npmignore
```

### Initialize Package

```bash
# Create package.json
npm init

# Or with defaults
npm init -y

# Or use npm init with scope
npm init --scope=@myorg
```

### package.json

```json
{
  "name": "my-package",
  "version": "1.0.0",
  "description": "A useful package",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "test": "jest",
    "prepublishOnly": "npm run build && npm test"
  },
  "keywords": ["utility", "helper"],
  "author": "Your Name <email@example.com>",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/username/my-package.git"
  },
  "bugs": {
    "url": "https://github.com/username/my-package/issues"
  },
  "homepage": "https://github.com/username/my-package#readme",
  "files": [
    "dist",
    "README.md",
    "LICENSE"
  ],
  "dependencies": {},
  "devDependencies": {
    "typescript": "^5.0.0",
    "jest": "^29.0.0"
  },
  "engines": {
    "node": ">=16.0.0"
  }
}
```

### Build Package

```bash
# Build (if using TypeScript)
npm run build

# Test
npm test

# Check what will be published
npm pack --dry-run

# Create tarball
npm pack
```

### Publish to npm Registry

```bash
# Create npm account at https://www.npmjs.com/signup

# Login
npm login

# Publish package
npm publish

# Publish scoped package (public)
npm publish --access public

# Publish scoped package (private, requires paid account)
npm publish --access restricted

# Publish with tag
npm publish --tag beta

# Unpublish (within 72 hours)
npm unpublish my-package@1.0.0

# Deprecate version
npm deprecate my-package@1.0.0 "Use version 2.0.0 instead"
```

### Versioning

```bash
# Update version
npm version patch  # 1.0.0 -> 1.0.1
npm version minor  # 1.0.0 -> 1.1.0
npm version major  # 1.0.0 -> 2.0.0

# Update and publish
npm version patch && npm publish

# Pre-release versions
npm version prepatch  # 1.0.0 -> 1.0.1-0
npm version preminor  # 1.0.0 -> 1.1.0-0
npm version premajor  # 1.0.0 -> 2.0.0-0
```

### Private npm Registry

#### Using Verdaccio (Self-hosted)

```bash
# Install Verdaccio
npm install -g verdaccio

# Run Verdaccio
verdaccio

# Configure npm to use Verdaccio
npm set registry http://localhost:4873/

# Create user
npm adduser --registry http://localhost:4873/

# Publish to Verdaccio
npm publish --registry http://localhost:4873/

# Install from Verdaccio
npm install my-package --registry http://localhost:4873/
```

#### Using GitHub Packages

```bash
# Create .npmrc in project
echo "@myorg:registry=https://npm.pkg.github.com" > .npmrc

# Login to GitHub Packages
npm login --scope=@myorg --registry=https://npm.pkg.github.com

# Publish
npm publish
```

#### Using Artifactory/Nexus

```bash
# Configure registry
npm config set registry https://artifactory.example.com/api/npm/npm-local/

# Login
npm login --registry=https://artifactory.example.com/api/npm/npm-local/

# Publish
npm publish --registry=https://artifactory.example.com/api/npm/npm-local/
```

### .npmignore

```
# .npmignore
src/
test/
*.test.js
.git
.gitignore
.env
node_modules/
coverage/
.DS_Store
```

---