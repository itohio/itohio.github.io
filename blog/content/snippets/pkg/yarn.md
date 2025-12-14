---
title: "Yarn Package Manager"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "yarn", "javascript", "nodejs", "package-manager"]
---


Yarn package manager for JavaScript/Node.js projects.

---

## Installation

```bash
# Via npm
npm install -g yarn

# Via Corepack (Node.js 16.10+)
corepack enable
corepack prepare yarn@stable --activate

# Verify
yarn --version
```

---

## Basic Commands

```bash
# Initialize project
yarn init
yarn init -y  # Skip questions

# Install dependencies
yarn install
yarn  # Shorthand

# Add package
yarn add package-name
yarn add package-name@version
yarn add package-name@tag

# Add dev dependency
yarn add -D package-name
yarn add --dev package-name

# Add peer dependency
yarn add -P package-name

# Add optional dependency
yarn add -O package-name

# Remove package
yarn remove package-name

# Upgrade package
yarn upgrade package-name
yarn upgrade package-name@version

# Upgrade all packages
yarn upgrade

# Upgrade interactive
yarn upgrade-interactive
yarn upgrade-interactive --latest
```

---

## Workspaces

### Setup Monorepo

```json
// package.json
{
  "name": "my-monorepo",
  "private": true,
  "workspaces": [
    "packages/*"
  ]
}
```

### Workspace Commands

```bash
# Install all workspace dependencies
yarn install

# Add dependency to specific workspace
yarn workspace package-a add lodash

# Run script in workspace
yarn workspace package-a run build

# Run script in all workspaces
yarn workspaces run build
yarn workspaces run test

# List workspaces
yarn workspaces info

# Run command in specific workspace
yarn workspace package-a <command>
```

---

## Scripts

```json
// package.json
{
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js",
    "build": "webpack",
    "test": "jest",
    "lint": "eslint ."
  }
}
```

```bash
# Run script
yarn run start
yarn start  # Shorthand for some scripts

# Run with arguments
yarn test -- --coverage

# List available scripts
yarn run
```

---

## Yarn Berry (v2+)

### Enable Yarn Berry

```bash
# Set version
yarn set version berry
yarn set version stable
yarn set version 3.6.0

# Check version
yarn --version
```

### Zero-Installs

```bash
# Enable PnP (Plug'n'Play)
yarn config set nodeLinker pnp

# Commit .yarn directory
git add .yarn
git add .pnp.cjs
git commit -m "Enable Zero-Installs"

# Install without node_modules
yarn install
```

### Plugins

```bash
# Add plugin
yarn plugin import interactive-tools
yarn plugin import workspace-tools
yarn plugin import version

# List plugins
yarn plugin list

# Remove plugin
yarn plugin remove @yarnpkg/plugin-interactive-tools
```

---

## Lock File

```bash
# Generate lock file
yarn install

# Update lock file
yarn install --mode=update-lockfile

# Check lock file
yarn install --immutable
yarn install --frozen-lockfile  # Yarn 1.x

# Why is package installed?
yarn why package-name
```

---

## Cache

```bash
# Clear cache
yarn cache clean

# Clear specific package
yarn cache clean package-name

# Cache directory
yarn cache dir

# List cache
yarn cache list
```

---

## Configuration

```bash
# Set config
yarn config set registry https://registry.npmjs.org/
yarn config set nodeLinker node-modules

# Get config
yarn config get registry

# List config
yarn config list

# Delete config
yarn config unset registry
```

### .yarnrc.yml (Yarn Berry)

```yaml
# .yarnrc.yml
nodeLinker: node-modules

npmRegistryServer: "https://registry.npmjs.org"

yarnPath: .yarn/releases/yarn-3.6.0.cjs

plugins:
  - path: .yarn/plugins/@yarnpkg/plugin-interactive-tools.cjs
    spec: "@yarnpkg/plugin-interactive-tools"
```

---

## Security

```bash
# Audit dependencies
yarn audit

# Audit and fix
yarn audit --fix  # Yarn 1.x
yarn npm audit --all --recursive  # Yarn Berry

# Check licenses
yarn licenses list
yarn licenses generate-disclaimer
```

---

## Publishing

```bash
# Login
yarn login

# Publish package
yarn publish

# Publish with tag
yarn publish --tag beta

# Publish with new version
yarn publish --new-version 1.0.1

# Unpublish
yarn unpublish package-name@version
```

---

## Yarn vs npm

| Feature | Yarn | npm |
|---------|------|-----|
| Lock file | yarn.lock | package-lock.json |
| Install | `yarn` | `npm install` |
| Add package | `yarn add` | `npm install` |
| Remove | `yarn remove` | `npm uninstall` |
| Run script | `yarn <script>` | `npm run <script>` |
| Workspaces | ✅ | ✅ |
| PnP | ✅ (v2+) | ❌ |

---

## Troubleshooting

```bash
# Clear cache and reinstall
yarn cache clean
rm -rf node_modules
rm yarn.lock
yarn install

# Check integrity
yarn install --check-files

# Verbose output
yarn install --verbose

# Network issues
yarn install --network-timeout 100000

# Offline install
yarn install --offline
```

---

## Docker Integration

```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile --production

# Copy source
COPY . .

# Build
RUN yarn build

CMD ["yarn", "start"]
```

---

## CI/CD

```yaml
# GitHub Actions
name: CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'yarn'
      
      - name: Install dependencies
        run: yarn install --frozen-lockfile
      
      - name: Run tests
        run: yarn test
      
      - name: Build
        run: yarn build
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
└── .yarnrc.yml  # For Yarn Berry
```

### Initialize Package

```bash
# Create package.json
yarn init

# Or interactive
yarn init -y
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
    "prepublishOnly": "yarn build && yarn test"
  },
  "keywords": ["utility", "helper"],
  "author": "Your Name <email@example.com>",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/username/my-package.git"
  },
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

### Build and Test

```bash
# Build
yarn build

# Test
yarn test

# Check what will be published
yarn pack --dry-run

# Create tarball
yarn pack
```

### Publish to npm Registry

```bash
# Login to npm
yarn login

# Publish package
yarn publish

# Publish with tag
yarn publish --tag beta

# Publish with new version
yarn publish --new-version 1.0.1

# Publish scoped package (public)
yarn publish --access public
```

### Versioning

```bash
# Update version
yarn version --patch  # 1.0.0 -> 1.0.1
yarn version --minor  # 1.0.0 -> 1.1.0
yarn version --major  # 1.0.0 -> 2.0.0

# Set specific version
yarn version --new-version 2.0.0

# Pre-release
yarn version --prepatch  # 1.0.0 -> 1.0.1-0
yarn version --preminor  # 1.0.0 -> 1.1.0-0
yarn version --premajor  # 1.0.0 -> 2.0.0-0
```

### Private Registry

#### Using Verdaccio

```bash
# Install Verdaccio
npm install -g verdaccio

# Run Verdaccio
verdaccio

# Configure Yarn
yarn config set registry http://localhost:4873/

# Publish
yarn publish --registry http://localhost:4873/
```

#### Using GitHub Packages

```bash
# Create .yarnrc.yml (Yarn Berry)
npmRegistries:
  "https://npm.pkg.github.com":
    npmAlwaysAuth: true
    npmAuthToken: "${GITHUB_TOKEN}"

# Or .npmrc (Yarn Classic)
@myorg:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}

# Publish
yarn publish
```

#### Using Artifactory

```bash
# Configure registry
yarn config set registry https://artifactory.example.com/api/npm/npm-local/

# Publish
yarn publish --registry https://artifactory.example.com/api/npm/npm-local/
```

### Yarn Workspaces Publishing

```bash
# Publish all workspace packages
yarn workspaces foreach --all publish

# Publish specific workspace
yarn workspace @myorg/package-a publish

# Publish with version bump
yarn workspaces foreach --all version patch
yarn workspaces foreach --all publish
```

---