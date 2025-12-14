---
title: "Ignite CLI for Cosmos SDK"
date: 2024-12-12
draft: false
category: "blockchain"
tags: ["blockchain-knowhow", "cosmos", "ignite", "cli"]
---


Ignite CLI commands for building Cosmos SDK blockchains. Quick reference for blockchain development workflow.

---

## Installation

```bash
# Install Ignite CLI
curl https://get.ignite.com/cli! | bash

# Or with Homebrew (macOS)
brew install ignite

# Verify installation
ignite version
```

---

## Scaffold New Blockchain

```bash
# Create new blockchain
ignite scaffold chain github.com/username/mychain

# Create with specific options
ignite scaffold chain github.com/username/mychain \
  --no-module \
  --address-prefix mychain

# Navigate to project
cd mychain
```

---

## Scaffold Modules

```bash
# Create new module
ignite scaffold module blog

# Create module with IBC
ignite scaffold module blog --ibc

# Create module with dependencies
ignite scaffold module blog --dep account,bank
```

---

## Scaffold Types & Messages

```bash
# Scaffold a list (CRUD)
ignite scaffold list post title body --module blog

# Scaffold a map
ignite scaffold map product name:string price:uint --module store

# Scaffold a single type
ignite scaffold single config maxPosts:uint --module blog

# Scaffold a message (transaction)
ignite scaffold message create-post title body --module blog

# Scaffold a query
ignite scaffold query posts --response posts:Post --module blog

# Scaffold with custom types
ignite scaffold list comment postID:uint body creator --module blog
```

---

## Scaffold IBC

```bash
# Scaffold IBC packet
ignite scaffold packet ibcPost title body --module blog

# Scaffold IBC packet with acknowledgement
ignite scaffold packet ibcPost title body --ack postID:uint --module blog

# Scaffold IBC module
ignite scaffold module blog --ibc
ignite scaffold packet sendPost title:string content:string --module blog
```

---

## Build & Serve

```bash
# Start development chain
ignite chain serve

# Serve with specific config
ignite chain serve -c config.yml

# Serve with reset
ignite chain serve --reset-once

# Serve with verbose output
ignite chain serve -v

# Build binary only
ignite chain build

# Build with specific output
ignite chain build --output ./build

# Install binary to $GOPATH/bin
ignite chain build --release
```

---

## Chain Initialization

```bash
# Initialize chain
ignite chain init

# Initialize with custom home directory
ignite chain init --home ~/.mychain

# Clear chain data
ignite chain init --reset-once
```

---

## Testing

```bash
# Run tests
ignite chain test

# Run specific test
ignite chain test ./x/blog/keeper/...

# Run with coverage
ignite chain test --coverage
```

---

## Code Generation

```bash
# Generate code (proto, OpenAPI, etc.)
ignite generate proto-go

# Generate OpenAPI spec
ignite generate openapi

# Generate TypeScript client
ignite generate ts-client

# Generate all
ignite generate
```

---

## Relayer (IBC)

```bash
# Configure relayer
ignite relayer configure

# Start relayer
ignite relayer connect

# Check relayer status
ignite relayer status
```

---

## Network

```bash
# Join a network
ignite network chain join <chain-id>

# Publish chain
ignite network chain publish github.com/username/mychain

# List available chains
ignite network chain list

# Show chain info
ignite network chain show <chain-id>
```

---

## Accounts & Transactions

```bash
# Create account
ignite account create alice

# List accounts
ignite account list

# Show account details
ignite account show alice

# Send transaction
mychaind tx blog create-post "Hello" "World" \
  --from alice \
  --chain-id mychain \
  --yes

# Query
mychaind query blog list-post
```

---

## Configuration Files

### config.yml

```yaml
version: 1

build:
  binary: mychaind
  proto:
    path: proto
    third_party_paths:
      - third_party/proto
      - proto_vendor

accounts:
  - name: alice
    coins:
      - 100000000stake
      - 1000000token
  - name: bob
    coins:
      - 50000000stake

validator:
  name: alice
  staked: 100000000stake

faucet:
  name: bob
  coins:
    - 5stake
    - 100token
  port: 4500

genesis:
  chain_id: mychain-1
  app_state:
    staking:
      params:
        bond_denom: stake
```

---

## Project Structure

```
mychain/
├── app/                    # Application logic
│   ├── app.go
│   └── encoding.go
├── cmd/
│   └── mychaind/          # CLI binary
│       └── main.go
├── proto/                 # Protocol Buffers
│   └── mychain/
│       └── blog/
│           ├── genesis.proto
│           ├── query.proto
│           ├── tx.proto
│           └── types.proto
├── x/                     # Custom modules
│   └── blog/
│       ├── client/
│       ├── keeper/
│       ├── types/
│       └── module.go
├── testutil/              # Test utilities
├── docs/                  # Documentation
├── config.yml             # Ignite config
└── go.mod
```

---

## Common Workflows

### Create Blog Module

```bash
# 1. Scaffold chain
ignite scaffold chain github.com/username/blogchain
cd blogchain

# 2. Scaffold blog module
ignite scaffold module blog

# 3. Scaffold post type
ignite scaffold list post title body creator --module blog

# 4. Scaffold comment type
ignite scaffold list comment postID:uint body creator --module blog

# 5. Add custom message
ignite scaffold message like-post postID:uint --module blog

# 6. Start chain
ignite chain serve

# 7. Test
blogchaind tx blog create-post "First Post" "Hello World" --from alice --yes
blogchaind query blog list-post
```

### Create IBC-Enabled Module

```bash
# 1. Scaffold chain
ignite scaffold chain github.com/username/ibcchain
cd ibcchain

# 2. Scaffold IBC module
ignite scaffold module ibcblog --ibc

# 3. Scaffold IBC packet
ignite scaffold packet sendPost title:string content:string \
  --ack postID:uint \
  --module ibcblog

# 4. Start chain
ignite chain serve

# 5. Configure relayer (in another terminal)
ignite relayer configure
ignite relayer connect
```

---

## Troubleshooting

```bash
# Clear cache and rebuild
ignite chain serve --reset-once

# Check for errors
ignite chain build --verbose

# View logs
tail -f ~/.mychain/logs/mychain.log

# Reset everything
rm -rf ~/.mychain
ignite chain serve --reset-once

# Update dependencies
go mod tidy
go mod download
```

---

## Makefile Integration

```makefile
.PHONY: serve build test proto clean

serve:
	ignite chain serve

serve-reset:
	ignite chain serve --reset-once

build:
	ignite chain build

test:
	ignite chain test

proto:
	ignite generate proto-go

ts-client:
	ignite generate ts-client

clean:
	rm -rf ~/.mychain
	rm -rf build/
```

---

## Best Practices

```
✅ Use semantic versioning for modules
✅ Write tests for keeper functions
✅ Document proto files with comments
✅ Use events for important state changes
✅ Validate input in message handlers
✅ Use proper error handling
✅ Keep modules focused and small
✅ Use IBC for cross-chain communication
```

---