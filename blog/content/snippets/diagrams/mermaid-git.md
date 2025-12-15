---
title: "Mermaid Git Diagrams"
date: 2024-12-12T22:40:00Z
draft: false
description: "Create Git branch and commit diagrams with Mermaid"
tags: ["mermaid", "git", "version-control", "branching", "diagram", "diagrams"]
category: "diagrams"
---

Git diagrams visualize Git workflows, branches, and commit history. Perfect for documenting branching strategies, release workflows, and Git operations.

## Use Case

Use Git diagrams when you need to:
- Document branching strategies
- Show Git workflows
- Visualize commit history
- Explain merge strategies
- Design release processes

## Code (Basic)

````markdown
```mermaid
gitGraph
    commit
    branch develop
    checkout develop
    commit
    commit
    checkout main
    merge develop
```
````

**Result:**

```mermaid
gitGraph
    commit
    branch develop
    checkout develop
    commit
    commit
    checkout main
    merge develop
```

## Explanation

- `gitGraph` - Start Git diagram (note the capital **G**)
- `commit` - Create a commit
- `branch` - Create a branch
- `checkout` - Switch to a branch
- `merge` - Merge branches

## Examples

### Example 1: Feature Branch Workflow

````markdown
```mermaid
gitGraph
    commit id: "Initial"
    branch develop
    checkout develop
    commit id: "Setup"
    branch feature/login
    checkout feature/login
    commit id: "Add login"
    commit id: "Add validation"
    checkout develop
    merge feature/login
    commit id: "Merge login"
    checkout main
    merge develop
    commit id: "Release v1.0"
```
````

**Result:**

```mermaid
gitGraph
    commit id: "Initial"
    branch develop
    checkout develop
    commit id: "Setup"
    branch feature/login
    checkout feature/login
    commit id: "Add login"
    commit id: "Add validation"
    checkout develop
    merge feature/login
    commit id: "Merge login"
    checkout main
    merge develop
    commit id: "Release v1.0"
```

### Example 2: Git Flow

````markdown
```mermaid
gitGraph
    commit id: "Initial"
    branch develop
    checkout develop
    commit id: "Dev work"
    branch release/v1.0
    checkout release/v1.0
    commit id: "Prepare release"
    checkout main
    merge release/v1.0
    commit id: "v1.0"
    branch hotfix/bugfix
    checkout hotfix/bugfix
    commit id: "Fix bug"
    checkout main
    merge hotfix/bugfix
    commit id: "v1.0.1"
    checkout develop
    merge hotfix/bugfix
```
````

**Result:**

```mermaid
gitGraph
    commit id: "Initial"
    branch develop
    checkout develop
    commit id: "Dev work"
    branch release/v1.0
    checkout release/v1.0
    commit id: "Prepare release"
    checkout main
    merge release/v1.0
    commit id: "v1.0"
    branch hotfix/bugfix
    checkout hotfix/bugfix
    commit id: "Fix bug"
    checkout main
    merge hotfix/bugfix
    commit id: "v1.0.1"
    checkout develop
    merge hotfix/bugfix
```

## Commands

- `commit` or `commit id: "message"` - Create commit (with optional label)
- `branch name` - Create branch
- `checkout name` - Switch to branch
- `merge name` - Merge branch into current
- `cherry-pick id: "message"` - Cherry-pick commit

## Notes

- Commits must have `id:` label
- Branch names should be descriptive
- Use `checkout` before committing to set branch
- Merge creates a merge commit

## Gotchas/Warnings

- ⚠️ **Commits**: Must include `id:` in commit command
- ⚠️ **Order**: Commits must be in chronological order
- ⚠️ **Branches**: Create branch before checking out
- ⚠️ **Merges**: Must checkout target branch before merging

