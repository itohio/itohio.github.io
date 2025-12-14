---
title: "Advanced Git Rebase"
date: 2024-12-12T16:00:00Z
draft: false
description: "Interactive rebase to clean up commit history"
type: "snippet"
tags: ["git", "rebase", "history"]
category: "git"
---



Interactive rebase allows you to edit, reorder, squash, or drop commits before pushing to a remote repository. This is essential for maintaining a clean commit history.

## Use Case

Use this when you have multiple WIP commits that you want to clean up before creating a pull request. It helps maintain a professional and readable git history.

## Code

```bash
# Start interactive rebase for last N commits
git rebase -i HEAD~N

# Or rebase from a specific commit
git rebase -i <commit-hash>

# Common commands in interactive rebase:
# pick = use commit
# reword = use commit, but edit commit message
# edit = use commit, but stop for amending
# squash = use commit, but meld into previous commit
# fixup = like squash, but discard commit message
# drop = remove commit
```

## Explanation

Interactive rebase opens your editor with a list of commits. Each line starts with a command (default is `pick`). You can change these commands to manipulate your commit history.

The commits are listed from oldest to newest (opposite of `git log`). Changes are applied from top to bottom.

## Parameters/Options

- `HEAD~N`: Rebase the last N commits
- `<commit-hash>`: Rebase from a specific commit (not including that commit)
- `-i, --interactive`: Start interactive rebase
- `--autosquash`: Automatically squash commits marked with `fixup!` or `squash!`
- `--continue`: Continue after resolving conflicts
- `--abort`: Cancel the rebase and return to original state

## Examples

### Example 1: Squash last 3 commits

```bash
# Start interactive rebase
git rebase -i HEAD~3

# In the editor, change:
# pick abc123 First commit
# pick def456 Second commit  
# pick ghi789 Third commit
#
# To:
# pick abc123 First commit
# squash def456 Second commit
# squash ghi789 Third commit

# Save and close editor
# Edit the combined commit message
# Save and close
```

**Output:**
```
Successfully rebased and updated refs/heads/main.
```

### Example 2: Reorder commits

```bash
git rebase -i HEAD~4

# Simply reorder the lines in the editor:
# pick abc123 Feature A
# pick def456 Feature B
# pick ghi789 Fix for Feature A
# pick jkl012 Feature C
#
# Becomes:
# pick abc123 Feature A
# pick ghi789 Fix for Feature A
# pick def456 Feature B
# pick jkl012 Feature C
```

### Example 3: Edit a commit message

```bash
git rebase -i HEAD~2

# Change 'pick' to 'reword':
# reword abc123 Old message
# pick def456 Another commit

# Editor will open for you to change the message
```

## Notes

Always create a backup branch before rebasing:
```bash
git branch backup-branch
```

Interactive rebase rewrites history, so only use it on commits that haven't been pushed to a shared remote, or be prepared to force push.

## Gotchas/Warnings

- ⚠️ **Never rebase commits that have been pushed to a public/shared repository** unless you coordinate with your team
- ⚠️ **Force push required after rebase**: Use `git push --force-with-lease` (safer than `--force`)
- ⚠️ **Conflicts may occur**: Be prepared to resolve them with `git rebase --continue` or abort with `git rebase --abort`
- ⚠️ **Lost commits**: If something goes wrong, use `git reflog` to find and recover lost commits