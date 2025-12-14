---
title: "Zsh/Sh Commands"
date: 2024-12-12
draft: false
description: "Z shell and Bourne shell commands"
tags: ["zsh", "sh", "shell", "unix", "linux"]
---



## Zsh-Specific Features

### Globbing

```bash
# Recursive globbing
ls **/*.txt

# Exclude pattern
ls ^*.txt

# Numeric ranges
ls file<1-10>.txt

# Qualifiers
ls *(.)      # Files only
ls *(/)      # Directories only
ls *(.x)     # Executable files
ls *(m-7)    # Modified in last 7 days
ls *(Lm+100) # Larger than 100MB
```

### Arrays

```bash
# Array creation
arr=(one two three)

# Access elements
echo $arr[1]  # First element (1-indexed in zsh)
echo $arr[-1] # Last element

# Array slicing
echo $arr[2,3]

# Array length
echo $#arr
```

### Parameter Expansion

```bash
# Default value
echo ${VAR:-default}

# Assign default
echo ${VAR:=default}

# Substring
str="hello world"
echo ${str:0:5}  # "hello"

# Remove pattern
path="/usr/local/bin"
echo ${path#*/}   # Remove shortest match from start
echo ${path##*/}  # Remove longest match from start
echo ${path%/*}   # Remove shortest match from end
echo ${path%%/*}  # Remove longest match from end

# Replace
echo ${str/world/universe}
```

### Completions

```bash
# Enable completions
autoload -Uz compinit && compinit

# Case-insensitive completion
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Za-z}'

# Menu selection
zstyle ':completion:*' menu select

# Rehash for new commands
rehash
```

### History

```bash
# History search
Ctrl+R  # Reverse search

# History expansion
!!      # Last command
!$      # Last argument
!^      # First argument
!*      # All arguments
!-2     # 2nd last command

# Search history
history | grep pattern

# Execute from history
!123    # Execute command #123
```

## Bourne Shell (sh) Compatible

### POSIX Scripting

```sh
#!/bin/sh

# Variables (no spaces around =)
VAR="value"

# Command substitution
TODAY=$(date +%Y-%m-%d)
TODAY=`date +%Y-%m-%d`  # Old style

# Conditionals
if [ -f "file.txt" ]; then
    echo "File exists"
elif [ -d "dir" ]; then
    echo "Directory exists"
else
    echo "Not found"
fi

# String tests
if [ "$VAR" = "value" ]; then
    echo "Match"
fi

if [ -z "$VAR" ]; then
    echo "Empty"
fi

if [ -n "$VAR" ]; then
    echo "Not empty"
fi

# Numeric tests
if [ "$NUM" -gt 10 ]; then
    echo "Greater than 10"
fi

# File tests
[ -e file ]  # Exists
[ -f file ]  # Regular file
[ -d dir ]   # Directory
[ -r file ]  # Readable
[ -w file ]  # Writable
[ -x file ]  # Executable
```

### Loops

```sh
# For loop
for file in *.txt; do
    echo "$file"
done

# While loop
while read line; do
    echo "$line"
done < file.txt

# Until loop
count=0
until [ $count -eq 5 ]; do
    echo $count
    count=$((count + 1))
done

# C-style (not POSIX)
i=0
while [ $i -lt 10 ]; do
    echo $i
    i=$((i + 1))
done
```

### Functions

```sh
# Function definition
my_function() {
    echo "Arg 1: $1"
    echo "Arg 2: $2"
    return 0
}

# Call function
my_function arg1 arg2

# Check return value
if my_function; then
    echo "Success"
fi
```

## Zsh Configuration

### .zshrc Essentials

```bash
# Oh My Zsh
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"
plugins=(git docker kubectl)
source $ZSH/oh-my-zsh.sh

# Aliases
alias ll='ls -lah'
alias gs='git status'
alias gp='git pull'

# Custom prompt
PROMPT='%F{cyan}%n@%m%f:%F{yellow}%~%f$ '

# History settings
HISTSIZE=10000
SAVEHIST=10000
HISTFILE=~/.zsh_history
setopt SHARE_HISTORY
setopt HIST_IGNORE_DUPS

# Key bindings
bindkey '^[[A' history-search-backward
bindkey '^[[B' history-search-forward

# Auto-cd
setopt AUTO_CD

# Correction
setopt CORRECT
setopt CORRECT_ALL
```

## Useful Aliases

```bash
# Navigation
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'

# Git
alias g='git'
alias ga='git add'
alias gc='git commit'
alias gco='git checkout'
alias gd='git diff'
alias gl='git log --oneline'

# Docker
alias d='docker'
alias dc='docker-compose'
alias dps='docker ps'
alias dimg='docker images'

# System
alias update='sudo apt update && sudo apt upgrade'
alias ports='netstat -tuln'
alias myip='curl ifconfig.me'
```

## Further Reading

- [Zsh Documentation](https://zsh.sourceforge.io/Doc/)
- [Oh My Zsh](https://ohmyz.sh/)
- [POSIX Shell Specification](https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html)

