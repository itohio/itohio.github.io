---
title: "Bash Commands & Scripting"
date: 2024-12-12
draft: false
description: "Useful bash commands and scripting techniques"
tags: ["bash", "shell", "linux", "unix", "scripting"]
---



## Command Dispatcher Pattern

Create a script that dispatches commands via switch/case:

```bash
#!/bin/bash

# Command dispatcher script
command_dispatcher() {
    local cmd="$1"
    shift  # Remove first argument, rest are parameters
    
    case "$cmd" in
        start)
            echo "Starting service..."
            # Your start logic here
            ;;
        stop)
            echo "Stopping service..."
            # Your stop logic here
            ;;
        restart)
            stop
            start
            ;;
        status)
            echo "Checking status..."
            # Your status logic here
            ;;
        deploy)
            local env="$1"
            echo "Deploying to $env..."
            # Your deploy logic here
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|deploy <env>}"
            exit 1
            ;;
    esac
}

# Call dispatcher with all arguments
command_dispatcher "$@"
```

## Useful Commands

### File Operations

```bash
# Find files modified in last 7 days
find . -type f -mtime -7

# Find and delete files older than 30 days
find . -type f -mtime +30 -delete

# Find files by size (larger than 100MB)
find . -type f -size +100M

# Find and replace in files
find . -type f -name "*.txt" -exec sed -i 's/old/new/g' {} +

# Copy with progress
rsync -ah --progress source/ destination/

# Create directory structure
mkdir -p path/to/nested/directories

# Disk usage sorted by size
du -sh * | sort -h

# Find largest files
find . -type f -exec du -h {} + | sort -rh | head -n 10
```

### Text Processing

```bash
# Extract column from CSV
cut -d',' -f2 file.csv

# Count lines/words/chars
wc -l file.txt

# Remove duplicate lines
sort file.txt | uniq

# Count unique lines
sort file.txt | uniq -c

# Get lines between patterns
sed -n '/START/,/END/p' file.txt

# Replace text in-place
sed -i 's/old/new/g' file.txt

# Print specific lines
sed -n '10,20p' file.txt

# AWK: sum column
awk '{sum+=$1} END {print sum}' file.txt

# AWK: print if condition
awk '$3 > 100' file.txt

# Join lines
paste -sd',' file.txt
```

### Process Management

```bash
# Find process by name
ps aux | grep process_name

# Kill process by name
pkill process_name

# Kill process by port
lsof -ti:8080 | xargs kill -9

# Run in background
command &

# Run and detach from terminal
nohup command &

# Check if process is running
pgrep -x process_name > /dev/null && echo "Running" || echo "Not running"

# Monitor process
watch -n 1 'ps aux | grep process_name'
```

### Network Commands

```bash
# Check port is open
nc -zv hostname 80

# Download file
curl -O https://example.com/file.zip

# POST JSON data
curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' https://api.example.com

# Check HTTP status
curl -I https://example.com

# Test DNS
dig example.com

# Show listening ports
netstat -tuln

# Show network connections
ss -tuln
```

### System Information

```bash
# CPU info
lscpu

# Memory info
free -h

# Disk space
df -h

# System uptime
uptime

# Kernel version
uname -r

# Distribution info
cat /etc/os-release

# List loaded modules
lsmod

# Hardware info
lshw -short
```

## Scripting Techniques

### Error Handling

```bash
#!/bin/bash
set -euo pipefail  # Exit on error, undefined var, pipe failure

# Trap errors
trap 'echo "Error on line $LINENO"' ERR

# Check command success
if ! command -v git &> /dev/null; then
    echo "git not found"
    exit 1
fi
```

### Argument Parsing

```bash
#!/bin/bash

# Parse options
while getopts "hf:v" opt; do
    case $opt in
        h)
            echo "Usage: $0 [-h] [-f file] [-v]"
            exit 0
            ;;
        f)
            file="$OPTARG"
            ;;
        v)
            verbose=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# Shift past options
shift $((OPTIND-1))

# Remaining arguments
echo "Remaining args: $@"
```

### Functions

```bash
#!/bin/bash

# Function with return value
get_timestamp() {
    date +%Y%m%d_%H%M%S
}

# Function with parameters
backup_file() {
    local file="$1"
    local backup_dir="$2"
    
    if [[ ! -f "$file" ]]; then
        echo "Error: File not found" >&2
        return 1
    fi
    
    cp "$file" "$backup_dir/$(basename "$file").$(get_timestamp)"
}

# Usage
backup_file "/path/to/file.txt" "/backup"
```

### Arrays

```bash
# Declare array
files=("file1.txt" "file2.txt" "file3.txt")

# Iterate array
for file in "${files[@]}"; do
    echo "$file"
done

# Array length
echo "${#files[@]}"

# Add to array
files+=("file4.txt")

# Associative array (dictionary)
declare -A config
config[host]="localhost"
config[port]="8080"

echo "${config[host]}"
```

### String Operations

```bash
# String length
str="hello"
echo "${#str}"

# Substring
echo "${str:0:2}"  # "he"

# Replace
echo "${str/l/L}"  # "heLlo"

# Replace all
echo "${str//l/L}"  # "heLLo"

# Remove prefix
path="/usr/local/bin"
echo "${path#/usr/}"  # "local/bin"

# Remove suffix
file="document.txt"
echo "${file%.txt}"  # "document"

# Convert to uppercase
echo "${str^^}"

# Convert to lowercase
echo "${str,,}"
```

### Conditionals

```bash
# File tests
if [[ -f "file.txt" ]]; then
    echo "File exists"
fi

if [[ -d "directory" ]]; then
    echo "Directory exists"
fi

if [[ -x "script.sh" ]]; then
    echo "File is executable"
fi

# String tests
if [[ -z "$var" ]]; then
    echo "Variable is empty"
fi

if [[ "$str1" == "$str2" ]]; then
    echo "Strings match"
fi

# Numeric tests
if [[ $num -gt 10 ]]; then
    echo "Greater than 10"
fi

# Logical operators
if [[ -f "file.txt" && -r "file.txt" ]]; then
    echo "File exists and is readable"
fi
```

### Loops

```bash
# For loop
for i in {1..10}; do
    echo "$i"
done

# C-style for loop
for ((i=0; i<10; i++)); do
    echo "$i"
done

# While loop
while read line; do
    echo "$line"
done < file.txt

# Until loop
count=0
until [[ $count -eq 5 ]]; do
    echo "$count"
    ((count++))
done
```

### Here Documents

```bash
# Multi-line string
cat << EOF > config.txt
server {
    listen 80;
    server_name example.com;
}
EOF

# Here string
grep "pattern" <<< "string to search"
```

## One-Liners

```bash
# Backup with timestamp
cp file.txt file.txt.$(date +%Y%m%d_%H%M%S)

# Find and archive
tar czf archive.tar.gz $(find . -name "*.log")

# Monitor log file
tail -f /var/log/syslog | grep --line-buffered "ERROR"

# Parallel execution
cat urls.txt | xargs -P 10 -I {} curl -O {}

# Generate random string
openssl rand -base64 32

# Check if script is run as root
[[ $EUID -ne 0 ]] && echo "Must run as root" && exit 1

# Create backup and restore functions
alias backup='tar czf backup-$(date +%Y%m%d).tar.gz'
alias restore='tar xzf'
```

## Further Reading

- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/)
- [Advanced Bash-Scripting Guide](https://tldp.org/LDP/abs/html/)
- [ShellCheck](https://www.shellcheck.net/) - Shell script analysis tool

