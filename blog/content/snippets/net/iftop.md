---
title: "iftop - Network Bandwidth Monitor"
date: 2024-12-12
draft: false
category: "net"
tags: ["networking", "monitoring", "bandwidth", "iftop"]
---

Real-time network bandwidth monitoring with iftop.

---

## Basic Usage

### Monitor Default Interface

```bash
# Monitor default network interface
sudo iftop

# Monitor specific interface
sudo iftop -i eth0

# Monitor wireless interface
sudo iftop -i wlan0
```

---

## Display Options

### Filter by Host

```bash
# Show traffic to/from specific host
sudo iftop -f "host 192.168.1.100"

# Show traffic to specific network
sudo iftop -f "net 192.168.1.0/24"

# Show traffic on specific port
sudo iftop -f "port 80"

# Show traffic to specific host and port
sudo iftop -f "host 192.168.1.100 and port 443"
```

### Display Modes

```bash
# Show ports instead of services
sudo iftop -P

# Show port numbers (not service names)
sudo iftop -N

# Don't resolve hostnames (faster, shows IPs)
sudo iftop -n

# Show bandwidth in bytes (not bits)
sudo iftop -B

# Disable promiscuous mode (only show traffic to/from this host)
sudo iftop -p
```

---

## Output Formats

### Text Mode (Non-Interactive)

```bash
# Text output (useful for logging)
sudo iftop -t -s 10

# Text output with specific number of iterations
sudo iftop -t -s 5 -L 10

# Output to file
sudo iftop -t -s 10 > bandwidth.log
```

**Flags**:
- `-t`: Text mode (non-interactive)
- `-s N`: Update every N seconds
- `-L N`: Number of lines to display

---

## Interactive Keys

While iftop is running, use these keys:

### Display Options

```
h           Show/hide help
n           Toggle hostname resolution
N           Toggle port resolution
s/d         Toggle source/destination display
S/D         Toggle source/destination ports
p           Toggle promiscuous mode
P           Toggle port display
b           Toggle bar graph display
B           Toggle bytes/bits display
t           Toggle text interface (cumulative/avg/peak)
T           Toggle cumulative line totals
j/k         Scroll display
```

### Sorting

```
< or >      Sort by source/destination
1/2/3       Sort by 2s/10s/40s average
```

### Filtering

```
f           Edit filter
l           Set screen filter
L           Set line display filter
!           Shell command
q           Quit
```

---

## Filters (BPF Syntax)

### Host Filters

```bash
# Single host
sudo iftop -f "host 192.168.1.100"

# Multiple hosts
sudo iftop -f "host 192.168.1.100 or host 192.168.1.101"

# Exclude host
sudo iftop -f "not host 192.168.1.100"

# Source host
sudo iftop -f "src host 192.168.1.100"

# Destination host
sudo iftop -f "dst host 192.168.1.100"
```

### Network Filters

```bash
# Specific network
sudo iftop -f "net 192.168.1.0/24"

# Exclude network
sudo iftop -f "not net 192.168.1.0/24"

# Multiple networks
sudo iftop -f "net 192.168.1.0/24 or net 10.0.0.0/8"
```

### Port Filters

```bash
# Specific port
sudo iftop -f "port 80"

# Multiple ports
sudo iftop -f "port 80 or port 443"

# Port range
sudo iftop -f "portrange 8000-9000"

# Source port
sudo iftop -f "src port 80"

# Destination port
sudo iftop -f "dst port 443"

# Exclude port
sudo iftop -f "not port 22"
```

### Protocol Filters

```bash
# TCP only
sudo iftop -f "tcp"

# UDP only
sudo iftop -f "udp"

# ICMP only
sudo iftop -f "icmp"

# Specific protocol
sudo iftop -f "ip proto 6"  # TCP
sudo iftop -f "ip proto 17" # UDP
```

### Combined Filters

```bash
# HTTP/HTTPS traffic
sudo iftop -f "port 80 or port 443"

# SSH traffic to specific host
sudo iftop -f "host 192.168.1.100 and port 22"

# All traffic except SSH
sudo iftop -f "not port 22"

# Web traffic from specific network
sudo iftop -f "src net 192.168.1.0/24 and (port 80 or port 443)"

# DNS queries
sudo iftop -f "udp and port 53"
```

---

## Common Use Cases

### Monitor Web Server Traffic

```bash
# HTTP/HTTPS traffic
sudo iftop -i eth0 -f "port 80 or port 443" -P

# Show top bandwidth consumers
sudo iftop -i eth0 -f "port 80 or port 443" -n -P
```

### Monitor Database Connections

```bash
# PostgreSQL
sudo iftop -f "port 5432"

# MySQL
sudo iftop -f "port 3306"

# Redis
sudo iftop -f "port 6379"

# MongoDB
sudo iftop -f "port 27017"
```

### Monitor Specific Application

```bash
# Docker containers (bridge network)
sudo iftop -i docker0

# Kubernetes pods
sudo iftop -i cni0

# VPN traffic
sudo iftop -i tun0
```

### Find Bandwidth Hogs

```bash
# Show all traffic, sorted by bandwidth
sudo iftop -n -P -B

# Text mode, log top consumers
sudo iftop -t -s 10 -n -P -B -L 20 > bandwidth_report.txt
```

---

## Logging & Monitoring

### Continuous Logging

```bash
# Log every 10 seconds
while true; do
  sudo iftop -t -s 10 -n -P -B -L 10 >> /var/log/bandwidth.log
  echo "---" >> /var/log/bandwidth.log
  sleep 10
done
```

### One-Time Snapshot

```bash
# Capture 30 seconds of data
sudo iftop -t -s 30 -n -P -B > bandwidth_snapshot.txt
```

### Top N Connections

```bash
# Show top 10 connections
sudo iftop -t -s 10 -n -P -L 10
```

---

## Comparison with Other Tools

### iftop vs nethogs

```bash
# iftop: Shows connections between hosts
sudo iftop -i eth0

# nethogs: Shows bandwidth per process
sudo nethogs eth0
```

**Use iftop when**:
- You want to see which remote hosts are using bandwidth
- You need to monitor specific ports or protocols
- You want real-time connection-level monitoring

**Use nethogs when**:
- You want to see which local processes are using bandwidth
- You need per-application monitoring

### iftop vs iptraf-ng

```bash
# iftop: Connection-based view
sudo iftop

# iptraf-ng: Packet-level statistics
sudo iptraf-ng
```

---

## Configuration File

### Create ~/.iftoprc

```bash
# ~/.iftoprc or /etc/iftoprc

# Interface to monitor
interface: eth0

# DNS resolution
dns-resolution: yes
port-resolution: yes

# Display options
show-bars: yes
promiscuous: no
port-display: on

# Bandwidth units
use-bytes: no

# Filter
filter-code: port 80 or port 443

# Update interval
line-display: two-line
```

---

## Installation

```bash
# Debian/Ubuntu
sudo apt install iftop

# RHEL/CentOS/Fedora
sudo yum install iftop
# or
sudo dnf install iftop

# Arch Linux
sudo pacman -S iftop

# macOS (Homebrew)
brew install iftop

# FreeBSD
sudo pkg install iftop
```

---

## Troubleshooting

### Permission Denied

```bash
# iftop requires root privileges
sudo iftop

# Or add user to pcap group (Debian/Ubuntu)
sudo usermod -a -G pcap $USER
# Logout and login again
```

### Interface Not Found

```bash
# List available interfaces
ip link show
# or
ifconfig -a

# Use correct interface name
sudo iftop -i eth0
```

### High CPU Usage

```bash
# Disable DNS resolution
sudo iftop -n

# Disable port resolution
sudo iftop -N

# Increase update interval
sudo iftop -s 10
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `sudo iftop` | Monitor default interface |
| `sudo iftop -i eth0` | Monitor specific interface |
| `sudo iftop -n` | Don't resolve hostnames |
| `sudo iftop -P` | Show ports |
| `sudo iftop -B` | Show bytes (not bits) |
| `sudo iftop -f "port 80"` | Filter by port |
| `sudo iftop -f "host 192.168.1.100"` | Filter by host |
| `sudo iftop -t -s 10` | Text mode, 10s updates |

### Interactive Keys

| Key | Action |
|-----|--------|
| `h` | Help |
| `n` | Toggle hostname resolution |
| `P` | Toggle port display |
| `b` | Toggle bar graph |
| `t` | Cycle through display modes |
| `1/2/3` | Sort by 2s/10s/40s average |
| `q` | Quit |

---

## Tips

- Use `-n` and `-N` for faster display (no DNS/port lookups)
- Use `-B` to see bandwidth in bytes (easier to read)
- Press `t` multiple times to cycle through different views
- Use filters to focus on specific traffic
- Combine with `nethogs` for complete bandwidth picture
- Use text mode (`-t`) for logging and automation
- Press `1`, `2`, or `3` to sort by different time averages

