---
title: "iperf & iperf3 Network Testing"
date: 2024-12-12
draft: false
category: "net"
tags: ["net-knowhow", "iperf", "bandwidth", "performance"]
---


Network bandwidth testing with iperf and iperf3. Understand differences and common usage patterns.

---

## iperf vs iperf3

### Key Differences

| Feature | iperf2 | iperf3 |
|---------|--------|--------|
| **Compatibility** | Not compatible with iperf3 | Not compatible with iperf2 |
| **Parallel Streams** | Multiple streams in both directions | One direction at a time |
| **Simultaneous Bi-directional** | Yes | No (sequential only) |
| **JSON Output** | No | Yes |
| **UDP Bandwidth** | Must specify | Auto-detects |
| **Active Development** | Less active | Actively maintained |
| **Default Port** | 5001 | 5201 |

**Recommendation**: Use **iperf3** for new projects (better maintained, JSON output, simpler).

---

## Installation

```bash
# Linux
sudo apt install iperf3
sudo apt install iperf  # For iperf2

# macOS
brew install iperf3
brew install iperf2

# Windows (Scoop)
scoop install iperf3
```

---

## iperf3 Commands

### Basic TCP Test

```bash
# Server mode (run on server)
iperf3 -s

# Client mode (run on client)
iperf3 -c server_ip

# Output:
# [ ID] Interval           Transfer     Bitrate
# [  5]   0.00-10.00  sec  1.09 GBytes   941 Mbits/sec
```

### Test Duration

```bash
# Test for 30 seconds (default is 10)
iperf3 -c server_ip -t 30

# Test for 60 seconds
iperf3 -c server_ip -t 60
```

### Parallel Streams

```bash
# Use 4 parallel streams
iperf3 -c server_ip -P 4

# Use 10 parallel streams
iperf3 -c server_ip -P 10
```

### Reverse Mode

```bash
# Server sends data to client (reverse direction)
iperf3 -c server_ip -R

# Useful for testing upload vs download speeds
```

### UDP Testing

```bash
# UDP test with 100 Mbps bandwidth
iperf3 -c server_ip -u -b 100M

# UDP test with 1 Gbps bandwidth
iperf3 -c server_ip -u -b 1G

# Show packet loss
iperf3 -c server_ip -u -b 100M
# Output includes: Jitter, Lost/Total Datagrams
```

### Specific Port

```bash
# Server on custom port
iperf3 -s -p 5555

# Client to custom port
iperf3 -c server_ip -p 5555
```

### JSON Output

```bash
# Output results in JSON format
iperf3 -c server_ip -J > results.json

# Parse with jq
iperf3 -c server_ip -J | jq '.end.sum_received.bits_per_second'
```

### Interval Reporting

```bash
# Report every 1 second (default)
iperf3 -c server_ip -i 1

# Report every 5 seconds
iperf3 -c server_ip -i 5

# No interval reports (only final)
iperf3 -c server_ip -i 0
```

### Bandwidth Limit

```bash
# Limit to 10 Mbps
iperf3 -c server_ip -b 10M

# Limit to 100 Mbps
iperf3 -c server_ip -b 100M
```

### Bi-directional Test

```bash
# Test both directions (sequential in iperf3)
iperf3 -c server_ip --bidir

# First tests client->server, then server->client
```

### IPv6

```bash
# Server with IPv6
iperf3 -s -6

# Client with IPv6
iperf3 -c server_ipv6 -6
```

---

## iperf2 Commands

### Basic TCP Test

```bash
# Server
iperf -s

# Client
iperf -c server_ip
```

### Simultaneous Bi-directional

```bash
# Test both directions simultaneously (iperf2 only)
iperf -c server_ip -d

# Full duplex test
iperf -c server_ip -r  # Sequential bi-directional
```

### Window Size

```bash
# Set TCP window size
iperf -c server_ip -w 256K
```

---

## Common Use Cases

### Test LAN Speed

```bash
# Server
iperf3 -s

# Client (expect ~940 Mbps on gigabit ethernet)
iperf3 -c 192.168.1.100 -t 30
```

### Test WiFi Speed

```bash
# Server (wired connection)
iperf3 -s

# Client (WiFi device)
iperf3 -c server_ip -t 60 -i 1

# Monitor for fluctuations in WiFi performance
```

### Test VPN Throughput

```bash
# Server (outside VPN)
iperf3 -s

# Client (inside VPN)
iperf3 -c vpn_server_ip -t 30

# Compare with direct connection
```

### Test Multiple Connections

```bash
# Simulate multiple users
iperf3 -c server_ip -P 10 -t 60

# Useful for testing router/switch capacity
```

### Test Packet Loss (UDP)

```bash
# Server
iperf3 -s

# Client - UDP test
iperf3 -c server_ip -u -b 100M -t 30

# Check output for packet loss percentage
# Jitter: 0.015 ms
# Lost/Total Datagrams: 0/75000 (0%)
```

### Continuous Monitoring

```bash
# Run server in background
iperf3 -s -D

# Or with systemd
sudo systemctl start iperf3

# Run periodic tests
while true; do
    iperf3 -c server_ip -t 10 -J >> results.log
    sleep 300  # Every 5 minutes
done
```

---

## Docker Setup

```yaml
version: '3.8'

services:
  iperf3-server:
    image: networkstatic/iperf3
    container_name: iperf3-server
    command: -s
    ports:
      - "5201:5201"
    restart: unless-stopped

  # Client (for testing)
  iperf3-client:
    image: networkstatic/iperf3
    container_name: iperf3-client
    command: -c iperf3-server -t 10
    depends_on:
      - iperf3-server
```

```bash
# Run server
docker run -d --name iperf3-server -p 5201:5201 networkstatic/iperf3 -s

# Run client
docker run --rm networkstatic/iperf3 -c host.docker.internal
```

---

## Interpreting Results

### TCP Results

```
[ ID] Interval           Transfer     Bitrate         Retr
[  5]   0.00-10.00  sec  1.09 GBytes   941 Mbits/sec    0    sender
[  5]   0.00-10.04  sec  1.09 GBytes   937 Mbits/sec         receiver
```

- **Transfer**: Amount of data transferred
- **Bitrate**: Speed in Mbits/sec or Gbits/sec
- **Retr**: Retransmissions (should be 0 or very low)

**Good Results:**
- Gigabit Ethernet: ~940 Mbps (TCP overhead)
- 10 Gigabit: ~9.4 Gbps
- WiFi 5 (802.11ac): 200-600 Mbps
- WiFi 6 (802.11ax): 600-1200 Mbps

### UDP Results

```
[ ID] Interval           Transfer     Bitrate         Jitter    Lost/Total Datagrams
[  5]   0.00-10.00  sec   119 MBytes  99.8 Mbits/sec  0.015 ms  0/85000 (0%)
```

- **Jitter**: Variation in packet delay (lower is better)
- **Lost/Total**: Packet loss (should be 0% or very low)

**Good Results:**
- Jitter: < 1 ms (excellent), < 10 ms (good)
- Packet Loss: 0% (excellent), < 1% (acceptable), > 5% (poor)

---

## Troubleshooting

### Low Throughput

```bash
# Check for retransmissions
iperf3 -c server_ip -t 30

# If high retransmissions:
# - Check network cables
# - Check switch/router
# - Check for interference (WiFi)
# - Check CPU usage on both ends

# Try increasing TCP window
iperf3 -c server_ip -w 256K

# Try parallel streams
iperf3 -c server_ip -P 4
```

### Firewall Issues

```bash
# Allow iperf3 port
sudo ufw allow 5201/tcp
sudo ufw allow 5201/udp

# Or iptables
sudo iptables -A INPUT -p tcp --dport 5201 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 5201 -j ACCEPT
```

### Connection Refused

```bash
# Check if server is running
ps aux | grep iperf

# Check if port is listening
netstat -tuln | grep 5201
ss -tuln | grep 5201

# Test connectivity
ping server_ip
telnet server_ip 5201
```

---

## Automation Script

```bash
#!/bin/bash
# iperf3-test.sh - Automated network testing

SERVER="192.168.1.100"
DURATION=30
OUTPUT_DIR="./iperf_results"

mkdir -p $OUTPUT_DIR

# TCP Test
echo "Running TCP test..."
iperf3 -c $SERVER -t $DURATION -J > $OUTPUT_DIR/tcp_$(date +%Y%m%d_%H%M%S).json

# UDP Test
echo "Running UDP test..."
iperf3 -c $SERVER -u -b 100M -t $DURATION -J > $OUTPUT_DIR/udp_$(date +%Y%m%d_%H%M%S).json

# Parallel streams
echo "Running parallel streams test..."
iperf3 -c $SERVER -P 4 -t $DURATION -J > $OUTPUT_DIR/parallel_$(date +%Y%m%d_%H%M%S).json

# Reverse test
echo "Running reverse test..."
iperf3 -c $SERVER -R -t $DURATION -J > $OUTPUT_DIR/reverse_$(date +%Y%m%d_%H%M%S).json

echo "Tests complete. Results in $OUTPUT_DIR"
```

---

## Quick Reference

```bash
# Server
iperf3 -s                          # Start server

# Basic tests
iperf3 -c server_ip                # TCP test
iperf3 -c server_ip -u -b 100M     # UDP test
iperf3 -c server_ip -R             # Reverse (server to client)
iperf3 -c server_ip -P 4           # 4 parallel streams

# Advanced
iperf3 -c server_ip -t 60          # 60 second test
iperf3 -c server_ip -J             # JSON output
iperf3 -c server_ip -i 5           # Report every 5 seconds
iperf3 -c server_ip -p 5555        # Custom port
```

---