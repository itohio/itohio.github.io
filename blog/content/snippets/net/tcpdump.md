---
title: "tcpdump - Packet Capture"
date: 2024-12-12
draft: false
category: "net"
tags: ["net-knowhow", "tcpdump", "packet-capture", "wireshark"]
---


Packet capture and analysis with tcpdump. Essential tool for network debugging and security analysis.

---

## Installation

```bash
# Linux
sudo apt install tcpdump

# macOS (pre-installed)
# Or update with brew
brew install tcpdump

# Verify
tcpdump --version
```

---

## Basic Commands

```bash
# Capture on default interface
sudo tcpdump

# Capture on specific interface
sudo tcpdump -i eth0
sudo tcpdump -i wlan0

# List interfaces
tcpdump -D
ip link show

# Capture N packets
sudo tcpdump -c 10

# Verbose output
sudo tcpdump -v
sudo tcpdump -vv
sudo tcpdump -vvv

# Show packet contents (hex + ASCII)
sudo tcpdump -X

# Show packet contents (hex)
sudo tcpdump -xx

# Don't resolve hostnames (faster)
sudo tcpdump -n

# Don't resolve port names
sudo tcpdump -nn
```

---

## Filtering

### By Host

```bash
# Capture traffic to/from specific host
sudo tcpdump host 192.168.1.100

# Traffic from specific host
sudo tcpdump src host 192.168.1.100

# Traffic to specific host
sudo tcpdump dst host 192.168.1.100

# Multiple hosts
sudo tcpdump host 192.168.1.100 or host 192.168.1.101
```

### By Port

```bash
# Specific port
sudo tcpdump port 80
sudo tcpdump port 443

# Source port
sudo tcpdump src port 80

# Destination port
sudo tcpdump dst port 443

# Port range
sudo tcpdump portrange 8000-9000

# Multiple ports
sudo tcpdump port 80 or port 443
```

### By Protocol

```bash
# TCP only
sudo tcpdump tcp

# UDP only
sudo tcpdump udp

# ICMP (ping)
sudo tcpdump icmp

# ARP
sudo tcpdump arp

# IPv6
sudo tcpdump ip6
```

### By Network

```bash
# Specific network
sudo tcpdump net 192.168.1.0/24

# Source network
sudo tcpdump src net 192.168.1.0/24

# Destination network
sudo tcpdump dst net 10.0.0.0/8
```

---

## Complex Filters

```bash
# HTTP traffic
sudo tcpdump 'tcp port 80 or tcp port 443'

# SSH traffic
sudo tcpdump 'tcp port 22'

# DNS queries
sudo tcpdump 'udp port 53'

# Traffic between two hosts
sudo tcpdump 'host 192.168.1.100 and host 192.168.1.101'

# Traffic NOT from specific host
sudo tcpdump 'not host 192.168.1.100'

# HTTP GET requests
sudo tcpdump -A 'tcp port 80 and (tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x47455420)'

# SYN packets (connection attempts)
sudo tcpdump 'tcp[tcpflags] & (tcp-syn) != 0'

# RST packets
sudo tcpdump 'tcp[tcpflags] & (tcp-rst) != 0'
```

---

## Saving & Reading Captures

```bash
# Save to file
sudo tcpdump -w capture.pcap

# Save with rotation (100MB files)
sudo tcpdump -w capture.pcap -C 100

# Save N files then stop
sudo tcpdump -w capture.pcap -C 100 -W 5

# Read from file
tcpdump -r capture.pcap

# Read and filter
tcpdump -r capture.pcap 'port 80'

# Read and save filtered
tcpdump -r capture.pcap -w filtered.pcap 'port 80'
```

---

## Practical Examples

### Capture HTTP Traffic

```bash
# Capture HTTP requests and responses
sudo tcpdump -i eth0 -A 'tcp port 80'

# Save HTTP traffic
sudo tcpdump -i eth0 -w http.pcap 'tcp port 80'

# Show HTTP headers
sudo tcpdump -i eth0 -A -s 0 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)'
```

### Capture HTTPS Handshake

```bash
# Capture TLS/SSL handshake
sudo tcpdump -i eth0 'tcp port 443 and (tcp[((tcp[12:1] & 0xf0) >> 2)] = 0x16)'
```

### Capture DNS Queries

```bash
# All DNS traffic
sudo tcpdump -i eth0 'udp port 53'

# DNS queries only
sudo tcpdump -i eth0 'udp port 53 and udp[10] & 0x80 = 0'

# DNS responses only
sudo tcpdump -i eth0 'udp port 53 and udp[10] & 0x80 = 0x80'
```

### Capture Ping (ICMP)

```bash
# All ICMP
sudo tcpdump -i eth0 icmp

# Ping requests
sudo tcpdump -i eth0 'icmp[icmptype] = icmp-echo'

# Ping replies
sudo tcpdump -i eth0 'icmp[icmptype] = icmp-echoreply'
```

### Capture ARP

```bash
# ARP requests and replies
sudo tcpdump -i eth0 arp

# ARP requests only
sudo tcpdump -i eth0 'arp[6:2] = 1'

# ARP replies only
sudo tcpdump -i eth0 'arp[6:2] = 2'
```

### Monitor Specific Connection

```bash
# Monitor connection between two hosts
sudo tcpdump -i eth0 'host 192.168.1.100 and host 192.168.1.101'

# Monitor specific TCP connection
sudo tcpdump -i eth0 'host 192.168.1.100 and port 22'
```

---

## Advanced Usage

### Capture with Timestamps

```bash
# Absolute timestamps
sudo tcpdump -tttt

# Relative timestamps (since first packet)
sudo tcpdump -ttt

# Delta timestamps (since previous packet)
sudo tcpdump -tttt
```

### Capture Specific Packet Size

```bash
# Packets larger than 1000 bytes
sudo tcpdump 'greater 1000'

# Packets smaller than 100 bytes
sudo tcpdump 'less 100'
```

### Capture with Snaplen

```bash
# Capture only first 96 bytes of each packet (default)
sudo tcpdump -s 96

# Capture full packets
sudo tcpdump -s 0

# Capture only headers (68 bytes)
sudo tcpdump -s 68
```

### Capture to Multiple Files

```bash
# Rotate files every 100MB, keep 10 files
sudo tcpdump -i eth0 -w capture-%Y%m%d-%H%M%S.pcap -C 100 -W 10

# Rotate files every 60 seconds
sudo tcpdump -i eth0 -w capture.pcap -G 60 -W 10
```

---

## Integration with Wireshark

```bash
# Capture and pipe to Wireshark
sudo tcpdump -i eth0 -U -w - | wireshark -k -i -

# Or save and open
sudo tcpdump -i eth0 -w capture.pcap
wireshark capture.pcap
```

---

## Docker Container Capture

```bash
# Find container network interface
docker inspect <container_id> | grep NetworkMode

# Capture container traffic
sudo tcpdump -i docker0

# Capture specific container
sudo tcpdump -i docker0 'host <container_ip>'

# Enter container network namespace
sudo nsenter -t $(docker inspect -f '{{.State.Pid}}' <container>) -n tcpdump -i eth0
```

---

## Quick Reference

```bash
# Basic
sudo tcpdump -i eth0                    # Capture on eth0
sudo tcpdump -i eth0 -c 100             # Capture 100 packets
sudo tcpdump -i eth0 -nn                # No name resolution

# Filters
sudo tcpdump host 192.168.1.100        # Specific host
sudo tcpdump port 80                    # Specific port
sudo tcpdump tcp                        # TCP only
sudo tcpdump net 192.168.1.0/24        # Specific network

# Save/Read
sudo tcpdump -w file.pcap              # Save to file
tcpdump -r file.pcap                   # Read from file

# Display
sudo tcpdump -A                        # ASCII output
sudo tcpdump -X                        # Hex + ASCII
sudo tcpdump -v                        # Verbose
```

---