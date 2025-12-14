---
title: "WireGuard VPN Setup"
date: 2024-12-12
draft: false
category: "net"
tags: ["net-knowhow", "wireguard", "vpn", "tunneling", "security"]
---


WireGuard VPN setup with port forwarding and tunneling. Modern, fast, and secure VPN solution.

---

## Installation

```bash
# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install wireguard

# Linux (Fedora/RHEL)
sudo dnf install wireguard-tools

# macOS
brew install wireguard-tools

# Windows
# Download from https://www.wireguard.com/install/
```

---

## Server Setup

### Generate Keys

```bash
# Generate private key
wg genkey | sudo tee /etc/wireguard/server_private.key
sudo chmod 600 /etc/wireguard/server_private.key

# Generate public key from private key
sudo cat /etc/wireguard/server_private.key | wg pubkey | sudo tee /etc/wireguard/server_public.key
```

### Server Configuration

```bash
# Create server config
sudo nano /etc/wireguard/wg0.conf
```

**`/etc/wireguard/wg0.conf`:**

```ini
[Interface]
# Server private key
PrivateKey = <SERVER_PRIVATE_KEY>

# Server VPN IP
Address = 10.0.0.1/24

# Listen port
ListenPort = 51820

# Post-up commands (enable forwarding and NAT)
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT
PostUp = iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostUp = ip6tables -A FORWARD -i wg0 -j ACCEPT
PostUp = ip6tables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# Post-down commands (cleanup)
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT
PostDown = iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE
PostDown = ip6tables -D FORWARD -i wg0 -j ACCEPT
PostDown = ip6tables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

# Client 1
[Peer]
PublicKey = <CLIENT1_PUBLIC_KEY>
AllowedIPs = 10.0.0.2/32

# Client 2
[Peer]
PublicKey = <CLIENT2_PUBLIC_KEY>
AllowedIPs = 10.0.0.3/32
```

### Enable IP Forwarding

```bash
# Enable IPv4 forwarding
echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf
echo "net.ipv6.conf.all.forwarding=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Verify
sysctl net.ipv4.ip_forward
```

### Start Server

```bash
# Start WireGuard
sudo wg-quick up wg0

# Stop WireGuard
sudo wg-quick down wg0

# Enable on boot
sudo systemctl enable wg-quick@wg0

# Check status
sudo wg show
```

---

## Client Setup

### Generate Client Keys

```bash
# Generate client private key
wg genkey | tee client_private.key
chmod 600 client_private.key

# Generate client public key
cat client_private.key | wg pubkey > client_public.key
```

### Client Configuration

**`/etc/wireguard/wg0.conf`:**

```ini
[Interface]
# Client private key
PrivateKey = <CLIENT_PRIVATE_KEY>

# Client VPN IP
Address = 10.0.0.2/24

# DNS (optional)
DNS = 1.1.1.1, 8.8.8.8

[Peer]
# Server public key
PublicKey = <SERVER_PUBLIC_KEY>

# Server endpoint (public IP and port)
Endpoint = server.example.com:51820

# Route all traffic through VPN
AllowedIPs = 0.0.0.0/0, ::/0

# Or only route specific networks
# AllowedIPs = 10.0.0.0/24, 192.168.1.0/24

# Keep connection alive (NAT traversal)
PersistentKeepalive = 25
```

### Connect Client

```bash
# Start VPN
sudo wg-quick up wg0

# Stop VPN
sudo wg-quick down wg0

# Check status
sudo wg show
```

---

## Port Forwarding

### Forward Specific Port

```bash
# On server, forward port 8080 to client
sudo iptables -t nat -A PREROUTING -p tcp --dport 8080 -j DNAT --to-destination 10.0.0.2:8080
sudo iptables -A FORWARD -p tcp -d 10.0.0.2 --dport 8080 -j ACCEPT

# Save rules
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

### Forward Port Range

```bash
# Forward ports 8000-9000
sudo iptables -t nat -A PREROUTING -p tcp --dport 8000:9000 -j DNAT --to-destination 10.0.0.2
sudo iptables -A FORWARD -p tcp -d 10.0.0.2 --dport 8000:9000 -j ACCEPT
```

### Add to WireGuard Config

```ini
[Interface]
PrivateKey = <SERVER_PRIVATE_KEY>
Address = 10.0.0.1/24
ListenPort = 51820

# Port forwarding rules
PostUp = iptables -t nat -A PREROUTING -p tcp --dport 8080 -j DNAT --to-destination 10.0.0.2:8080
PostUp = iptables -A FORWARD -p tcp -d 10.0.0.2 --dport 8080 -j ACCEPT

PostDown = iptables -t nat -D PREROUTING -p tcp --dport 8080 -j DNAT --to-destination 10.0.0.2:8080
PostDown = iptables -D FORWARD -p tcp -d 10.0.0.2 --dport 8080 -j ACCEPT
```

---

## Site-to-Site VPN

### Server A Configuration

```ini
[Interface]
PrivateKey = <SERVER_A_PRIVATE_KEY>
Address = 10.0.0.1/24
ListenPort = 51820

PostUp = iptables -A FORWARD -i wg0 -j ACCEPT
PostUp = iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

PostDown = iptables -D FORWARD -i wg0 -j ACCEPT
PostDown = iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
PublicKey = <SERVER_B_PUBLIC_KEY>
Endpoint = server-b.example.com:51820
AllowedIPs = 10.0.0.2/32, 192.168.2.0/24  # Server B VPN IP + LAN
PersistentKeepalive = 25
```

### Server B Configuration

```ini
[Interface]
PrivateKey = <SERVER_B_PRIVATE_KEY>
Address = 10.0.0.2/24
ListenPort = 51820

PostUp = iptables -A FORWARD -i wg0 -j ACCEPT
PostUp = iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

PostDown = iptables -D FORWARD -i wg0 -j ACCEPT
PostDown = iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
PublicKey = <SERVER_A_PUBLIC_KEY>
Endpoint = server-a.example.com:51820
AllowedIPs = 10.0.0.1/32, 192.168.1.0/24  # Server A VPN IP + LAN
PersistentKeepalive = 25
```

---

## Docker Setup

```yaml
version: '3.8'

services:
  wireguard:
    image: linuxserver/wireguard
    container_name: wireguard
    cap_add:
      - NET_ADMIN
      - SYS_MODULE
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
      - SERVERURL=vpn.example.com
      - SERVERPORT=51820
      - PEERS=3  # Number of client configs to generate
      - PEERDNS=auto
      - INTERNAL_SUBNET=10.13.13.0/24
    volumes:
      - ./config:/config
      - /lib/modules:/lib/modules
    ports:
      - 51820:51820/udp
    sysctls:
      - net.ipv4.conf.all.src_valid_mark=1
    restart: unless-stopped
```

```bash
# Start
docker-compose up -d

# View client configs (QR codes)
docker exec -it wireguard /app/show-peer 1
```

---

## Mobile Setup

### Generate QR Code

```bash
# Install qrencode
sudo apt install qrencode

# Generate QR code for mobile
qrencode -t ansiutf8 < /etc/wireguard/mobile.conf

# Or save as image
qrencode -o mobile.png < /etc/wireguard/mobile.conf
```

**Mobile Client Config:**

```ini
[Interface]
PrivateKey = <MOBILE_PRIVATE_KEY>
Address = 10.0.0.10/24
DNS = 1.1.1.1

[Peer]
PublicKey = <SERVER_PUBLIC_KEY>
Endpoint = server.example.com:51820
AllowedIPs = 0.0.0.0/0, ::/0
PersistentKeepalive = 25
```

---

## Troubleshooting

### Check Status

```bash
# Show interface status
sudo wg show

# Show with transfer stats
sudo wg show wg0

# Show specific peer
sudo wg show wg0 peers
```

### Test Connectivity

```bash
# Ping VPN server from client
ping 10.0.0.1

# Ping client from server
ping 10.0.0.2

# Check routing
ip route show

# Check if traffic is going through VPN
curl ifconfig.me
```

### Debug Connection

```bash
# Check firewall
sudo ufw status
sudo iptables -L -n -v

# Check if port is open
sudo ss -tulpn | grep 51820

# Test UDP connectivity
nc -u -v server.example.com 51820

# Enable debug logging
sudo wg set wg0 private-key /etc/wireguard/server_private.key listen-port 51820
```

### Common Issues

```bash
# Issue: Connection timeout
# Solution: Check firewall allows UDP port 51820
sudo ufw allow 51820/udp

# Issue: No internet after connecting
# Solution: Check DNS and routing
# Verify AllowedIPs in config

# Issue: Peer not connecting
# Solution: Check keys match
# Verify endpoint is correct
# Check PersistentKeepalive is set

# Issue: IP forwarding not working
# Solution: Enable IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1
```

---

## Security Best Practices

```
✅ Use strong keys (automatically generated)
✅ Limit AllowedIPs to only what's needed
✅ Use firewall rules to restrict access
✅ Rotate keys periodically
✅ Use different keys for each client
✅ Enable PersistentKeepalive for NAT traversal
✅ Monitor connection logs
✅ Use DNS over VPN
✅ Disable unused peers
✅ Keep WireGuard updated
```

---

## Quick Reference

```bash
# Server
wg genkey | tee private.key | wg pubkey > public.key
sudo wg-quick up wg0
sudo wg show

# Client
sudo wg-quick up wg0
sudo wg show
ping 10.0.0.1

# Management
sudo systemctl enable wg-quick@wg0
sudo systemctl start wg-quick@wg0
sudo systemctl status wg-quick@wg0

# Firewall
sudo ufw allow 51820/udp
sudo iptables -A INPUT -p udp --dport 51820 -j ACCEPT
```

---