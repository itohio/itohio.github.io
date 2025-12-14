---
title: "Cisco IOS Commands"
date: 2024-12-12
draft: false
description: "Cisco switch and router CLI commands"
tags: ["cisco", "ios", "networking", "switches", "routers"]
---



## Basic Navigation

```cisco
# Enter privileged EXEC mode
enable

# Enter global configuration mode
configure terminal

# Exit to previous mode
exit

# Exit to privileged EXEC
end

# Show current mode
# (indicated by prompt: > user, # privileged, (config)# global config)
```

## Configuration Modes

```cisco
# Global configuration
Router(config)#

# Interface configuration
Router(config)# interface gigabitethernet 0/1
Router(config-if)#

# Line configuration (console, vty)
Router(config)# line console 0
Router(config-line)#

# Router configuration
Router(config)# router ospf 1
Router(config-router)#

# VLAN configuration
Switch(config)# vlan 10
Switch(config-vlan)#
```

## Show Commands

```cisco
# Running configuration
show running-config
show run

# Startup configuration
show startup-config
show start

# Interfaces
show interfaces
show ip interface brief
show interfaces status
show interfaces gigabitethernet 0/1

# IP routing
show ip route
show ip protocols

# VLANs
show vlan
show vlan brief
show vlan id 10

# MAC address table
show mac address-table
show mac address-table dynamic

# ARP table
show arp
show ip arp

# Version and hardware
show version
show inventory

# Logs
show logging
show log

# CDP (Cisco Discovery Protocol)
show cdp neighbors
show cdp neighbors detail

# LLDP (Link Layer Discovery Protocol)
show lldp neighbors
show lldp neighbors detail

# Spanning Tree
show spanning-tree
show spanning-tree vlan 10
```

## Interface Configuration

```cisco
# Enter interface
interface gigabitethernet 0/1

# Set IP address
ip address 192.168.1.1 255.255.255.0

# Enable interface
no shutdown

# Disable interface
shutdown

# Description
description "Uplink to Core Switch"

# Speed and duplex
speed 1000
duplex full
speed auto
duplex auto

# Access port (single VLAN)
switchport mode access
switchport access vlan 10

# Trunk port (multiple VLANs)
switchport mode trunk
switchport trunk allowed vlan 10,20,30
switchport trunk native vlan 1

# Port security
switchport port-security
switchport port-security maximum 2
switchport port-security mac-address sticky
switchport port-security violation restrict
```

## VLAN Configuration

```cisco
# Create VLAN
vlan 10
name Sales
exit

# Delete VLAN
no vlan 10

# Assign interface to VLAN
interface gigabitethernet 0/5
switchport mode access
switchport access vlan 10

# Inter-VLAN routing (router-on-a-stick)
interface gigabitethernet 0/1.10
encapsulation dot1Q 10
ip address 192.168.10.1 255.255.255.0
```

## Routing

### Static Routes

```cisco
# IPv4 static route
ip route 192.168.2.0 255.255.255.0 192.168.1.1

# Default route
ip route 0.0.0.0 0.0.0.0 192.168.1.1

# IPv6 static route
ipv6 route 2001:db8::/32 2001:db8::1
```

### OSPF

```cisco
# Enable OSPF
router ospf 1
network 192.168.1.0 0.0.0.255 area 0
network 10.0.0.0 0.255.255.255 area 1

# Set router ID
router-id 1.1.1.1

# Passive interface
passive-interface gigabitethernet 0/1

# Show OSPF
show ip ospf
show ip ospf neighbor
show ip ospf database
```

### EIGRP

```cisco
# Enable EIGRP
router eigrp 100
network 192.168.1.0
network 10.0.0.0 0.255.255.255

# Show EIGRP
show ip eigrp neighbors
show ip eigrp topology
```

## Security

### Passwords

```cisco
# Enable secret (encrypted)
enable secret MySecretPassword

# Console password
line console 0
password MyConsolePassword
login

# VTY (Telnet/SSH) password
line vty 0 4
password MyVTYPassword
login

# Encrypt passwords
service password-encryption
```

### SSH Configuration

```cisco
# Set hostname and domain
hostname Router1
ip domain-name example.com

# Generate RSA keys
crypto key generate rsa
# (choose key size, e.g., 2048)

# Configure VTY for SSH
line vty 0 4
transport input ssh
login local

# Create user
username admin privilege 15 secret AdminPassword

# SSH version
ip ssh version 2
```

### Access Control Lists (ACLs)

```cisco
# Standard ACL
access-list 10 permit 192.168.1.0 0.0.0.255
access-list 10 deny any

# Extended ACL
access-list 100 permit tcp 192.168.1.0 0.0.0.255 any eq 80
access-list 100 permit tcp 192.168.1.0 0.0.0.255 any eq 443
access-list 100 deny ip any any

# Named ACL
ip access-list extended WEB-TRAFFIC
permit tcp 192.168.1.0 0.0.0.255 any eq 80
permit tcp 192.168.1.0 0.0.0.255 any eq 443
deny ip any any

# Apply ACL to interface
interface gigabitethernet 0/1
ip access-group 100 in

# Show ACLs
show access-lists
show ip access-lists
```

## Spanning Tree

```cisco
# Set spanning tree mode
spanning-tree mode rapid-pvst

# Set priority (lower = preferred root)
spanning-tree vlan 10 priority 4096

# PortFast (access ports only)
interface gigabitethernet 0/5
spanning-tree portfast

# BPDU Guard
spanning-tree portfast bpduguard default

# Show spanning tree
show spanning-tree
show spanning-tree summary
```

## Troubleshooting

```cisco
# Ping
ping 192.168.1.1
ping 192.168.1.1 repeat 100

# Traceroute
traceroute 8.8.8.8

# Debug (use with caution!)
debug ip icmp
debug ip routing
undebug all  # Disable all debugging

# Clear commands
clear mac address-table dynamic
clear arp-cache
clear ip route *

# Reload
reload
reload in 10  # Reload in 10 minutes
reload cancel
```

## Saving Configuration

```cisco
# Save running config to startup config
copy running-config startup-config
write memory
wr

# Backup config to TFTP
copy running-config tftp:
# (enter TFTP server IP and filename)

# Restore config from TFTP
copy tftp: running-config
```

## Common Tasks

### Reset to Factory Defaults

```cisco
# Erase startup config
erase startup-config

# Delete VLAN database (switches)
delete flash:vlan.dat

# Reload
reload
```

### Password Recovery

```cisco
# 1. Interrupt boot process (Ctrl+Break)
# 2. Change config register
confreg 0x2142
reset

# 3. After boot, copy startup to running
copy startup-config running-config

# 4. Change password
enable secret NewPassword

# 5. Restore config register
config-register 0x2102

# 6. Save and reload
copy running-config startup-config
reload
```

## Further Reading

- [Cisco IOS Command Reference](https://www.cisco.com/c/en/us/support/ios-nx-os-software/ios-15-4m-t/products-command-reference-list.html)
- [Cisco Networking Academy](https://www.netacad.com/)
- [Packet Tracer](https://www.netacad.com/courses/packet-tracer) - Network simulation tool

