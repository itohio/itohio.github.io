---
title: "Hardware Enumeration"
date: 2024-12-12
draft: false
category: "hardware"
tags: ["hardware-knowhow", "enumeration", "lspci", "lsusb", "proc", "dev"]
---


List and enumerate hardware components: CPUs, GPUs, TPUs, and useful /proc and /dev commands.

---

## List CPUs

### Linux

```bash
# CPU information
lscpu

# Detailed CPU info
cat /proc/cpuinfo

# CPU model
cat /proc/cpuinfo | grep "model name" | uniq

# Number of cores
nproc

# CPU architecture
uname -m

# CPU frequency
cat /proc/cpuinfo | grep MHz

# Current CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Available governors
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors

# Set performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Windows

```powershell
# CPU information
Get-WmiObject Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed

# Detailed info
wmic cpu get name,numberofcores,numberoflogicalprocessors,maxclockspeed

# CPU architecture
echo %PROCESSOR_ARCHITECTURE%

# System info
systeminfo | findstr /C:"Processor"
```

---

## List GPUs

### Linux

```bash
# PCI devices (includes GPU)
lspci | grep -i vga
lspci | grep -i 3d
lspci | grep -i nvidia

# Detailed GPU info
lspci -v | grep -A 10 VGA

# NVIDIA GPU
nvidia-smi
nvidia-smi -L  # List GPUs
nvidia-smi -q  # Detailed query

# AMD GPU
lspci | grep -i amd
glxinfo | grep -i "opengl renderer"

# Intel GPU
lspci | grep -i intel.*graphics
intel_gpu_top  # Monitoring tool

# GPU memory
cat /sys/class/drm/card0/device/mem_info_vram_total
```

### Windows

```powershell
# GPU information
Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion

# Detailed info
wmic path win32_VideoController get name,adapterram,driverversion

# NVIDIA
nvidia-smi.exe

# DirectX info
dxdiag
```

---

## List TPUs/Accelerators

### Google Coral TPU

```bash
# List USB TPUs
lsusb | grep "Google"

# Check TPU device
ls /dev/apex_*

# TPU info (if driver installed)
cat /sys/class/apex/apex_0/device_type
```

### Intel Neural Compute Stick

```bash
# List USB devices
lsusb | grep "Movidius"

# OpenVINO detection
python3 -c "from openvino.inference_engine import IECore; ie = IECore(); print(ie.available_devices)"
```

### NVIDIA Jetson

```bash
# Jetson info
cat /etc/nv_tegra_release

# GPU info
tegrastats

# CUDA devices
nvidia-smi
```

---

## PCI Devices

```bash
# List all PCI devices
lspci

# Detailed view
lspci -v

# Tree view
lspci -t

# Numeric IDs
lspci -nn

# Specific device details
lspci -v -s 00:02.0

# Update PCI database
sudo update-pciids
```

---

## USB Devices

```bash
# List USB devices
lsusb

# Detailed view
lsusb -v

# Tree view
lsusb -t

# Specific device
lsusb -d 046d:c52b

# USB device details
cat /sys/kernel/debug/usb/devices

# Monitor USB events
udevadm monitor --subsystem-match=usb
```

---

## Block Devices (Disks)

```bash
# List block devices
lsblk

# Detailed disk info
sudo fdisk -l

# Disk usage
df -h

# Disk partitions
cat /proc/partitions

# SMART status
sudo smartctl -a /dev/sda

# Disk I/O stats
iostat

# Disk speed test
sudo hdparm -Tt /dev/sda
```

---

## Network Interfaces

```bash
# List network interfaces
ip link show

# Detailed info
ip addr show

# Interface statistics
ip -s link

# Wireless interfaces
iwconfig

# PCI network cards
lspci | grep -i network
lspci | grep -i ethernet

# USB network adapters
lsusb | grep -i network
```

---

## /proc Commands

### CPU & Memory

```bash
# CPU info
cat /proc/cpuinfo

# Memory info
cat /proc/meminfo

# Free memory
free -h

# Load average
cat /proc/loadavg
uptime

# Running processes
cat /proc/[PID]/status
cat /proc/[PID]/cmdline
```

### System Info

```bash
# Kernel version
cat /proc/version
uname -r

# System uptime
cat /proc/uptime

# Mounted filesystems
cat /proc/mounts

# Kernel modules
cat /proc/modules
lsmod

# Interrupts
cat /proc/interrupts

# I/O ports
cat /proc/ioports

# DMA channels
cat /proc/dma
```

### Process Info

```bash
# Process tree
pstree

# Process info
cat /proc/[PID]/status
cat /proc/[PID]/stat
cat /proc/[PID]/cmdline
cat /proc/[PID]/environ
cat /proc/[PID]/maps     # Memory maps
cat /proc/[PID]/fd/      # Open file descriptors

# All processes
ls /proc/ | grep -E '^[0-9]+$'
```

---

## /dev Commands

### Device Files

```bash
# List all devices
ls -l /dev/

# Block devices
ls -l /dev/sd*   # SCSI/SATA disks
ls -l /dev/nvme* # NVMe disks
ls -l /dev/mmcblk* # SD cards

# Character devices
ls -l /dev/tty*  # Terminals
ls -l /dev/input/* # Input devices

# Special devices
ls -l /dev/null
ls -l /dev/zero
ls -l /dev/random
ls -l /dev/urandom
```

### Device Information

```bash
# Device major/minor numbers
ls -l /dev/sda

# Device by UUID
ls -l /dev/disk/by-uuid/

# Device by label
ls -l /dev/disk/by-label/

# Device by path
ls -l /dev/disk/by-path/

# Device by ID
ls -l /dev/disk/by-id/
```

---

## Hardware Monitoring

```bash
# Temperature sensors
sensors

# Detect sensors
sudo sensors-detect

# Watch temperatures
watch -n 1 sensors

# Fan speeds
sensors | grep fan

# Voltage
sensors | grep in

# Power consumption
sudo powertop

# Battery status
cat /sys/class/power_supply/BAT0/capacity
cat /sys/class/power_supply/BAT0/status
upower -i /org/freedesktop/UPower/devices/battery_BAT0
```

---

## DMI/SMBIOS Information

```bash
# System information
sudo dmidecode

# BIOS info
sudo dmidecode -t bios

# System info
sudo dmidecode -t system

# Motherboard info
sudo dmidecode -t baseboard

# CPU info
sudo dmidecode -t processor

# Memory info
sudo dmidecode -t memory

# All info types
sudo dmidecode --type 0  # BIOS
sudo dmidecode --type 1  # System
sudo dmidecode --type 2  # Baseboard
sudo dmidecode --type 4  # Processor
sudo dmidecode --type 17 # Memory Device
```

---

## Comprehensive Hardware Report

```bash
# Generate full hardware report
sudo lshw > hardware-report.txt

# HTML report
sudo lshw -html > hardware-report.html

# Short format
sudo lshw -short

# Specific class
sudo lshw -C network
sudo lshw -C storage
sudo lshw -C display
sudo lshw -C processor
sudo lshw -C memory
```

---

## Windows Hardware Enumeration

```powershell
# System information
systeminfo

# Hardware list
Get-WmiObject Win32_ComputerSystem

# CPU
Get-WmiObject Win32_Processor

# Memory
Get-WmiObject Win32_PhysicalMemory

# Disk
Get-WmiObject Win32_DiskDrive

# GPU
Get-WmiObject Win32_VideoController

# Network adapters
Get-WmiObject Win32_NetworkAdapter

# USB devices
Get-WmiObject Win32_USBControllerDevice

# PCI devices
Get-WmiObject Win32_PnPEntity

# Device Manager (GUI)
devmgmt.msc
```

---

## Kernel Ring Buffer

```bash
# View kernel messages
dmesg

# Follow kernel messages
dmesg -w

# Clear ring buffer
sudo dmesg -c

# Filter by facility
dmesg --facility=kern
dmesg --facility=user

# Filter by level
dmesg --level=err
dmesg --level=warn

# Human-readable timestamps
dmesg -T

# Search for device
dmesg | grep -i usb
dmesg | grep -i pci
dmesg | grep -i sda
```

---