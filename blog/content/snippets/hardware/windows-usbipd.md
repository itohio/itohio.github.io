---
title: "Windows USB/IP (usbipd)"
date: 2024-12-12
draft: false
category: "hardware"
tags: ["hardware-knowhow", "windows", "usb", "wsl", "usbipd"]
---


Share USB devices from Windows to WSL2 or remote machines using usbipd-win.

---

## Installation

### Windows (Host)

```powershell
# Install usbipd-win
winget install --interactive --exact dorssel.usbipd-win

# Or download from GitHub
# https://github.com/dorssel/usbipd-win/releases

# Verify installation
usbipd --version
```

### WSL2 (Client)

```bash
# Update WSL
wsl --update

# Inside WSL, install USB/IP tools
sudo apt install linux-tools-generic hwdata
sudo update-alternatives --install /usr/local/bin/usbip usbip /usr/lib/linux-tools/*-generic/usbip 20
```

---

## List USB Devices

```powershell
# List all USB devices
usbipd list

# Example output:
# BUSID  VID:PID    DEVICE                                            STATE
# 1-1    046d:c52b  Logitech USB Input Device                         Not shared
# 1-2    0bda:0129  Realtek USB 2.0 Card Reader                       Not shared
# 2-3    8087:0026  Intel(R) Wireless Bluetooth(R)                    Not shared
```

---

## Share USB Device to WSL

### Bind Device

```powershell
# Bind device (makes it shareable)
usbipd bind --busid 1-1

# Verify
usbipd list
# STATE should show "Shared"
```

### Attach to WSL

```powershell
# Attach to default WSL distribution
usbipd attach --wsl --busid 1-1

# Attach to specific distribution
usbipd attach --wsl --distribution Ubuntu --busid 1-1

# Auto-attach (attach automatically when device is connected)
usbipd bind --busid 1-1 --auto-attach
```

### Verify in WSL

```bash
# Inside WSL, check if device is attached
lsusb

# Check device details
lsusb -v -d 046d:c52b

# Check kernel messages
dmesg | tail

# List USB devices
ls -l /dev/bus/usb/*/*
```

---

## Detach USB Device

```powershell
# Detach from WSL
usbipd detach --busid 1-1

# Unbind (stop sharing)
usbipd unbind --busid 1-1
```

---

## Common Use Cases

### Serial Device (Arduino, ESP32)

```powershell
# Windows: Find device
usbipd list | findstr "Serial"

# Bind and attach
usbipd bind --busid 1-4
usbipd attach --wsl --busid 1-4
```

```bash
# WSL: Access serial device
ls -l /dev/ttyUSB*
ls -l /dev/ttyACM*

# Set permissions
sudo chmod 666 /dev/ttyUSB0

# Or add user to dialout group
sudo usermod -a -G dialout $USER

# Test with screen
screen /dev/ttyUSB0 115200

# Or minicom
sudo apt install minicom
minicom -D /dev/ttyUSB0 -b 115200
```

### USB Storage

```powershell
# Attach USB drive
usbipd attach --wsl --busid 2-1
```

```bash
# WSL: Mount USB drive
sudo mkdir /mnt/usb
sudo mount /dev/sdb1 /mnt/usb

# Unmount
sudo umount /mnt/usb
```

### Webcam

```powershell
# Attach webcam
usbipd attach --wsl --busid 1-3
```

```bash
# WSL: Check video devices
ls -l /dev/video*

# Test with v4l2
sudo apt install v4l-utils
v4l2-ctl --list-devices

# Capture image
ffmpeg -f v4l2 -i /dev/video0 -frames 1 capture.jpg
```

### Hardware Security Key (YubiKey)

```powershell
# Attach YubiKey
usbipd attach --wsl --busid 1-5
```

```bash
# WSL: Verify YubiKey
lsusb | grep Yubico

# Install tools
sudo apt install yubikey-manager

# Test
ykman info
```

---

## Share USB Over Network

### Server (Windows)

```powershell
# Bind device for network sharing
usbipd bind --busid 1-1

# Start server (listens on port 3240)
usbipd server
```

### Client (Linux)

```bash
# Install usbip
sudo apt install linux-tools-generic

# List remote devices
usbip list --remote=192.168.1.100

# Attach remote device
sudo usbip attach --remote=192.168.1.100 --busid=1-1

# List attached devices
usbip port

# Detach
sudo usbip detach --port=00
```

---

## Automation

### PowerShell Script

```powershell
# auto-attach-usb.ps1
$BUSID = "1-1"

# Wait for device
while ($true) {
    $device = usbipd list | Select-String -Pattern $BUSID
    if ($device -match "Not attached") {
        Write-Host "Attaching device $BUSID..."
        usbipd attach --wsl --busid $BUSID
        break
    }
    Start-Sleep -Seconds 2
}
```

### WSL Startup Script

```bash
# ~/.bashrc or ~/.zshrc

# Auto-mount USB drive
if [ -b /dev/sdb1 ] && ! mountpoint -q /mnt/usb; then
    sudo mkdir -p /mnt/usb
    sudo mount /dev/sdb1 /mnt/usb
fi

# Auto-set serial permissions
if [ -e /dev/ttyUSB0 ]; then
    sudo chmod 666 /dev/ttyUSB0
fi
```

---

## Troubleshooting

### Device Not Showing in WSL

```powershell
# Windows: Check if attached
usbipd list

# Detach and reattach
usbipd detach --busid 1-1
usbipd attach --wsl --busid 1-1
```

```bash
# WSL: Reload USB modules
sudo modprobe -r usbip_core
sudo modprobe usbip_core

# Check kernel messages
dmesg | grep -i usb
```

### Permission Denied

```bash
# WSL: Add user to groups
sudo usermod -a -G dialout,plugdev $USER

# Logout and login again
exit
wsl

# Or set permissions directly
sudo chmod 666 /dev/ttyUSB0
```

### Device Busy

```powershell
# Windows: Check if device is in use
# Close applications using the device

# Force detach
usbipd detach --busid 1-1 --force
```

### WSL Kernel Doesn't Support USB

```bash
# Update WSL kernel
wsl --update

# Check kernel version (should be 5.10.60.1+)
uname -r

# If old kernel, update Windows and WSL
```

---

## Persistent Configuration

### Windows Task Scheduler

```powershell
# Create scheduled task to auto-attach on boot
$action = New-ScheduledTaskAction -Execute "usbipd" -Argument "attach --wsl --busid 1-1"
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName "USB Auto-Attach" -Action $action -Trigger $trigger -Principal $principal
```

### WSL systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/usb-setup.service

[Unit]
Description=USB Device Setup
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/usb-setup.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target

# Create script
sudo nano /usr/local/bin/usb-setup.sh

#!/bin/bash
# Wait for USB device
sleep 5

# Set permissions
if [ -e /dev/ttyUSB0 ]; then
    chmod 666 /dev/ttyUSB0
fi

# Make executable
sudo chmod +x /usr/local/bin/usb-setup.sh

# Enable service
sudo systemctl enable usb-setup
```

---

## Security Considerations

1. **Bind only trusted devices** - Malicious USB devices can compromise system
2. **Use firewall** - Block port 3240 if not sharing over network
3. **Verify device IDs** - Check VID:PID before attaching
4. **Limit network sharing** - Only share to trusted networks
5. **Monitor attached devices** - Regularly check `usbipd list`

---