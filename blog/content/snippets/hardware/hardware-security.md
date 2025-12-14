---
title: "Hardware Security"
date: 2024-12-12
draft: false
category: "hardware"
tags: ["hardware-knowhow", "security", "tpm", "secure-boot", "encryption"]
---


Hardware security features: TPM, Secure Boot, hardware encryption, and security best practices.

---

## TPM (Trusted Platform Module)

### Check TPM Status (Linux)

```bash
# Check if TPM exists
ls /dev/tpm*

# TPM version
cat /sys/class/tpm/tpm0/tpm_version_major

# TPM info
sudo tpm2_getcap properties-fixed

# Install TPM tools
sudo apt install tpm2-tools

# List PCR values
sudo tpm2_pcrread
```

### Check TPM (Windows)

```powershell
# TPM status
Get-Tpm

# Detailed TPM info
Get-WmiObject -Namespace "Root\CIMv2\Security\MicrosoftTpm" -Class Win32_Tpm

# TPM version
(Get-Tpm).TpmPresent
(Get-Tpm).TpmReady
(Get-Tpm).TpmEnabled

# Open TPM Management
tpm.msc
```

---

## Secure Boot

### Check Secure Boot Status (Linux)

```bash
# Check if Secure Boot is enabled
mokutil --sb-state

# Check Secure Boot from EFI
cat /sys/firmware/efi/efivars/SecureBoot-*

# Install mokutil
sudo apt install mokutil

# List enrolled keys
mokutil --list-enrolled

# Check if in setup mode
mokutil --sb-state
```

### Check Secure Boot (Windows)

```powershell
# Secure Boot status
Confirm-SecureBootUEFI

# Detailed info
Get-SecureBootPolicy

# System information
msinfo32
# Look for "Secure Boot State"
```

---

## Hardware Encryption

### Check Disk Encryption Support

```bash
# Check if drive supports hardware encryption (SED)
sudo hdparm -I /dev/sda | grep -i "security"

# Check for AES-NI (CPU encryption acceleration)
grep -o 'aes' /proc/cpuinfo

# LUKS encryption status
sudo cryptsetup status /dev/mapper/encrypted

# List encrypted devices
lsblk -o NAME,FSTYPE,SIZE,MOUNTPOINT,LABEL
```

### Enable LUKS Encryption

```bash
# Encrypt partition
sudo cryptsetup luksFormat /dev/sdb1

# Open encrypted partition
sudo cryptsetup luksOpen /dev/sdb1 encrypted

# Create filesystem
sudo mkfs.ext4 /dev/mapper/encrypted

# Mount
sudo mount /dev/mapper/encrypted /mnt/encrypted
```

---

## CPU Security Features

### Check CPU Security Features

```bash
# Intel features
grep -E 'aes|sgx|txt|smx' /proc/cpuinfo

# AMD features
grep -E 'aes|sev|sme' /proc/cpuinfo

# Spectre/Meltdown mitigations
cat /sys/devices/system/cpu/vulnerabilities/*

# Or detailed view
grep . /sys/devices/system/cpu/vulnerabilities/*
```

### CPU Vulnerabilities

```bash
# Check all vulnerabilities
ls /sys/devices/system/cpu/vulnerabilities/

# Spectre v1
cat /sys/devices/system/cpu/vulnerabilities/spectre_v1

# Spectre v2
cat /sys/devices/system/cpu/vulnerabilities/spectre_v2

# Meltdown
cat /sys/devices/system/cpu/vulnerabilities/meltdown

# Check mitigations
dmesg | grep -i "mitigation"
```

---

## IOMMU (VT-d / AMD-Vi)

### Enable IOMMU

```bash
# Check if IOMMU is enabled
dmesg | grep -i iommu

# Enable in GRUB
sudo nano /etc/default/grub

# For Intel:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on iommu=pt"

# For AMD:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=on iommu=pt"

# Update GRUB
sudo update-grub

# Reboot
sudo reboot

# Verify
dmesg | grep -i "IOMMU enabled"
```

---

## Hardware Security Keys

### YubiKey

```bash
# Install tools
sudo apt install yubikey-manager

# List YubiKeys
ykman list

# YubiKey info
ykman info

# Configure FIDO2
ykman fido info

# Configure OTP
ykman otp info

# Configure PIV (smart card)
ykman piv info
```

### U2F/FIDO2

```bash
# Install libfido2
sudo apt install libfido2-dev fido2-tools

# List FIDO devices
fido2-token -L

# Get device info
fido2-token -I /dev/hidraw0

# Register credential
fido2-token -M -i challenge.txt /dev/hidraw0
```

---

## Firmware Security

### Check Firmware Updates

```bash
# fwupd (Linux firmware updater)
sudo apt install fwupd

# Check for updates
fwupdmgr get-updates

# Update firmware
sudo fwupdmgr update

# List devices
fwupdmgr get-devices

# Check security attributes
fwupdmgr security
```

### BIOS/UEFI Security

```bash
# Check UEFI variables
efibootmgr -v

# List EFI variables
ls /sys/firmware/efi/efivars/

# Check boot order
efibootmgr

# Secure Boot keys
ls /sys/firmware/efi/efivars/ | grep -i "PK\|KEK\|db"
```

---

## Memory Protection

### Check Memory Encryption

```bash
# AMD SME/SEV
dmesg | grep -i sme
dmesg | grep -i sev

# Check if enabled
cat /sys/devices/system/cpu/sme/active

# Intel TME
dmesg | grep -i tme
```

### Check ASLR

```bash
# ASLR status (0=off, 1=conservative, 2=full)
cat /proc/sys/kernel/randomize_va_space

# Enable full ASLR
echo 2 | sudo tee /proc/sys/kernel/randomize_va_space

# Make permanent
echo "kernel.randomize_va_space = 2" | sudo tee -a /etc/sysctl.conf
```

---

## Hardware Monitoring for Security

### Check for Hardware Changes

```bash
# Generate hardware baseline
sudo lshw > baseline.txt

# Compare later
sudo lshw > current.txt
diff baseline.txt current.txt

# Monitor PCI devices
watch -n 1 'lspci'

# Monitor USB devices
watch -n 1 'lsusb'

# USB device events
udevadm monitor --subsystem-match=usb
```

### Detect Hardware Keyloggers

```bash
# List USB input devices
ls -l /dev/input/by-id/

# Monitor input events
sudo evtest

# Check for unexpected USB devices
lsusb -t
```

---

## Secure Erase

### Secure Disk Erase

```bash
# Check if drive supports secure erase
sudo hdparm -I /dev/sda | grep -i "erase"

# Set password
sudo hdparm --user-master u --security-set-pass password /dev/sda

# Secure erase
sudo hdparm --user-master u --security-erase password /dev/sda

# Or use shred
sudo shred -vfz -n 3 /dev/sda

# Or dd with random data
sudo dd if=/dev/urandom of=/dev/sda bs=1M status=progress
```

---

## Windows BitLocker

```powershell
# Check BitLocker status
Get-BitLockerVolume

# Enable BitLocker
Enable-BitLocker -MountPoint "C:" -EncryptionMethod XtsAes256 -UsedSpaceOnly -TpmProtector

# Backup recovery key
BackupToAAD-BitLockerKeyProtector -MountPoint "C:" -KeyProtectorId "{ID}"

# Check TPM
Get-Tpm

# Suspend BitLocker (for updates)
Suspend-BitLocker -MountPoint "C:"

# Resume BitLocker
Resume-BitLocker -MountPoint "C:"
```

---

## Security Checklist

### BIOS/UEFI

- [ ] Enable Secure Boot
- [ ] Set BIOS/UEFI password
- [ ] Disable unused ports (USB, Thunderbolt)
- [ ] Enable TPM
- [ ] Disable boot from USB/CD (or set password)
- [ ] Enable VT-d/AMD-Vi (IOMMU)

### Operating System

- [ ] Enable full disk encryption
- [ ] Enable ASLR
- [ ] Keep firmware updated
- [ ] Enable firewall
- [ ] Disable unnecessary services
- [ ] Use hardware security key (YubiKey)

### Physical Security

- [ ] Lock computer when away
- [ ] Use Kensington lock
- [ ] Disable DMA ports (Thunderbolt, FireWire)
- [ ] Monitor for hardware keyloggers
- [ ] Secure boot order

---