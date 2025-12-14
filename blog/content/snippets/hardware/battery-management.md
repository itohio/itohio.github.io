---
title: "Laptop Battery Management"
date: 2024-12-12
draft: false
category: "hardware"
tags: ["hardware-knowhow", "battery", "power-management", "laptop"]
---


Configure laptop battery charging thresholds to extend battery lifespan.

---

## Why Limit Charging?

Keeping battery at 80% instead of 100% significantly extends battery lifespan:
- **100% charge**: ~300-500 cycles
- **80% charge**: ~1000-1500 cycles

---

## Linux (ThinkPad)

### Using TLP

```bash
# Install TLP
sudo apt install tlp tlp-rdw

# Configure charging thresholds
sudo tee /etc/tlp.conf <<EOF
# Battery charge thresholds (ThinkPad only)
START_CHARGE_THRESH_BAT0=75
STOP_CHARGE_THRESH_BAT0=80

START_CHARGE_THRESH_BAT1=75
STOP_CHARGE_THRESH_BAT1=80
EOF

# Start TLP
sudo tlp start

# Check status
sudo tlp-stat -b
```

### Manual Configuration

```bash
# Set charge threshold (ThinkPad)
echo 80 | sudo tee /sys/class/power_supply/BAT0/charge_control_end_threshold
echo 75 | sudo tee /sys/class/power_supply/BAT0/charge_control_start_threshold

# Make permanent (add to /etc/rc.local or systemd)
sudo tee /etc/systemd/system/battery-threshold.service <<EOF
[Unit]
Description=Set battery charge threshold
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo 80 > /sys/class/power_supply/BAT0/charge_control_end_threshold'
ExecStart=/bin/bash -c 'echo 75 > /sys/class/power_supply/BAT0/charge_control_start_threshold'

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable battery-threshold
sudo systemctl start battery-threshold
```

---

## Linux (Dell)

```bash
# Install Dell command tools
sudo apt install libsmbios-bin

# Set charging mode
sudo smbios-battery-ctl --set-custom-charge-interval=75-80

# Check current settings
sudo smbios-battery-ctl --get-charging-cfg
```

---

## Linux (ASUS)

```bash
# ASUS laptops
echo 80 | sudo tee /sys/class/power_supply/BAT0/charge_control_end_threshold

# Check current threshold
cat /sys/class/power_supply/BAT0/charge_control_end_threshold
```

---

## Windows (Lenovo Vantage)

```powershell
# Using Lenovo Vantage app (GUI)
# 1. Install Lenovo Vantage from Microsoft Store
# 2. Open Lenovo Vantage
# 3. Go to Device > Power
# 4. Enable "Conservation Mode" (charges to 55-60%)
# OR
# 5. Set custom threshold under "Battery Charge Threshold"

# PowerShell (requires Lenovo System Interface Foundation)
# Check if available
Get-WmiObject -Namespace root\WMI -Class Lenovo_SetBatteryChargeThreshold

# Set threshold (requires admin)
# Note: This varies by model, use Lenovo Vantage for reliability
```

---

## Windows (Dell)

```powershell
# Using Dell Power Manager (GUI)
# 1. Install Dell Power Manager
# 2. Open Dell Power Manager
# 3. Go to Battery Settings
# 4. Select "Primarily AC Use" or "Custom" (80%)

# Check battery info
powercfg /batteryreport
# Open: C:\Windows\system32\battery-report.html
```

---

## Windows (ASUS)

```powershell
# ASUS Battery Health Charging
# 1. Install MyASUS from Microsoft Store
# 2. Open MyASUS > Customization > Battery Health Charging
# 3. Select mode:
#    - Full Capacity Mode (100%)
#    - Balanced Mode (80%)
#    - Maximum Lifespan Mode (60%)
```

---

## macOS

```bash
# macOS Monterey+ has built-in battery optimization
# System Preferences > Battery > Battery Health > Manage battery longevity

# Check battery status
pmset -g batt

# For older macOS, use AlDente (third-party app)
# https://github.com/davidwernhart/AlDente
```

---

## Verify Settings

### Linux

```bash
# Check current charge level
cat /sys/class/power_supply/BAT0/capacity

# Check charging status
cat /sys/class/power_supply/BAT0/status

# Monitor battery
watch -n 2 'cat /sys/class/power_supply/BAT0/capacity'

# Detailed battery info
upower -i /org/freedesktop/UPower/devices/battery_BAT0
```

### Windows

```powershell
# Battery report
powercfg /batteryreport

# Check battery health
powercfg /batteryreport /output "C:\battery-report.html"

# Battery status
WMIC Path Win32_Battery Get EstimatedChargeRemaining
```

---

## Best Practices

1. **Keep charge between 20-80%** for daily use
2. **Full discharge once a month** to calibrate battery meter
3. **Avoid high temperatures** (keep laptop cool)
4. **Remove from charger** when fully charged (if no threshold control)
5. **Store at 50%** if not using for extended period

---