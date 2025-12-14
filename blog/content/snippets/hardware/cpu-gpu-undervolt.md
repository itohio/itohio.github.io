---
title: "CPU/GPU Undervolting"
date: 2024-12-12
draft: false
category: "hardware"
tags: ["hardware-knowhow", "cpu", "gpu", "undervolt", "performance", "power"]
---


Reduce CPU/GPU voltage to lower temperatures and power consumption while maintaining performance.

---

## Why Undervolt?

**Benefits**:
- Lower temperatures (5-15°C reduction)
- Reduced power consumption (10-30W savings)
- Quieter fans
- Extended battery life
- Same or better performance (less thermal throttling)

**Risks**: System instability if too aggressive (easily reversible)

---

## Intel CPU Undervolting (Linux)

### Using intel-undervolt

```bash
# Install
sudo apt install intel-undervolt

# Check current values
sudo intel-undervolt read

# Configure
sudo nano /etc/intel-undervolt.conf

# Example configuration (adjust values for your CPU)
# Start conservative, test stability
undervolt 0 'CPU' -100
undervolt 1 'GPU' -75
undervolt 2 'CPU Cache' -100
undervolt 3 'System Agent' -0
undervolt 4 'Analog I/O' -0

# Apply settings
sudo intel-undervolt apply

# Enable on boot
sudo systemctl enable intel-undervolt

# Monitor temperatures
watch -n 1 'sensors | grep Core'
```

### Stress Test

```bash
# Install stress test tools
sudo apt install stress-ng

# CPU stress test (run for 30 minutes)
stress-ng --cpu $(nproc) --timeout 30m --metrics-brief

# Monitor during test
watch -n 1 'sensors && cat /proc/cpuinfo | grep MHz'
```

---

## Intel CPU Undervolting (Windows)

### Using ThrottleStop

```powershell
# Download ThrottleStop
# https://www.techpowerup.com/download/techpowerup-throttlestop/

# Steps:
# 1. Run ThrottleStop as Administrator
# 2. Click "FIVR" button
# 3. Select "CPU Core" from dropdown
# 4. Check "Unlock Adjustable Voltage"
# 5. Set offset (start with -50mV, increase gradually)
# 6. Click "Apply"
# 7. Repeat for "CPU Cache", "Intel GPU"
# 8. Save profile
# 9. Enable "Start Minimized" and "Minimize on Close"
# 10. Add to startup

# Recommended starting values:
# CPU Core: -100mV
# CPU Cache: -100mV
# Intel GPU: -75mV
# System Agent: -50mV
```

### Stress Test (Windows)

```powershell
# Download Prime95
# https://www.mersenne.org/download/

# Run blend test for 30 minutes
# Monitor with HWiNFO64 or Core Temp
```

---

## AMD CPU Undervolting (Linux)

### Using ryzenadj

```bash
# Install
git clone https://github.com/FlyGoat/RyzenAdj.git
cd RyzenAdj
mkdir build && cd build
cmake ..
make

# Set power limits and voltage
sudo ./ryzenadj --stapm-limit=25000 --fast-limit=30000 --slow-limit=25000 --tctl-temp=85

# Create systemd service
sudo tee /etc/systemd/system/ryzenadj.service <<EOF
[Unit]
Description=RyzenAdj Power Management
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/path/to/ryzenadj --stapm-limit=25000 --fast-limit=30000 --slow-limit=25000

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable ryzenadj
```

---

## AMD CPU Undervolting (Windows)

### Using Ryzen Master

```powershell
# Download AMD Ryzen Master
# https://www.amd.com/en/technologies/ryzen-master

# Steps:
# 1. Install and run as Administrator
# 2. Go to "Profile 1"
# 3. Enable "Precision Boost Overdrive"
# 4. Adjust "Curve Optimizer" (negative offset)
# 5. Start with -10 for all cores
# 6. Test stability
# 7. Gradually increase to -20 or -30
# 8. Save profile
```

---

## NVIDIA GPU Undervolting (Linux)

```bash
# Enable coolbits
sudo nvidia-xconfig --cool-bits=28

# Restart X server
sudo systemctl restart display-manager

# Using nvidia-settings GUI
nvidia-settings

# Or command line
nvidia-settings -a "[gpu:0]/GPUGraphicsClockOffset[3]=100"
nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffset[3]=200"

# Set power limit
sudo nvidia-smi -pl 150  # 150W limit

# Monitor
watch -n 1 nvidia-smi
```

---

## NVIDIA GPU Undervolting (Windows)

### Using MSI Afterburner

```powershell
# Download MSI Afterburner
# https://www.msi.com/Landing/afterburner

# Steps:
# 1. Install and run
# 2. Press Ctrl+F to open voltage/frequency curve
# 3. Select desired voltage point (e.g., 900mV)
# 4. Drag up to desired frequency
# 5. Flatten curve to the right
# 6. Click checkmark to apply
# 7. Test with benchmark (3DMark, Unigine Heaven)
# 8. Save profile

# Example settings:
# Stock: 1.05V @ 1800MHz
# Undervolted: 0.90V @ 1800MHz
# Result: Same performance, -20W power, -10°C temp
```

---

## AMD GPU Undervolting (Linux)

```bash
# Using CoreCtrl
sudo apt install corectrl

# Run CoreCtrl
corectrl

# Enable performance mode
# Adjust voltage/frequency curve in GUI

# Or manual via sysfs
echo "manual" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level

# Set voltage
echo "s 0 800" | sudo tee /sys/class/drm/card0/device/pp_od_clk_voltage
```

---

## AMD GPU Undervolting (Windows)

### Using AMD Adrenalin

```powershell
# AMD Adrenalin Software
# 1. Open AMD Software
# 2. Go to Performance > Tuning
# 3. Enable "Manual" tuning
# 4. Adjust "GPU Tuning"
# 5. Reduce voltage (start with -50mV)
# 6. Test with games/benchmarks
# 7. Save profile
```

---

## Monitoring Tools

### Linux

```bash
# CPU temperature
sensors

# Detailed monitoring
sudo apt install lm-sensors
sudo sensors-detect
watch -n 1 sensors

# Power consumption
sudo apt install powertop
sudo powertop

# GPU monitoring
nvidia-smi  # NVIDIA
radeontop   # AMD
```

### Windows

```powershell
# HWiNFO64 (recommended)
# https://www.hwinfo.com/

# Core Temp (CPU)
# https://www.alcpu.com/CoreTemp/

# GPU-Z (GPU)
# https://www.techpowerup.com/gpuz/
```

---

## Stability Testing

### CPU Stress Tests

```bash
# Linux
stress-ng --cpu $(nproc) --timeout 30m
prime95  # Download from mersenne.org

# Monitor for:
# - System crashes/freezes
# - Errors in dmesg
# - Temperature spikes
```

### GPU Stress Tests

```bash
# Unigine Heaven
# https://benchmark.unigine.com/heaven

# FurMark (extreme test)
# https://geeks3d.com/furmark/

# Run for 30 minutes minimum
# Monitor for artifacts, crashes, throttling
```

---

## Safe Undervolting Values

### Intel CPU (Starting Points)

| Component | Conservative | Aggressive |
|-----------|-------------|------------|
| CPU Core | -50mV | -125mV |
| CPU Cache | -50mV | -125mV |
| Intel GPU | -25mV | -75mV |
| System Agent | 0mV | -50mV |

### AMD CPU (Curve Optimizer)

| Starting | Stable | Aggressive |
|----------|--------|------------|
| -5 | -15 | -30 |

### GPU

| Type | Conservative | Aggressive |
|------|-------------|------------|
| NVIDIA | -50mV | -150mV |
| AMD | -50mV | -100mV |

---

## Troubleshooting

```bash
# System crashes/freezes
# - Reduce undervolt by 25mV
# - Test again

# Blue screen (Windows)
# - Boot to safe mode
# - Reset ThrottleStop/Ryzen Master settings

# Black screen (GPU)
# - Reboot (settings reset automatically)
# - Use less aggressive undervolt

# Check system logs (Linux)
sudo dmesg | grep -i error
sudo journalctl -xe
```

---