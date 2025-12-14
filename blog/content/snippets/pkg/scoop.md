---
title: "Scoop Package Manager (Windows)"
date: 2024-12-12
draft: false
category: "pkg"
tags: ["pkg-knowhow", "scoop", "windows", "package-manager"]
---


Scoop - Command-line installer for Windows.

---

## Installation

```powershell
# Set execution policy
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install Scoop
irm get.scoop.sh | iex

# Or long form
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression

# Verify
scoop --version
```

---

## Basic Commands

```powershell
# Search for app
scoop search app-name

# Install app
scoop install app-name

# Install specific version
scoop install app-name@version

# Install from bucket
scoop install bucket/app-name

# Uninstall app
scoop uninstall app-name

# Update Scoop
scoop update

# Update all apps
scoop update *

# Update specific app
scoop update app-name

# List installed apps
scoop list

# Show app info
scoop info app-name
```

---

## Buckets (Repositories)

```powershell
# List known buckets
scoop bucket known

# Add bucket
scoop bucket add extras
scoop bucket add versions
scoop bucket add java
scoop bucket add games

# Add custom bucket
scoop bucket add bucket-name https://github.com/user/bucket

# List added buckets
scoop bucket list

# Remove bucket
scoop bucket rm bucket-name

# Common buckets:
# - main (default)
# - extras (GUI apps)
# - versions (older versions)
# - java (JDK/JRE)
# - nerd-fonts
# - games
```

---

## Common Applications

```powershell
# Development tools
scoop install git
scoop install python
scoop install nodejs
scoop install go
scoop install rust
scoop install vscode

# System tools
scoop install 7zip
scoop install curl
scoop install wget
scoop install grep
scoop install sed
scoop install jq

# Extras bucket
scoop bucket add extras
scoop install googlechrome
scoop install firefox
scoop install vlc
scoop install obs-studio
```

---

## App Management

```powershell
# Show app status
scoop status

# Check for updates
scoop status

# Hold app (prevent updates)
scoop hold app-name

# Unhold app
scoop unhold app-name

# List held apps
scoop list | Where-Object { $_.held }

# Cleanup old versions
scoop cleanup app-name

# Cleanup all apps
scoop cleanup *

# Cache management
scoop cache show
scoop cache rm app-name
scoop cache rm *
```

---

## Global Installation

```powershell
# Install globally (for all users)
sudo scoop install -g app-name

# Note: Requires 'sudo' from scoop
scoop install sudo

# List global apps
scoop list -g

# Update global apps
sudo scoop update * -g

# Uninstall global app
sudo scoop uninstall -g app-name
```

---

## Shims and Environment

```powershell
# Scoop adds apps to PATH via shims
# Shim location: ~\scoop\shims

# Reset shims
scoop reset *

# Reset specific app
scoop reset app-name

# Check PATH
$env:PATH -split ';' | Select-String scoop
```

---

## Configuration

```powershell
# Show config
scoop config

# Set config
scoop config aria2-enabled true
scoop config aria2-warning-enabled false

# Proxy settings
scoop config proxy proxy.example.com:8080

# Cache directory
scoop config cache_path D:\scoop-cache

# Global install directory
scoop config global_path C:\ProgramData\scoop
```

---

## Manifest Files

### Create Custom Manifest

```json
// myapp.json
{
    "version": "1.0.0",
    "description": "My custom application",
    "homepage": "https://example.com",
    "license": "MIT",
    "url": "https://example.com/myapp-1.0.0.zip",
    "hash": "sha256:...",
    "bin": "myapp.exe",
    "shortcuts": [
        [
            "myapp.exe",
            "My Application"
        ]
    ],
    "checkver": {
        "url": "https://example.com/latest",
        "regex": "v([\\d.]+)"
    },
    "autoupdate": {
        "url": "https://example.com/myapp-$version.zip"
    }
}
```

```powershell
# Install from local manifest
scoop install .\myapp.json

# Install from URL
scoop install https://example.com/myapp.json
```

---

## Troubleshooting

```powershell
# Check for problems
scoop checkup

# Diagnose issues
scoop doctor

# Reset app
scoop reset app-name

# Reinstall app
scoop uninstall app-name
scoop install app-name

# Clear cache
scoop cache rm *

# Update Scoop itself
scoop update

# Verbose output
scoop install app-name -v
```

---

## Aliases

```powershell
# Create alias
scoop alias add update-all 'scoop update * ; scoop cleanup *'

# List aliases
scoop alias list

# Remove alias
scoop alias rm update-all

# Use alias
scoop update-all
```

---

## Backup and Restore

```powershell
# Export installed apps
scoop export > scoop-apps.txt

# Restore apps
Get-Content scoop-apps.txt | ForEach-Object { scoop install $_ }

# Or manually
scoop list | Out-File apps.txt
```

---

## Comparison with Other Package Managers

| Feature | Scoop | Chocolatey | Winget |
|---------|-------|------------|--------|
| Admin required | No | Sometimes | Sometimes |
| Install location | User profile | Program Files | Program Files |
| GUI apps | Yes (extras) | Yes | Yes |
| Portable apps | Yes | No | No |
| Custom buckets | Yes | Yes | Limited |

---

## Integration with Other Tools

### PowerShell Profile

```powershell
# Add to $PROFILE
# Scoop completion
Import-Module scoop-completion

# Aliases
Set-Alias -Name s -Value scoop
Set-Alias -Name si -Value 'scoop install'
Set-Alias -Name su -Value 'scoop update'
```

### Windows Terminal

```json
// settings.json
{
    "profiles": {
        "list": [
            {
                "name": "PowerShell (Scoop)",
                "commandline": "pwsh.exe -NoLogo",
                "startingDirectory": "%USERPROFILE%"
            }
        ]
    }
}
```

---

## Advanced Usage

```powershell
# Install multiple apps
scoop install git python nodejs

# Update and cleanup
scoop update * ; scoop cleanup *

# Search in specific bucket
scoop search extras/app-name

# Show app dependencies
scoop depends app-name

# Show app manifest
scoop cat app-name

# Edit app manifest
scoop edit app-name

# Download app without installing
scoop download app-name
```

---