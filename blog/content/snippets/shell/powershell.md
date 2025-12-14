---
title: "PowerShell Commands"
date: 2024-12-12
draft: false
description: "Useful PowerShell commands and scripting techniques"
tags: ["powershell", "shell", "windows", "scripting"]
---



## Command Dispatcher Pattern

```powershell
# CommandDispatcher.ps1
param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Command,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    $Arguments
)

switch ($Command) {
    "start" {
        Write-Host "Starting service..."
        # Your start logic here
    }
    "stop" {
        Write-Host "Stopping service..."
        # Your stop logic here
    }
    "restart" {
        & $PSCommandPath stop
        & $PSCommandPath start
    }
    "status" {
        Write-Host "Checking status..."
        # Your status logic here
    }
    "deploy" {
        $env = $Arguments[0]
        Write-Host "Deploying to $env..."
        # Your deploy logic here
    }
    default {
        Write-Host "Usage: .\script.ps1 {start|stop|restart|status|deploy <env>}"
        exit 1
    }
}
```

## Useful Cmdlets

### File Operations

```powershell
# Get files modified in last 7 days
Get-ChildItem -Recurse | Where-Object {$_.LastWriteTime -gt (Get-Date).AddDays(-7)}

# Find and delete old files
Get-ChildItem -Recurse | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item

# Find large files (>100MB)
Get-ChildItem -Recurse | Where-Object {$_.Length -gt 100MB} | Sort-Object Length -Descending

# Copy with progress
Copy-Item -Path "source\*" -Destination "dest\" -Recurse -Verbose

# Create directory structure
New-Item -ItemType Directory -Path "path\to\nested\directories" -Force

# Get folder sizes
Get-ChildItem | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    [PSCustomObject]@{
        Name = $_.Name
        SizeMB = [math]::Round($size, 2)
    }
} | Sort-Object SizeMB -Descending

# Find largest files
Get-ChildItem -Recurse -File | Sort-Object Length -Descending | Select-Object -First 10 FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}
```

### Text Processing

```powershell
# Read file
Get-Content file.txt

# Read specific lines
Get-Content file.txt | Select-Object -First 10
Get-Content file.txt | Select-Object -Last 10
Get-Content file.txt | Select-Object -Skip 5 -First 10

# Search in files
Select-String -Path "*.txt" -Pattern "error"

# Replace text
(Get-Content file.txt) -replace 'old', 'new' | Set-Content file.txt

# Count lines
(Get-Content file.txt).Count

# Remove duplicates
Get-Content file.txt | Select-Object -Unique

# Sort and export
Get-Content file.txt | Sort-Object | Set-Content sorted.txt

# CSV operations
Import-Csv data.csv | Where-Object {$_.Age -gt 30} | Export-Csv filtered.csv -NoTypeInformation
```

### Process Management

```powershell
# Get processes
Get-Process

# Find process by name
Get-Process | Where-Object {$_.Name -like "*chrome*"}

# Kill process
Stop-Process -Name "notepad" -Force

# Kill process by ID
Stop-Process -Id 1234

# Kill process by port
Get-NetTCPConnection -LocalPort 8080 | ForEach-Object {Stop-Process -Id $_.OwningProcess -Force}

# Start process
Start-Process "notepad.exe"

# Run as administrator
Start-Process powershell -Verb RunAs

# Run and wait
Start-Process "setup.exe" -Wait

# Background job
Start-Job -ScriptBlock { Get-Process }
Get-Job
Receive-Job -Id 1
```

### Network Commands

```powershell
# Test connection
Test-NetConnection google.com -Port 443

# Download file
Invoke-WebRequest -Uri "https://example.com/file.zip" -OutFile "file.zip"

# POST JSON
$body = @{key="value"} | ConvertTo-Json
Invoke-RestMethod -Uri "https://api.example.com" -Method Post -Body $body -ContentType "application/json"

# GET request
Invoke-RestMethod -Uri "https://api.example.com/data"

# Test DNS
Resolve-DnsName example.com

# Get IP configuration
Get-NetIPConfiguration

# Show listening ports
Get-NetTCPConnection -State Listen

# Ping sweep
1..254 | ForEach-Object {Test-Connection -ComputerName "192.168.1.$_" -Count 1 -Quiet}
```

### System Information

```powershell
# Computer info
Get-ComputerInfo

# OS version
Get-CimInstance Win32_OperatingSystem

# CPU info
Get-CimInstance Win32_Processor

# Memory info
Get-CimInstance Win32_PhysicalMemory

# Disk info
Get-Disk
Get-Volume

# Installed software
Get-ItemProperty HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\* | Select-Object DisplayName, DisplayVersion

# Windows features
Get-WindowsOptionalFeature -Online

# Services
Get-Service
Get-Service | Where-Object {$_.Status -eq "Running"}

# Event logs
Get-EventLog -LogName System -Newest 10
Get-WinEvent -LogName Application -MaxEvents 10
```

## Scripting Techniques

### Error Handling

```powershell
# Stop on error
$ErrorActionPreference = "Stop"

# Try-catch
try {
    Get-Content "nonexistent.txt"
} catch {
    Write-Error "An error occurred: $_"
}

# Finally block
try {
    # Code
} catch {
    # Handle error
} finally {
    # Cleanup
}

# Check command exists
if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "Git is installed"
}
```

### Parameters

```powershell
param(
    [Parameter(Mandatory=$true)]
    [string]$Name,
    
    [Parameter(Mandatory=$false)]
    [int]$Age = 0,
    
    [switch]$Verbose,
    
    [ValidateSet("Dev","Test","Prod")]
    [string]$Environment = "Dev"
)

Write-Host "Name: $Name, Age: $Age, Env: $Environment"
if ($Verbose) {
    Write-Host "Verbose mode enabled"
}
```

### Functions

```powershell
function Get-Timestamp {
    return Get-Date -Format "yyyyMMdd_HHmmss"
}

function Backup-File {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath,
        
        [Parameter(Mandatory=$true)]
        [string]$BackupDir
    )
    
    if (-not (Test-Path $FilePath)) {
        Write-Error "File not found: $FilePath"
        return
    }
    
    $timestamp = Get-Timestamp
    $fileName = Split-Path $FilePath -Leaf
    $backupPath = Join-Path $BackupDir "$fileName.$timestamp"
    
    Copy-Item $FilePath $backupPath
    Write-Host "Backed up to: $backupPath"
}

# Advanced function with pipeline support
function Get-LargeFiles {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline=$true)]
        [string]$Path = ".",
        
        [int]$SizeMB = 100
    )
    
    process {
        Get-ChildItem -Path $Path -Recurse -File |
            Where-Object {$_.Length -gt ($SizeMB * 1MB)} |
            Select-Object FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}
    }
}
```

### Arrays and Hash Tables

```powershell
# Arrays
$array = @("item1", "item2", "item3")
$array += "item4"
$array.Count
$array[0]

# Iterate
foreach ($item in $array) {
    Write-Host $item
}

# Hash tables (dictionaries)
$config = @{
    Host = "localhost"
    Port = 8080
    Enabled = $true
}

$config["Host"]
$config.Port

# Add/remove
$config["NewKey"] = "value"
$config.Remove("NewKey")

# Iterate
foreach ($key in $config.Keys) {
    Write-Host "$key = $($config[$key])"
}
```

### String Operations

```powershell
# String length
$str = "hello"
$str.Length

# Substring
$str.Substring(0, 2)  # "he"

# Replace
$str.Replace("l", "L")  # "heLLo"

# Split
$str.Split("l")

# Join
$array -join ","

# Format
"Name: {0}, Age: {1}" -f "John", 30

# Contains
$str.Contains("ell")

# Case conversion
$str.ToUpper()
$str.ToLower()

# Trim
"  text  ".Trim()
```

### Conditionals

```powershell
# File tests
if (Test-Path "file.txt") {
    Write-Host "File exists"
}

# String comparison
if ($str -eq "value") {
    Write-Host "Match"
}

# Numeric comparison
if ($num -gt 10) {
    Write-Host "Greater than 10"
}

# Logical operators
if ((Test-Path "file.txt") -and (Test-Path "file2.txt")) {
    Write-Host "Both files exist"
}

# Pattern matching
if ($str -like "*pattern*") {
    Write-Host "Pattern found"
}

# Regex
if ($str -match "\d+") {
    Write-Host "Contains digits"
}
```

### Loops

```powershell
# For loop
for ($i = 0; $i -lt 10; $i++) {
    Write-Host $i
}

# ForEach loop
foreach ($item in $array) {
    Write-Host $item
}

# ForEach-Object (pipeline)
1..10 | ForEach-Object {
    Write-Host $_
}

# While loop
$i = 0
while ($i -lt 10) {
    Write-Host $i
    $i++
}

# Do-While
do {
    Write-Host $i
    $i++
} while ($i -lt 10)
```

## One-Liners

```powershell
# Backup with timestamp
Copy-Item file.txt "file.txt.$(Get-Date -Format 'yyyyMMdd_HHmmss')"

# Find and archive
Compress-Archive -Path (Get-ChildItem -Filter "*.log") -DestinationPath "logs.zip"

# Monitor file changes
Get-Content -Path "log.txt" -Wait -Tail 10

# Parallel execution
1..10 | ForEach-Object -Parallel { Start-Sleep 1; $_ } -ThrottleLimit 5

# Generate random string
-join ((65..90) + (97..122) | Get-Random -Count 16 | ForEach-Object {[char]$_})

# Check if admin
([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# Quick HTTP server
Start-Process "python" "-m http.server 8000"
```

## Further Reading

- [PowerShell Documentation](https://docs.microsoft.com/en-us/powershell/)
- [PowerShell Gallery](https://www.powershellgallery.com/)
- [PSScriptAnalyzer](https://github.com/PowerShell/PSScriptAnalyzer) - PowerShell script analyzer

