---
title: "CMD Commands"
date: 2024-12-12
draft: false
description: "Useful Windows Command Prompt commands"
tags: ["cmd", "shell", "windows", "command-prompt"]
---



## Command Dispatcher Pattern

```batch
@echo off
setlocal

set "command=%~1"

if "%command%"=="start" goto :start
if "%command%"=="stop" goto :stop
if "%command%"=="restart" goto :restart
if "%command%"=="status" goto :status
goto :usage

:start
echo Starting service...
rem Your start logic here
goto :end

:stop
echo Stopping service...
rem Your stop logic here
goto :end

:restart
call :stop
call :start
goto :end

:status
echo Checking status...
rem Your status logic here
goto :end

:usage
echo Usage: %~nx0 {start^|stop^|restart^|status}
exit /b 1

:end
endlocal
```

## Useful Commands

### File Operations

```batch
rem Copy files
copy source.txt dest.txt
xcopy /E /I source_dir dest_dir

rem Move files
move file.txt newlocation\

rem Delete files
del file.txt
del /Q /S *.tmp

rem Create directory
mkdir path\to\directory
md nested\directories

rem Remove directory
rmdir /S /Q directory

rem List files
dir
dir /S /B *.txt

rem Find files
where /R C:\ filename.txt

rem File attributes
attrib +R file.txt
attrib -H -S file.txt
```

### Text Processing

```batch
rem Display file
type file.txt

rem Find string
find "text" file.txt
findstr /I "pattern" *.txt

rem Sort
sort file.txt

rem More (paginate)
type file.txt | more

rem Count lines
find /C /V "" file.txt
```

### Process Management

```batch
rem List processes
tasklist

rem Find process
tasklist | findstr "notepad"

rem Kill process
taskkill /IM notepad.exe /F
taskkill /PID 1234 /F

rem Start process
start notepad.exe
start "" "C:\Program Files\App\app.exe"

rem Run elevated
runas /user:Administrator cmd
```

### Network Commands

```batch
rem Ping
ping google.com
ping -n 10 192.168.1.1

rem Trace route
tracert google.com

rem DNS lookup
nslookup google.com

rem IP configuration
ipconfig
ipconfig /all
ipconfig /flushdns

rem Network connections
netstat -ano
netstat -ano | findstr :80

rem Test port
telnet hostname 80
```

### System Information

```batch
rem System info
systeminfo

rem Computer name
hostname

rem Windows version
ver

rem Environment variables
set
echo %PATH%
echo %USERPROFILE%

rem Disk info
wmic logicaldisk get name,size,freespace

rem CPU info
wmic cpu get name,numberofcores

rem Memory info
wmic memorychip get capacity

rem Services
sc query
sc query state= all
net start
```

## Scripting Techniques

### Variables

```batch
rem Set variable
set VAR=value
set /A NUM=10+5

rem User input
set /P NAME=Enter name: 

rem Command output
for /F %%i in ('date /t') do set TODAY=%%i

rem Environment variables
echo %COMPUTERNAME%
echo %USERNAME%
echo %CD%
```

### Conditionals

```batch
rem If statement
if exist file.txt (
    echo File exists
) else (
    echo File not found
)

rem String comparison
if "%VAR%"=="value" (
    echo Match
)

rem Numeric comparison
if %NUM% GTR 10 (
    echo Greater than 10
)

rem Check error level
command
if errorlevel 1 (
    echo Command failed
)

rem Check if variable is defined
if defined VAR (
    echo Variable is set
)
```

### Loops

```batch
rem For loop (files)
for %%F in (*.txt) do (
    echo %%F
)

rem For loop (numbers)
for /L %%N in (1,1,10) do (
    echo %%N
)

rem For loop (directories)
for /D %%D in (*) do (
    echo %%D
)

rem For loop (command output)
for /F %%i in ('dir /B') do (
    echo %%i
)

rem Recursive
for /R %%F in (*.log) do (
    del "%%F"
)
```

### Functions (Subroutines)

```batch
@echo off

call :function arg1 arg2
goto :eof

:function
echo Arg1: %~1
echo Arg2: %~2
exit /b 0
```

## One-Liners

```batch
rem Backup with timestamp
for /F "tokens=1-4 delims=/ " %%a in ('date /t') do set DATE=%%c%%a%%b
copy file.txt file.txt.%DATE%

rem Find large files
forfiles /S /C "cmd /c if @fsize gtr 104857600 echo @path @fsize"

rem Kill process by port
for /f "tokens=5" %a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %a

rem Batch rename
for %f in (*.txt) do ren "%f" "prefix_%f"

rem Create multiple directories
for /L %i in (1,1,10) do mkdir folder%i
```

## Further Reading

- [CMD Reference](https://ss64.com/nt/)
- [Windows Commands](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/windows-commands)

