---
title: "FFmpeg Video Conversion"
date: 2024-12-12
description: "Format conversion, codecs, and quality settings"
tags: ["ffmpeg", "video", "conversion"]
---



## Basic Conversion

### Convert to MP4 (H.264)

```bash
# High quality
ffmpeg -i input.avi -c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k output.mp4

# Medium quality (default)
ffmpeg -i input.avi -c:v libx264 -crf 23 output.mp4

# Fast encoding
ffmpeg -i input.avi -c:v libx264 -preset ultrafast -crf 23 output.mp4
```

### Convert to WebM (VP9)

```bash
# High quality
ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 -c:a libopus output.webm

# Two-pass encoding (better quality)
ffmpeg -i input.mp4 -c:v libvpx-vp9 -b:v 1M -pass 1 -f null /dev/null
ffmpeg -i input.mp4 -c:v libvpx-vp9 -b:v 1M -pass 2 -c:a libopus output.webm
```

### Convert to H.265/HEVC

```bash
# Better compression than H.264
ffmpeg -i input.mp4 -c:v libx265 -crf 28 -c:a aac -b:a 128k output.mp4

# Hardware acceleration (NVIDIA)
ffmpeg -i input.mp4 -c:v hevc_nvenc -preset slow -crf 23 output.mp4
```

---

## Quality Settings

### CRF (Constant Rate Factor)

Lower = better quality, larger file

| CRF | Quality | Use Case |
|-----|---------|----------|
| 18-22 | Very high | Archival, professional |
| 23-28 | Good | General use, streaming |
| 29-35 | Lower | Small files, previews |

```bash
# Set CRF
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4
```

### Bitrate Control

```bash
# Constant bitrate
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -maxrate 2M -bufsize 4M output.mp4

# Variable bitrate with target
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M output.mp4

# Two-pass encoding (best quality for target bitrate)
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 1 -f null /dev/null
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 2 output.mp4
```

---

## Codec Options

### H.264 (libx264)

```bash
# Presets: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 23 output.mp4

# Profile (for compatibility)
ffmpeg -i input.mp4 -c:v libx264 -profile:v baseline -level 3.0 output.mp4
# Profiles: baseline, main, high

# Tune (optional optimization)
ffmpeg -i input.mp4 -c:v libx264 -tune film output.mp4
# Tunes: film, animation, grain, stillimage, fastdecode, zerolatency
```

### H.265/HEVC (libx265)

```bash
# Basic
ffmpeg -i input.mp4 -c:v libx265 -crf 28 output.mp4

# With preset
ffmpeg -i input.mp4 -c:v libx265 -preset medium -crf 28 output.mp4

# 10-bit encoding (better quality)
ffmpeg -i input.mp4 -c:v libx265 -pix_fmt yuv420p10le -crf 28 output.mp4
```

### VP9 (libvpx-vp9)

```bash
# Single pass
ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 output.webm

# Two-pass (recommended)
ffmpeg -i input.mp4 -c:v libvpx-vp9 -b:v 1M -pass 1 -f null /dev/null && \
ffmpeg -i input.mp4 -c:v libvpx-vp9 -b:v 1M -pass 2 output.webm
```

---

## Resolution & Frame Rate

### Change Resolution

```bash
# Scale to width (maintain aspect ratio)
ffmpeg -i input.mp4 -vf scale=1280:-2 output.mp4

# Scale to height
ffmpeg -i input.mp4 -vf scale=-2:720 output.mp4

# Specific resolution
ffmpeg -i input.mp4 -vf scale=1920:1080 output.mp4

# Scale with high-quality algorithm
ffmpeg -i input.mp4 -vf scale=1280:720:flags=lanczos output.mp4
```

### Change Frame Rate

```bash
# Set frame rate
ffmpeg -i input.mp4 -r 30 output.mp4

# Convert to 60fps (interpolation)
ffmpeg -i input.mp4 -filter:v "minterpolate='fps=60'" output.mp4

# Reduce frame rate (drop frames)
ffmpeg -i input.mp4 -r 24 output.mp4
```

---

## Audio Codecs

### AAC

```bash
# Default quality
ffmpeg -i input.mp4 -c:a aac -b:a 192k output.mp4

# High quality
ffmpeg -i input.mp4 -c:a aac -b:a 256k output.mp4
```

### Opus (best for WebM)

```bash
ffmpeg -i input.mp4 -c:a libopus -b:a 128k output.webm
```

### MP3

```bash
ffmpeg -i input.mp4 -c:a libmp3lame -b:a 192k output.mp4
```

### Copy audio (no re-encoding)

```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -c:a copy output.mp4
```

---

## Container Formats

### MP4

```bash
# Standard MP4
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4

# MP4 with faststart (web streaming)
ffmpeg -i input.avi -c:v libx264 -c:a aac -movflags +faststart output.mp4
```

### MKV

```bash
# Matroska (supports multiple audio/subtitle tracks)
ffmpeg -i input.mp4 -c copy output.mkv
```

### WebM

```bash
# For web (VP9 + Opus)
ffmpeg -i input.mp4 -c:v libvpx-vp9 -c:a libopus output.webm
```

### AVI

```bash
ffmpeg -i input.mp4 -c:v mpeg4 -q:v 5 -c:a libmp3lame output.avi
```

---

## Hardware Acceleration

### NVIDIA (NVENC)

```bash
# H.264
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc -preset slow output.mp4

# H.265
ffmpeg -hwaccel cuda -i input.mp4 -c:v hevc_nvenc -preset slow output.mp4
```

### Intel Quick Sync (QSV)

```bash
ffmpeg -hwaccel qsv -i input.mp4 -c:v h264_qsv -preset slow output.mp4
```

### AMD (AMF)

```bash
ffmpeg -i input.mp4 -c:v h264_amf -quality quality output.mp4
```

---

## Copy Streams (No Re-encoding)

```bash
# Copy both video and audio
ffmpeg -i input.mp4 -c copy output.mkv

# Copy video, re-encode audio
ffmpeg -i input.mp4 -c:v copy -c:a aac -b:a 192k output.mp4

# Copy audio, re-encode video
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -c:a copy output.mp4
```

---

## Useful Flags

```bash
# Overwrite output without asking
-y

# Don't overwrite
-n

# Limit output file size (approximate)
-fs 100M

# Set metadata
-metadata title="My Video" -metadata author="Name"

# Verbose output
-v verbose

# Quiet (only errors)
-v quiet

# Show progress
-progress pipe:1
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Convert to MP4 | `ffmpeg -i input.avi -c:v libx264 -crf 23 output.mp4` |
| High quality | `ffmpeg -i input.avi -c:v libx264 -crf 18 output.mp4` |
| Small file | `ffmpeg -i input.avi -c:v libx264 -crf 28 output.mp4` |
| WebM | `ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 30 output.webm` |
| H.265 | `ffmpeg -i input.mp4 -c:v libx265 -crf 28 output.mp4` |
| Scale to 720p | `ffmpeg -i input.mp4 -vf scale=-2:720 output.mp4` |
| Copy streams | `ffmpeg -i input.mp4 -c copy output.mkv` |

---

## Tips

- Use `-crf` for variable bitrate (recommended)
- Use `-b:v` for constant bitrate (streaming)
- Use `-preset slow` for better compression (slower encoding)
- Use `-preset ultrafast` for quick encoding (larger files)
- Add `-movflags +faststart` for MP4 web streaming
- Use hardware acceleration for faster encoding on supported hardware

