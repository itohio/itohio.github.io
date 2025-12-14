---
title: "FFmpeg Video Editing"
date: 2024-12-12
description: "Cutting, trimming, concatenating, and basic editing"
tags: ["ffmpeg", "video", "editing", "trim", "concat"]
---



## Cutting & Trimming

### Cut by Time

```bash
# Cut from 00:01:00 to 00:02:00 (1 minute duration)
ffmpeg -i input.mp4 -ss 00:01:00 -to 00:02:00 -c copy output.mp4

# Cut from 00:01:00 for 30 seconds
ffmpeg -i input.mp4 -ss 00:01:00 -t 00:00:30 -c copy output.mp4

# Fast seek (less accurate but faster)
ffmpeg -ss 00:01:00 -i input.mp4 -t 00:00:30 -c copy output.mp4
```

### Remove Beginning/End

```bash
# Remove first 10 seconds
ffmpeg -i input.mp4 -ss 00:00:10 -c copy output.mp4

# Remove last 10 seconds (if duration is 60s)
ffmpeg -i input.mp4 -t 00:00:50 -c copy output.mp4
```

## Concatenation

### Method 1: Concat Demuxer (Same Format)

```bash
# Create file list
echo "file 'part1.mp4'" > list.txt
echo "file 'part2.mp4'" >> list.txt
echo "file 'part3.mp4'" >> list.txt

# Concatenate
ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4
```

### Method 2: Concat Filter (Different Formats)

```bash
ffmpeg -i part1.mp4 -i part2.mp4 -i part3.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" output.mp4
```

## Speed Control

### Speed Up

```bash
# 2x speed (video and audio)
ffmpeg -i input.mp4 -filter_complex "[0:v]setpts=0.5*PTS[v];[0:a]atempo=2.0[a]" \
  -map "[v]" -map "[a]" output.mp4

# 4x speed
ffmpeg -i input.mp4 -filter_complex "[0:v]setpts=0.25*PTS[v];[0:a]atempo=2.0,atempo=2.0[a]" \
  -map "[v]" -map "[a]" output.mp4
```

### Slow Down

```bash
# 0.5x speed (half speed)
ffmpeg -i input.mp4 -filter_complex "[0:v]setpts=2.0*PTS[v];[0:a]atempo=0.5[a]" \
  -map "[v]" -map "[a]" output.mp4
```

## Rotation & Flipping

```bash
# Rotate 90° clockwise
ffmpeg -i input.mp4 -vf "transpose=1" output.mp4

# Rotate 90° counter-clockwise
ffmpeg -i input.mp4 -vf "transpose=2" output.mp4

# Rotate 180°
ffmpeg -i input.mp4 -vf "transpose=2,transpose=2" output.mp4

# Flip horizontal
ffmpeg -i input.mp4 -vf "hflip" output.mp4

# Flip vertical
ffmpeg -i input.mp4 -vf "vflip" output.mp4
```

## Cropping

```bash
# Crop to 1280x720 from top-left
ffmpeg -i input.mp4 -vf "crop=1280:720:0:0" output.mp4

# Crop center 1280x720
ffmpeg -i input.mp4 -vf "crop=1280:720" output.mp4

# Crop with offset (x=100, y=50)
ffmpeg -i input.mp4 -vf "crop=1280:720:100:50" output.mp4
```

## Extract Audio/Video

```bash
# Extract audio only
ffmpeg -i input.mp4 -vn -acodec copy audio.m4a

# Extract video only (no audio)
ffmpeg -i input.mp4 -an -vcodec copy video.mp4

# Extract specific audio track
ffmpeg -i input.mkv -map 0:a:1 -c copy audio_track2.aac
```

## Add/Replace Audio

```bash
# Replace audio
ffmpeg -i video.mp4 -i audio.mp3 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output.mp4

# Add audio track (keep original)
ffmpeg -i video.mp4 -i audio.mp3 -c copy -map 0 -map 1:a output.mkv

# Remove audio
ffmpeg -i input.mp4 -an -c:v copy output.mp4
```

## Further Reading

- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [FFmpeg Wiki](https://trac.ffmpeg.org/wiki)

