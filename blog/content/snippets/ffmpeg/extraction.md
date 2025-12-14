---
title: "FFmpeg Stream Extraction & Merging"
date: 2024-12-12
description: "Extract and merge audio, video, and subtitle streams"
tags: ["ffmpeg", "extraction", "audio", "video", "subtitles"]
---

Extract individual streams (audio, video, subtitles) from media files and merge them back together.

---

## List Streams in File

### Show All Streams

```bash
# Detailed stream information
ffmpeg -i input.mp4

# More detailed (includes codec details)
ffprobe -i input.mp4

# JSON format (parseable)
ffprobe -v quiet -print_format json -show_streams input.mp4

# Show only specific stream types
ffprobe -v quiet -select_streams v -show_streams input.mp4  # Video only
ffprobe -v quiet -select_streams a -show_streams input.mp4  # Audio only
ffprobe -v quiet -select_streams s -show_streams input.mp4  # Subtitles only

# Compact format
ffprobe -v error -show_entries stream=index,codec_type,codec_name:stream_tags=language \
  -of compact input.mp4
```

### Human-Readable Stream List

```bash
# Show streams in readable format
ffprobe -v error -show_entries stream=index,codec_name,codec_type,width,height,channels,sample_rate:stream_tags=language \
  -of default=noprint_wrappers=1 input.mp4

# Example output:
# [STREAM]
# index=0
# codec_name=h264
# codec_type=video
# width=1920
# height=1080
# [/STREAM]
# [STREAM]
# index=1
# codec_name=aac
# codec_type=audio
# channels=2
# sample_rate=48000
# TAG:language=eng
# [/STREAM]
```

---

## Extract Video Stream

### Extract Video Only (No Audio)

```bash
# Copy video stream (no re-encoding, fast)
ffmpeg -i input.mp4 -an -c:v copy output_video.mp4

# Extract specific video stream (if multiple)
ffmpeg -i input.mkv -map 0:v:0 -c copy output_video.mp4

# Extract and re-encode video
ffmpeg -i input.mp4 -an -c:v libx264 -crf 23 output_video.mp4
```

**Flags**:
- `-an`: Remove audio
- `-c:v copy`: Copy video without re-encoding
- `-map 0:v:0`: Select first video stream

---

## Extract Audio Stream

### Extract Audio Only

```bash
# Copy audio stream (no re-encoding)
ffmpeg -i input.mp4 -vn -c:a copy output_audio.m4a

# Extract to MP3
ffmpeg -i input.mp4 -vn -c:a libmp3lame -b:a 192k output_audio.mp3

# Extract to WAV (uncompressed)
ffmpeg -i input.mp4 -vn -c:a pcm_s16le output_audio.wav

# Extract to FLAC (lossless)
ffmpeg -i input.mp4 -vn -c:a flac output_audio.flac

# Extract to Opus
ffmpeg -i input.mp4 -vn -c:a libopus -b:a 128k output_audio.opus
```

### Extract Specific Audio Track

```bash
# Extract first audio track
ffmpeg -i input.mkv -map 0:a:0 -c copy audio_track1.m4a

# Extract second audio track
ffmpeg -i input.mkv -map 0:a:1 -c copy audio_track2.m4a

# Extract audio track by language
ffmpeg -i input.mkv -map 0:m:language:eng -c copy audio_english.m4a
```

**Flags**:
- `-vn`: Remove video
- `-c:a copy`: Copy audio without re-encoding
- `-map 0:a:0`: Select first audio stream
- `-map 0:a:1`: Select second audio stream

---

## Extract Subtitles

### Extract Subtitle Tracks

```bash
# Extract first subtitle track to SRT
ffmpeg -i input.mkv -map 0:s:0 output.srt

# Extract all subtitle tracks
ffmpeg -i input.mkv -map 0:s -c copy output_%d.srt

# Extract subtitle by language
ffmpeg -i input.mkv -map 0:m:language:eng subtitle_english.srt

# Extract to ASS format
ffmpeg -i input.mkv -map 0:s:0 output.ass

# Extract to VTT (WebVTT)
ffmpeg -i input.mkv -map 0:s:0 output.vtt

# Extract embedded subtitles (e.g., from MP4)
ffmpeg -i input.mp4 -map 0:s:0 -c copy output.srt
```

### Extract Image-Based Subtitles (DVD/Blu-ray)

```bash
# Extract DVD subtitles (VobSub)
ffmpeg -i input.mkv -map 0:s:0 -c copy output.sub

# Extract PGS subtitles (Blu-ray)
ffmpeg -i input.mkv -map 0:s:0 -c copy output.sup
```

---

## Extract Multiple Streams

### Extract All Streams Separately

```bash
# Extract video
ffmpeg -i input.mkv -map 0:v:0 -c copy video.mp4

# Extract all audio tracks
ffmpeg -i input.mkv -map 0:a:0 -c copy audio1.m4a
ffmpeg -i input.mkv -map 0:a:1 -c copy audio2.m4a

# Extract all subtitles
ffmpeg -i input.mkv -map 0:s:0 subtitle1.srt
ffmpeg -i input.mkv -map 0:s:1 subtitle2.srt
```

### Batch Extract Script

```bash
#!/bin/bash
# Extract all streams from a file

INPUT="input.mkv"

# Extract video
ffmpeg -i "$INPUT" -map 0:v:0 -c copy video.mp4

# Extract audio tracks
for i in {0..9}; do
  ffmpeg -i "$INPUT" -map 0:a:$i -c copy "audio_$i.m4a" 2>/dev/null || break
done

# Extract subtitles
for i in {0..9}; do
  ffmpeg -i "$INPUT" -map 0:s:$i "subtitle_$i.srt" 2>/dev/null || break
done
```

---

## Merge Streams

### Merge Video and Audio

```bash
# Merge video and audio (copy both)
ffmpeg -i video.mp4 -i audio.m4a -c copy output.mp4

# Merge with specific mapping
ffmpeg -i video.mp4 -i audio.m4a -map 0:v -map 1:a -c copy output.mp4

# Merge and re-encode audio
ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac -b:a 192k output.mp4
```

### Merge Multiple Audio Tracks

```bash
# Add second audio track
ffmpeg -i video.mp4 -i audio1.m4a -i audio2.m4a \
  -map 0:v -map 1:a -map 2:a -c copy output.mkv

# Set audio track metadata
ffmpeg -i video.mp4 -i audio_eng.m4a -i audio_spa.m4a \
  -map 0:v -map 1:a -map 2:a -c copy \
  -metadata:s:a:0 language=eng -metadata:s:a:0 title="English" \
  -metadata:s:a:1 language=spa -metadata:s:a:1 title="Spanish" \
  output.mkv
```

### Merge with Subtitles

```bash
# Add subtitle track
ffmpeg -i video.mp4 -i subtitle.srt -c copy -c:s mov_text output.mp4

# Add multiple subtitle tracks
ffmpeg -i video.mp4 -i sub_eng.srt -i sub_spa.srt \
  -map 0:v -map 0:a -map 1:s -map 2:s -c copy -c:s mov_text \
  -metadata:s:s:0 language=eng -metadata:s:s:0 title="English" \
  -metadata:s:s:1 language=spa -metadata:s:s:1 title="Spanish" \
  output.mp4

# For MKV (supports more subtitle formats)
ffmpeg -i video.mp4 -i subtitle.srt -c copy -c:s srt output.mkv
```

---

## Replace Streams

### Replace Audio Track

```bash
# Replace audio (keep video)
ffmpeg -i input.mp4 -i new_audio.m4a -map 0:v -map 1:a -c copy output.mp4

# Replace audio and re-sync (delay audio by 0.5 seconds)
ffmpeg -i input.mp4 -i new_audio.m4a -map 0:v -map 1:a \
  -c copy -itsoffset 0.5 output.mp4
```

### Replace Video Track

```bash
# Replace video (keep audio)
ffmpeg -i input.mp4 -i new_video.mp4 -map 1:v -map 0:a -c copy output.mp4
```

### Remove Stream

```bash
# Remove audio
ffmpeg -i input.mp4 -an -c:v copy output.mp4

# Remove video
ffmpeg -i input.mp4 -vn -c:a copy output.m4a

# Remove subtitles
ffmpeg -i input.mkv -map 0 -map -0:s -c copy output.mkv

# Remove specific audio track (keep others)
ffmpeg -i input.mkv -map 0 -map -0:a:1 -c copy output.mkv
```

---

## Advanced Stream Selection

### Select by Stream Index

```bash
# Stream indices start at 0
# Example: input has video(0), audio(1), audio(2), subtitle(3)

# Select specific streams
ffmpeg -i input.mkv -map 0:0 -map 0:2 -c copy output.mkv  # video + audio2

# Select all except one
ffmpeg -i input.mkv -map 0 -map -0:1 -c copy output.mkv  # Remove audio1
```

### Select by Language

```bash
# Extract English audio
ffmpeg -i input.mkv -map 0:v -map 0:m:language:eng -c copy output.mkv

# Extract all English streams (audio + subtitles)
ffmpeg -i input.mkv -map 0:v -map 0:m:language:eng -c copy output.mkv
```

### Select by Codec

```bash
# Select only H.264 video
ffmpeg -i input.mkv -map 0:v:0 -c:v copy output.mp4

# Select only AAC audio
ffmpeg -i input.mkv -map 0:a:0 -c:a copy output.m4a
```

---

## Set Default Stream

### Set Default Audio Track

```bash
# Mark first audio as default
ffmpeg -i input.mkv -map 0 -c copy \
  -disposition:a:0 default -disposition:a:1 0 output.mkv

# Set second audio as default
ffmpeg -i input.mkv -map 0 -c copy \
  -disposition:a:0 0 -disposition:a:1 default output.mkv
```

### Set Default Subtitle Track

```bash
# Mark first subtitle as default
ffmpeg -i input.mkv -map 0 -c copy \
  -disposition:s:0 default -disposition:s:1 0 output.mkv
```

---

## Stream Metadata

### Add Metadata to Streams

```bash
# Set stream titles and languages
ffmpeg -i input.mp4 -i audio.m4a -i subtitle.srt \
  -map 0:v -map 1:a -map 2:s -c copy -c:s mov_text \
  -metadata:s:v:0 title="Main Video" \
  -metadata:s:a:0 language=eng -metadata:s:a:0 title="English Audio" \
  -metadata:s:s:0 language=eng -metadata:s:s:0 title="English Subtitles" \
  output.mp4

# Set global metadata
ffmpeg -i input.mp4 -c copy \
  -metadata title="My Movie" \
  -metadata author="Director Name" \
  -metadata year="2024" \
  output.mp4
```

---

## Extract Chapters

### List Chapters

```bash
# Show chapters
ffprobe -v quiet -print_format json -show_chapters input.mkv

# Extract chapter information
ffprobe -v quiet -show_entries chapter=start_time,end_time:chapter_tags=title \
  -of compact input.mkv
```

### Extract Chapter as Separate File

```bash
# Extract specific chapter (e.g., chapter 2: 120-240 seconds)
ffmpeg -i input.mkv -ss 120 -to 240 -c copy chapter2.mkv
```

---

## Quick Reference

| Task | Command |
|------|---------|
| List streams | `ffprobe -i input.mp4` |
| Extract video | `ffmpeg -i input.mp4 -an -c:v copy video.mp4` |
| Extract audio | `ffmpeg -i input.mp4 -vn -c:a copy audio.m4a` |
| Extract subtitle | `ffmpeg -i input.mkv -map 0:s:0 subtitle.srt` |
| Merge video+audio | `ffmpeg -i video.mp4 -i audio.m4a -c copy output.mp4` |
| Add subtitle | `ffmpeg -i video.mp4 -i sub.srt -c copy -c:s mov_text output.mp4` |
| Remove audio | `ffmpeg -i input.mp4 -an -c:v copy output.mp4` |
| Replace audio | `ffmpeg -i input.mp4 -i new_audio.m4a -map 0:v -map 1:a -c copy output.mp4` |

---

## Tips

- Use `-c copy` to avoid re-encoding (faster, no quality loss)
- Use `-map` for precise stream selection
- MKV supports more stream types than MP4
- Use `-metadata:s:a:0` to set metadata for first audio stream
- Use `-disposition:a:0 default` to mark stream as default
- Check stream indices with `ffprobe` before extraction
- Use `-vn` to remove video, `-an` to remove audio, `-sn` to remove subtitles

