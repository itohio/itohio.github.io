---
title: "FFmpeg Streaming"
date: 2024-12-12
description: "Streaming video from files, cameras, and capture devices"
tags: ["ffmpeg", "streaming", "rtmp", "hls", "camera"]
---

Streaming video and audio using FFmpeg to various protocols and platforms.

---

## Stream to RTMP (YouTube, Twitch, etc.)

### Stream Video File to RTMP

```bash
# Basic RTMP stream
ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -maxrate 3000k \
  -bufsize 6000k -pix_fmt yuv420p -g 50 -c:a aac -b:a 160k -ar 44100 \
  -f flv rtmp://live.twitch.tv/app/YOUR_STREAM_KEY

# YouTube Live
ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -b:v 2500k \
  -maxrate 2500k -bufsize 5000k -pix_fmt yuv420p -g 50 \
  -c:a aac -b:a 128k -ar 44100 -f flv \
  rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY

# Loop video continuously
ffmpeg -re -stream_loop -1 -i input.mp4 -c:v libx264 -preset veryfast \
  -b:v 2500k -maxrate 2500k -bufsize 5000k -pix_fmt yuv420p -g 50 \
  -c:a aac -b:a 128k -f flv rtmp://server/app/key
```

**Key Parameters**:
- `-re`: Read input at native frame rate (real-time)
- `-stream_loop -1`: Loop input indefinitely
- `-g 50`: GOP size (keyframe interval, 2 seconds at 25fps)
- `-preset veryfast`: Fast encoding for live streaming
- `-maxrate` & `-bufsize`: Rate control for stable streaming

---

## Stream from Webcam/Camera

### Linux (V4L2)

```bash
# List available cameras
ffmpeg -f v4l2 -list_formats all -i /dev/video0

# Stream from webcam
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -c:v libx264 -preset veryfast -b:v 2000k -maxrate 2000k -bufsize 4000k \
  -pix_fmt yuv420p -g 60 -f flv rtmp://server/app/key

# Webcam with audio (ALSA)
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -f alsa -i hw:0 -c:v libx264 -preset veryfast -b:v 2000k \
  -c:a aac -b:a 128k -f flv rtmp://server/app/key

# Webcam with PulseAudio
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -f pulse -i default -c:v libx264 -preset veryfast -b:v 2000k \
  -c:a aac -b:a 128k -f flv rtmp://server/app/key
```

### Windows (DirectShow)

```bash
# List devices
ffmpeg -list_devices true -f dshow -i dummy

# Stream from webcam
ffmpeg -f dshow -video_size 1280x720 -framerate 30 -i video="Integrated Camera" \
  -c:v libx264 -preset veryfast -b:v 2000k -maxrate 2000k -bufsize 4000k \
  -pix_fmt yuv420p -g 60 -f flv rtmp://server/app/key

# Webcam with microphone
ffmpeg -f dshow -video_size 1280x720 -framerate 30 \
  -i video="Integrated Camera":audio="Microphone" \
  -c:v libx264 -preset veryfast -b:v 2000k \
  -c:a aac -b:a 128k -f flv rtmp://server/app/key
```

### macOS (AVFoundation)

```bash
# List devices
ffmpeg -f avfoundation -list_devices true -i ""

# Stream from webcam
ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "0:0" \
  -c:v libx264 -preset veryfast -b:v 2000k -maxrate 2000k -bufsize 4000k \
  -pix_fmt yuv420p -g 60 -f flv rtmp://server/app/key

# Webcam with audio (device 0 = video, device 0 = audio)
ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "0:0" \
  -c:v libx264 -preset veryfast -b:v 2000k \
  -c:a aac -b:a 128k -f flv rtmp://server/app/key
```

---

## Screen Capture & Stream

### Linux (X11)

```bash
# Capture entire screen
ffmpeg -f x11grab -framerate 30 -video_size 1920x1080 -i :0.0 \
  -c:v libx264 -preset veryfast -b:v 3000k -maxrate 3000k -bufsize 6000k \
  -pix_fmt yuv420p -g 60 -f flv rtmp://server/app/key

# Capture specific window (get window ID with xwininfo)
ffmpeg -f x11grab -framerate 30 -video_size 1280x720 -i :0.0+100,200 \
  -c:v libx264 -preset veryfast -b:v 2000k -f flv rtmp://server/app/key

# Screen with audio
ffmpeg -f x11grab -framerate 30 -video_size 1920x1080 -i :0.0 \
  -f pulse -i default -c:v libx264 -preset veryfast -b:v 3000k \
  -c:a aac -b:a 128k -f flv rtmp://server/app/key
```

### Windows (GDI)

```bash
# Capture desktop
ffmpeg -f gdigrab -framerate 30 -i desktop \
  -c:v libx264 -preset veryfast -b:v 3000k -maxrate 3000k -bufsize 6000k \
  -pix_fmt yuv420p -g 60 -f flv rtmp://server/app/key

# Capture specific window
ffmpeg -f gdigrab -framerate 30 -i title="Window Title" \
  -c:v libx264 -preset veryfast -b:v 2000k -f flv rtmp://server/app/key
```

### macOS (AVFoundation)

```bash
# Capture screen (device 1 is usually screen capture)
ffmpeg -f avfoundation -framerate 30 -i "1" \
  -c:v libx264 -preset veryfast -b:v 3000k -maxrate 3000k -bufsize 6000k \
  -pix_fmt yuv420p -g 60 -f flv rtmp://server/app/key
```

---

## HLS Streaming (HTTP Live Streaming)

### Generate HLS Stream

```bash
# Basic HLS
ffmpeg -re -i input.mp4 -c:v libx264 -c:a aac -f hls \
  -hls_time 4 -hls_list_size 5 -hls_flags delete_segments \
  output.m3u8

# HLS with multiple quality levels (adaptive bitrate)
ffmpeg -re -i input.mp4 \
  -map 0:v -map 0:a -map 0:v -map 0:a -map 0:v -map 0:a \
  -c:v libx264 -c:a aac \
  -b:v:0 1000k -s:v:0 640x360 \
  -b:v:1 2500k -s:v:1 1280x720 \
  -b:v:2 5000k -s:v:2 1920x1080 \
  -b:a:0 96k -b:a:1 128k -b:a:2 192k \
  -var_stream_map "v:0,a:0 v:1,a:1 v:2,a:2" \
  -master_pl_name master.m3u8 \
  -f hls -hls_time 6 -hls_list_size 10 \
  -hls_segment_filename "stream_%v/data%03d.ts" \
  stream_%v/index.m3u8

# HLS from webcam (Linux)
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -c:v libx264 -preset veryfast -b:v 2000k -c:a aac -b:a 128k \
  -f hls -hls_time 4 -hls_list_size 5 output.m3u8
```

**HLS Parameters**:
- `-hls_time`: Segment duration in seconds
- `-hls_list_size`: Number of segments in playlist
- `-hls_flags delete_segments`: Delete old segments
- `-hls_segment_filename`: Pattern for segment files

---

## DASH Streaming

```bash
# Generate DASH stream
ffmpeg -re -i input.mp4 -c:v libx264 -c:a aac \
  -f dash -seg_duration 4 -use_template 1 -use_timeline 1 \
  output.mpd

# DASH with multiple bitrates
ffmpeg -re -i input.mp4 \
  -map 0:v -map 0:a -map 0:v -map 0:a \
  -c:v libx264 -c:a aac \
  -b:v:0 1000k -s:v:0 640x360 \
  -b:v:1 2500k -s:v:1 1280x720 \
  -b:a:0 96k -b:a:1 128k \
  -f dash -seg_duration 4 -use_template 1 -use_timeline 1 \
  -adaptation_sets "id=0,streams=v id=1,streams=a" \
  output.mpd
```

---

## UDP/RTP Streaming

```bash
# Stream to UDP
ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -b:v 2000k \
  -c:a aac -b:a 128k -f mpegts udp://192.168.1.100:1234

# Stream with SDP file (for VLC playback)
ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -b:v 2000k \
  -c:a aac -b:a 128k -f rtp rtp://192.168.1.100:1234 \
  -sdp_file stream.sdp

# Multicast streaming
ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -b:v 2000k \
  -c:a aac -b:a 128k -f mpegts udp://239.255.0.1:1234?ttl=1
```

---

## RTSP Server

```bash
# Stream to RTSP (requires RTSP server like MediaMTX/rtsp-simple-server)
ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -b:v 2000k \
  -c:a aac -b:a 128k -f rtsp rtsp://localhost:8554/mystream

# From webcam to RTSP
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -c:v libx264 -preset veryfast -b:v 2000k -c:a aac -b:a 128k \
  -f rtsp rtsp://localhost:8554/webcam
```

---

## IP Camera Streaming

### Receive from IP Camera (RTSP)

```bash
# View RTSP stream
ffplay rtsp://username:password@192.168.1.100:554/stream

# Re-stream IP camera to RTMP
ffmpeg -rtsp_transport tcp -i rtsp://username:password@192.168.1.100:554/stream \
  -c:v copy -c:a copy -f flv rtmp://server/app/key

# Re-stream with transcoding
ffmpeg -rtsp_transport tcp -i rtsp://username:password@192.168.1.100:554/stream \
  -c:v libx264 -preset veryfast -b:v 2000k -c:a aac -b:a 128k \
  -f flv rtmp://server/app/key
```

---

## Low-Latency Streaming

### SRT (Secure Reliable Transport)

```bash
# SRT server (listener)
ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -tune zerolatency \
  -b:v 2000k -c:a aac -b:a 128k -f mpegts srt://0.0.0.0:9000?mode=listener

# SRT client (caller)
ffmpeg -i srt://192.168.1.100:9000?mode=caller -c copy output.mp4

# Webcam to SRT
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -c:v libx264 -preset veryfast -tune zerolatency -b:v 2000k \
  -c:a aac -b:a 128k -f mpegts srt://0.0.0.0:9000?mode=listener
```

---

## Platform-Specific Recommendations

### Twitch

```bash
ffmpeg -re -i input.mp4 \
  -c:v libx264 -preset veryfast -b:v 3000k -maxrate 3000k -bufsize 6000k \
  -pix_fmt yuv420p -g 50 -r 30 \
  -c:a aac -b:a 160k -ar 44100 \
  -f flv rtmp://live.twitch.tv/app/YOUR_STREAM_KEY
```

**Twitch Recommendations**:
- Bitrate: 3000-6000 kbps
- Resolution: 1920x1080 or 1280x720
- Frame rate: 30 or 60 fps
- Keyframe interval: 2 seconds

### YouTube Live

```bash
ffmpeg -re -i input.mp4 \
  -c:v libx264 -preset veryfast -b:v 2500k -maxrate 2500k -bufsize 5000k \
  -pix_fmt yuv420p -g 60 -r 30 \
  -c:a aac -b:a 128k -ar 44100 \
  -f flv rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY
```

**YouTube Recommendations**:
- Bitrate: 2500-4500 kbps (1080p30)
- Resolution: 1920x1080
- Frame rate: 30 fps
- Keyframe interval: 2 seconds

### Facebook Live

```bash
ffmpeg -re -i input.mp4 \
  -c:v libx264 -preset veryfast -b:v 4000k -maxrate 4000k -bufsize 8000k \
  -pix_fmt yuv420p -g 60 -r 30 \
  -c:a aac -b:a 128k -ar 44100 \
  -f flv rtmps://live-api-s.facebook.com:443/rtmp/YOUR_STREAM_KEY
```

---

## Monitoring & Testing

### Test Stream Locally

```bash
# Start local RTMP server (using nginx-rtmp or similar)
# Then stream to it
ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -b:v 2000k \
  -c:a aac -b:a 128k -f flv rtmp://localhost/live/test

# Play with ffplay
ffplay rtmp://localhost/live/test

# Play with VLC
vlc rtmp://localhost/live/test
```

### Stream Health Check

```bash
# Show stream statistics
ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -b:v 2000k \
  -c:a aac -b:a 128k -f flv rtmp://server/app/key \
  -progress pipe:1 -v verbose
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Stream file to RTMP | `ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -b:v 2500k -c:a aac -f flv rtmp://server/app/key` |
| Webcam to RTMP (Linux) | `ffmpeg -f v4l2 -i /dev/video0 -c:v libx264 -preset veryfast -b:v 2000k -c:a aac -f flv rtmp://server/app/key` |
| Screen capture (Linux) | `ffmpeg -f x11grab -i :0.0 -c:v libx264 -preset veryfast -b:v 3000k -f flv rtmp://server/app/key` |
| Generate HLS | `ffmpeg -re -i input.mp4 -c:v libx264 -c:a aac -f hls -hls_time 4 output.m3u8` |
| UDP stream | `ffmpeg -re -i input.mp4 -c:v libx264 -b:v 2000k -f mpegts udp://192.168.1.100:1234` |
| Loop video | `ffmpeg -re -stream_loop -1 -i input.mp4 -c:v libx264 -f flv rtmp://server/app/key` |

---

## Tips

- Use `-re` for real-time streaming (matches input frame rate)
- Use `-preset veryfast` or `-preset ultrafast` for live encoding
- Use `-tune zerolatency` for minimal latency
- Set `-g` (GOP size) to 2x frame rate for 2-second keyframe interval
- Use `-maxrate` and `-bufsize` for stable bitrate
- Test locally before streaming to production
- Monitor CPU usage - hardware encoding helps for high resolutions

