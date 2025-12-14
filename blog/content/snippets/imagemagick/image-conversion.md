---
title: "ImageMagick Image Conversion"
date: 2024-12-12
description: "Format conversion, quality, and compression"
tags: ["imagemagick", "images", "conversion"]
---



## Basic Conversion

### Convert Format

```bash
# Simple conversion
convert input.png output.jpg

# Multiple files
convert input1.png input2.png input3.png output.pdf

# Specify quality
convert input.png -quality 90 output.jpg
```

### Common Formats

```bash
# PNG to JPG
convert image.png image.jpg

# JPG to PNG
convert image.jpg image.png

# Any to WebP
convert image.jpg -quality 80 image.webp

# SVG to PNG (rasterize)
convert -density 300 image.svg image.png

# PDF to images (one per page)
convert -density 150 document.pdf page-%03d.png
```

---

## Quality & Compression

### JPEG Quality

```bash
# High quality (90-100)
convert input.png -quality 95 output.jpg

# Medium quality (75-90)
convert input.png -quality 85 output.jpg

# Low quality (60-75)
convert input.png -quality 70 output.jpg

# Progressive JPEG
convert input.png -quality 85 -interlace Plane output.jpg
```

### PNG Compression

```bash
# Maximum compression (slow)
convert input.png -quality 95 -define png:compression-level=9 output.png

# Fast compression
convert input.png -quality 95 -define png:compression-level=1 output.png

# 8-bit PNG (smaller file)
convert input.png -colors 256 output.png
```

### WebP

```bash
# Lossy WebP
convert input.jpg -quality 80 output.webp

# Lossless WebP
convert input.png -define webp:lossless=true output.webp

# With alpha channel
convert input.png -quality 80 -define webp:alpha-quality=100 output.webp
```

---

## Resizing & Scaling

### Resize by Dimensions

```bash
# Exact size (may distort)
convert input.jpg -resize 800x600! output.jpg

# Fit within dimensions (maintain aspect ratio)
convert input.jpg -resize 800x600 output.jpg

# Resize width only
convert input.jpg -resize 800x output.jpg

# Resize height only
convert input.jpg -resize x600 output.jpg

# Resize by percentage
convert input.jpg -resize 50% output.jpg
```

### Thumbnails

```bash
# Create thumbnail (max 200x200)
convert input.jpg -thumbnail 200x200 thumb.jpg

# Square thumbnail (crop to center)
convert input.jpg -thumbnail 200x200^ -gravity center -extent 200x200 thumb.jpg

# Multiple thumbnails
convert input.jpg -thumbnail 100x100 small.jpg \
                  -thumbnail 200x200 medium.jpg \
                  -thumbnail 400x400 large.jpg
```

### Sampling & Filtering

```bash
# High-quality resize (Lanczos)
convert input.jpg -filter Lanczos -resize 800x600 output.jpg

# Fast resize (Point)
convert input.jpg -filter Point -resize 800x600 output.jpg

# Sharpen after resize
convert input.jpg -resize 800x600 -sharpen 0x1 output.jpg
```

---

## Cropping

### Crop by Dimensions

```bash
# Crop 800x600 from top-left
convert input.jpg -crop 800x600+0+0 output.jpg

# Crop from center
convert input.jpg -gravity center -crop 800x600+0+0 output.jpg

# Crop and remove canvas
convert input.jpg -crop 800x600+100+100 +repage output.jpg
```

### Auto-Crop

```bash
# Remove white borders
convert input.jpg -fuzz 10% -trim +repage output.jpg

# Remove black borders
convert input.jpg -bordercolor black -border 1x1 -fuzz 10% -trim +repage output.jpg
```

### Aspect Ratio Crop

```bash
# Crop to 16:9
convert input.jpg -gravity center -crop 16:9 +repage output.jpg

# Crop to square
convert input.jpg -gravity center -crop 1:1 +repage output.jpg
```

---

## Color Operations

### Adjust Brightness/Contrast

```bash
# Increase brightness
convert input.jpg -modulate 120,100,100 output.jpg

# Increase contrast
convert input.jpg -contrast output.jpg

# Brightness and contrast
convert input.jpg -brightness-contrast 10x20 output.jpg
```

### Color Adjustments

```bash
# Grayscale
convert input.jpg -colorspace Gray output.jpg

# Sepia
convert input.jpg -sepia-tone 80% output.jpg

# Adjust saturation
convert input.jpg -modulate 100,150,100 output.jpg

# Adjust hue
convert input.jpg -modulate 100,100,120 output.jpg
```

### Color Space Conversion

```bash
# RGB to CMYK
convert input.jpg -colorspace CMYK output.jpg

# sRGB to Linear RGB
convert input.jpg -colorspace RGB output.jpg

# Apply ICC profile
convert input.jpg -profile sRGB.icc -profile AdobeRGB.icc output.jpg
```

---

## Transparency & Alpha

### Add/Remove Alpha Channel

```bash
# Add alpha channel
convert input.jpg -alpha on output.png

# Remove alpha (flatten to white)
convert input.png -background white -alpha remove output.jpg

# Flatten to specific color
convert input.png -background "#FF0000" -alpha remove output.jpg
```

### Make Color Transparent

```bash
# Make white transparent
convert input.jpg -transparent white output.png

# Make color range transparent (with fuzz)
convert input.jpg -fuzz 10% -transparent white output.png
```

### Alpha Operations

```bash
# Extract alpha channel
convert input.png -alpha extract alpha.png

# Apply alpha mask
convert input.jpg mask.png -compose CopyOpacity -composite output.png
```

---

## Batch Processing

### Process Multiple Files

```bash
# Convert all JPGs to PNG
mogrify -format png *.jpg

# Resize all images in directory
mogrify -resize 800x600 *.jpg

# Process with new names
for file in *.jpg; do
    convert "$file" -resize 800x600 "resized_$file"
done
```

### Batch with Different Operations

```bash
# Resize and convert
for file in *.png; do
    convert "$file" -resize 800x600 -quality 90 "${file%.png}.jpg"
done

# Create thumbnails
for file in *.jpg; do
    convert "$file" -thumbnail 200x200 "thumb_$file"
done
```

---

## Image Information

### Get Image Info

```bash
# Basic info
identify image.jpg

# Detailed info
identify -verbose image.jpg

# Just dimensions
identify -format "%wx%h" image.jpg

# File size
identify -format "%b" image.jpg
```

---

## Advanced Operations

### Composite Images

```bash
# Overlay image
convert background.jpg overlay.png -gravity center -composite output.jpg

# Blend two images
convert image1.jpg image2.jpg -blend 50x50 output.jpg
```

### Add Border

```bash
# Simple border
convert input.jpg -border 10x10 -bordercolor black output.jpg

# Frame effect
convert input.jpg -mattecolor "#FF0000" -frame 10x10+5+5 output.jpg
```

### Rotate & Flip

```bash
# Rotate 90 degrees
convert input.jpg -rotate 90 output.jpg

# Flip horizontal
convert input.jpg -flop output.jpg

# Flip vertical
convert input.jpg -flip output.jpg

# Auto-orient based on EXIF
convert input.jpg -auto-orient output.jpg
```

---

## Optimization

### Optimize File Size

```bash
# Strip metadata
convert input.jpg -strip output.jpg

# Optimize PNG
convert input.png -strip -define png:compression-level=9 output.png

# Optimize JPEG
convert input.jpg -strip -quality 85 -sampling-factor 4:2:0 output.jpg
```

### Progressive Images

```bash
# Progressive JPEG
convert input.jpg -interlace Plane output.jpg

# Progressive PNG
convert input.png -interlace PNG output.png
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Convert format | `convert input.png output.jpg` |
| Resize | `convert input.jpg -resize 800x600 output.jpg` |
| Thumbnail | `convert input.jpg -thumbnail 200x200 thumb.jpg` |
| Crop | `convert input.jpg -crop 800x600+0+0 output.jpg` |
| Grayscale | `convert input.jpg -colorspace Gray output.jpg` |
| Quality | `convert input.png -quality 90 output.jpg` |
| Strip metadata | `convert input.jpg -strip output.jpg` |
| Batch resize | `mogrify -resize 800x600 *.jpg` |

---

## Tips

- Use `-quality 85-90` for JPEGs (good balance)
- Use `-strip` to remove metadata and reduce file size
- Use `-thumbnail` instead of `-resize` for small images (faster)
- Use `mogrify` for in-place batch processing
- Use `-sampling-factor 4:2:0` for smaller JPEGs
- Add `-interlace Plane` for progressive JPEGs (better web loading)
- Use `-filter Lanczos` for high-quality resizing
- Always use `+repage` after cropping to remove canvas info

