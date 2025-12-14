---
title: "ImageMagick Image Effects"
date: 2024-12-12
description: "Filters, effects, and artistic transformations"
tags: ["imagemagick", "images", "effects", "filters"]
---



## Blur & Sharpen

### Blur

```bash
# Gaussian blur
convert input.jpg -blur 0x8 output.jpg

# Motion blur
convert input.jpg -motion-blur 0x20+45 output.jpg

# Radial blur
convert input.jpg -radial-blur 10 output.jpg
```

### Sharpen

```bash
# Sharpen
convert input.jpg -sharpen 0x1 output.jpg

# Unsharp mask (best for photos)
convert input.jpg -unsharp 0x1+1+0.05 output.jpg
```

## Noise & Texture

```bash
# Add noise
convert input.jpg -attenuate 0.5 +noise Gaussian output.jpg

# Reduce noise
convert input.jpg -enhance output.jpg

# Despeckle
convert input.jpg -despeckle output.jpg
```

## Edge Detection

```bash
# Edge detection
convert input.jpg -edge 1 output.jpg

# Canny edge detection
convert input.jpg -canny 0x1+10%+30% output.jpg

# Sobel edge detection
convert input.jpg -define convolve:scale='!' \
  -morphology Convolve Sobel:0 output.jpg
```

## Artistic Effects

### Oil Paint

```bash
convert input.jpg -paint 5 output.jpg
```

### Charcoal

```bash
convert input.jpg -charcoal 2 output.jpg
```

### Sketch

```bash
convert input.jpg -sketch 0x20+120 output.jpg
```

### Emboss

```bash
convert input.jpg -emboss 2 output.jpg
```

### Posterize

```bash
convert input.jpg -posterize 8 output.jpg
```

## Color Effects

### Tint

```bash
# Blue tint
convert input.jpg -fill blue -colorize 30% output.jpg

# Sepia tone
convert input.jpg -sepia-tone 80% output.jpg
```

### Duotone

```bash
convert input.jpg -colorspace Gray \
  \( +clone -fill "#FF6600" -colorize 100% \) \
  -compose overlay -composite output.jpg
```

### Vignette

```bash
convert input.jpg -background black -vignette 0x20 output.jpg
```

## Distortion & Warping

```bash
# Swirl
convert input.jpg -swirl 90 output.jpg

# Wave
convert input.jpg -wave 10x100 output.jpg

# Implode/Explode
convert input.jpg -implode 0.5 output.jpg
convert input.jpg -implode -0.5 output.jpg
```

## Text & Annotations

```bash
# Add text
convert input.jpg -pointsize 48 -fill white -gravity south \
  -annotate +0+10 'Hello World' output.jpg

# Text with background
convert input.jpg -pointsize 48 -fill white -gravity south \
  -undercolor '#00000080' -annotate +0+10 'Hello World' output.jpg

# Draw shapes
convert input.jpg -fill none -stroke red -strokewidth 5 \
  -draw "rectangle 100,100 300,300" output.jpg
```

## Further Reading

- [ImageMagick Usage Examples](https://imagemagick.org/Usage/)
- [ImageMagick Command-Line Options](https://imagemagick.org/script/command-line-options.php)

