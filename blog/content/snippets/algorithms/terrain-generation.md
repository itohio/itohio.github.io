---
title: "Terrain & Heightmap Generation"
date: 2025-12-13
tags: ["terrain", "heightmap", "procedural-generation", "perlin-noise", "algorithms"]
---

Procedural terrain generation creates realistic landscapes using mathematical algorithms. Common techniques include Perlin noise, Diamond-Square, and hydraulic erosion.

## Heightmap Basics

A heightmap is a 2D array where each value represents elevation:

```go
type Heightmap struct {
    width, height int
    data          [][]float64
}

func NewHeightmap(width, height int) *Heightmap {
    data := make([][]float64, height)
    for i := range data {
        data[i] = make([]float64, width)
    }
    
    return &Heightmap{
        width:  width,
        height: height,
        data:   data,
    }
}

func (hm *Heightmap) Get(x, y int) float64 {
    if x < 0 || x >= hm.width || y < 0 || y >= hm.height {
        return 0
    }
    return hm.data[y][x]
}

func (hm *Heightmap) Set(x, y int, value float64) {
    if x >= 0 && x < hm.width && y >= 0 && y < hm.height {
        hm.data[y][x] = value
    }
}

// Normalize to [0, 1]
func (hm *Heightmap) Normalize() {
    min, max := math.MaxFloat64, -math.MaxFloat64
    
    for y := 0; y < hm.height; y++ {
        for x := 0; x < hm.width; x++ {
            if hm.data[y][x] < min {
                min = hm.data[y][x]
            }
            if hm.data[y][x] > max {
                max = hm.data[y][x]
            }
        }
    }
    
    rangeVal := max - min
    if rangeVal == 0 {
        return
    }
    
    for y := 0; y < hm.height; y++ {
        for x := 0; x < hm.width; x++ {
            hm.data[y][x] = (hm.data[y][x] - min) / rangeVal
        }
    }
}
```

## 1. Perlin Noise

Classic noise function for natural-looking terrain.

```go
import "math"

type PerlinNoise struct {
    permutation []int
}

func NewPerlinNoise(seed int64) *PerlinNoise {
    rand.Seed(seed)
    
    // Generate permutation table
    perm := make([]int, 512)
    p := make([]int, 256)
    for i := range p {
        p[i] = i
    }
    
    // Shuffle
    for i := 255; i > 0; i-- {
        j := rand.Intn(i + 1)
        p[i], p[j] = p[j], p[i]
    }
    
    // Duplicate for overflow
    for i := 0; i < 256; i++ {
        perm[i] = p[i]
        perm[i+256] = p[i]
    }
    
    return &PerlinNoise{permutation: perm}
}

func (pn *PerlinNoise) Noise2D(x, y float64) float64 {
    // Find unit grid cell
    X := int(math.Floor(x)) & 255
    Y := int(math.Floor(y)) & 255
    
    // Relative position in cell
    x -= math.Floor(x)
    y -= math.Floor(y)
    
    // Fade curves
    u := fade(x)
    v := fade(y)
    
    // Hash coordinates of 4 corners
    aa := pn.permutation[pn.permutation[X]+Y]
    ab := pn.permutation[pn.permutation[X]+Y+1]
    ba := pn.permutation[pn.permutation[X+1]+Y]
    bb := pn.permutation[pn.permutation[X+1]+Y+1]
    
    // Blend results from 4 corners
    return lerp(v,
        lerp(u, grad2D(aa, x, y), grad2D(ba, x-1, y)),
        lerp(u, grad2D(ab, x, y-1), grad2D(bb, x-1, y-1)))
}

func fade(t float64) float64 {
    return t * t * t * (t*(t*6-15) + 10)
}

func lerp(t, a, b float64) float64 {
    return a + t*(b-a)
}

func grad2D(hash int, x, y float64) float64 {
    h := hash & 3
    u := x
    v := y
    
    if h == 0 {
        return u + v
    } else if h == 1 {
        return -u + v
    } else if h == 2 {
        return u - v
    }
    return -u - v
}

// Generate heightmap using Perlin noise
func GeneratePerlinHeightmap(width, height int, seed int64, scale, octaves int, persistence, lacunarity float64) *Heightmap {
    hm := NewHeightmap(width, height)
    pn := NewPerlinNoise(seed)
    
    for y := 0; y < height; y++ {
        for x := 0; x < width; x++ {
            amplitude := 1.0
            frequency := 1.0
            noiseHeight := 0.0
            
            // Combine multiple octaves
            for i := 0; i < octaves; i++ {
                sampleX := float64(x) / float64(scale) * frequency
                sampleY := float64(y) / float64(scale) * frequency
                
                perlinValue := pn.Noise2D(sampleX, sampleY)
                noiseHeight += perlinValue * amplitude
                
                amplitude *= persistence
                frequency *= lacunarity
            }
            
            hm.Set(x, y, noiseHeight)
        }
    }
    
    hm.Normalize()
    return hm
}
```

**Parameters**:
- **Scale**: Controls zoom level (larger = smoother)
- **Octaves**: Number of noise layers (more = more detail)
- **Persistence**: Amplitude decrease per octave (0.5 typical)
- **Lacunarity**: Frequency increase per octave (2.0 typical)

## 2. Diamond-Square Algorithm

Classic fractal terrain generation.

```go
func GenerateDiamondSquare(size int, roughness float64, seed int64) *Heightmap {
    // Size must be 2^n + 1
    if !isPowerOfTwoPlusOne(size) {
        panic("Size must be 2^n + 1")
    }
    
    rand.Seed(seed)
    hm := NewHeightmap(size, size)
    
    // Initialize corners
    hm.Set(0, 0, rand.Float64())
    hm.Set(size-1, 0, rand.Float64())
    hm.Set(0, size-1, rand.Float64())
    hm.Set(size-1, size-1, rand.Float64())
    
    stepSize := size - 1
    scale := 1.0
    
    for stepSize > 1 {
        halfStep := stepSize / 2
        
        // Diamond step
        for y := halfStep; y < size; y += stepSize {
            for x := halfStep; x < size; x += stepSize {
                avg := (hm.Get(x-halfStep, y-halfStep) +
                       hm.Get(x+halfStep, y-halfStep) +
                       hm.Get(x-halfStep, y+halfStep) +
                       hm.Get(x+halfStep, y+halfStep)) / 4.0
                
                hm.Set(x, y, avg+randomRange(-scale, scale))
            }
        }
        
        // Square step
        for y := 0; y < size; y += halfStep {
            for x := (y + halfStep) % stepSize; x < size; x += stepSize {
                sum := 0.0
                count := 0
                
                if y-halfStep >= 0 {
                    sum += hm.Get(x, y-halfStep)
                    count++
                }
                if y+halfStep < size {
                    sum += hm.Get(x, y+halfStep)
                    count++
                }
                if x-halfStep >= 0 {
                    sum += hm.Get(x-halfStep, y)
                    count++
                }
                if x+halfStep < size {
                    sum += hm.Get(x+halfStep, y)
                    count++
                }
                
                hm.Set(x, y, sum/float64(count)+randomRange(-scale, scale))
            }
        }
        
        stepSize /= 2
        scale *= math.Pow(2, -roughness)
    }
    
    hm.Normalize()
    return hm
}

func isPowerOfTwoPlusOne(n int) bool {
    n--
    return n > 0 && (n&(n-1)) == 0
}

func randomRange(min, max float64) float64 {
    return min + rand.Float64()*(max-min)
}
```

**Roughness**: Controls terrain variation (0.5-1.0 typical)

## 3. Simplex Noise (Improved Perlin)

More efficient and fewer artifacts than Perlin.

```go
type SimplexNoise struct {
    perm []int
}

func NewSimplexNoise(seed int64) *SimplexNoise {
    rand.Seed(seed)
    
    perm := make([]int, 512)
    p := make([]int, 256)
    for i := range p {
        p[i] = i
    }
    
    for i := 255; i > 0; i-- {
        j := rand.Intn(i + 1)
        p[i], p[j] = p[j], p[i]
    }
    
    for i := 0; i < 256; i++ {
        perm[i] = p[i]
        perm[i+256] = p[i]
    }
    
    return &SimplexNoise{perm: perm}
}

func (sn *SimplexNoise) Noise2D(x, y float64) float64 {
    const F2 = 0.5 * (math.Sqrt(3.0) - 1.0)
    const G2 = (3.0 - math.Sqrt(3.0)) / 6.0
    
    // Skew input space
    s := (x + y) * F2
    i := int(math.Floor(x + s))
    j := int(math.Floor(y + s))
    
    t := float64(i+j) * G2
    X0 := float64(i) - t
    Y0 := float64(j) - t
    x0 := x - X0
    y0 := y - Y0
    
    // Determine simplex
    var i1, j1 int
    if x0 > y0 {
        i1, j1 = 1, 0
    } else {
        i1, j1 = 0, 1
    }
    
    x1 := x0 - float64(i1) + G2
    y1 := y0 - float64(j1) + G2
    x2 := x0 - 1.0 + 2.0*G2
    y2 := y0 - 1.0 + 2.0*G2
    
    // Calculate contributions
    n0, n1, n2 := 0.0, 0.0, 0.0
    
    t0 := 0.5 - x0*x0 - y0*y0
    if t0 > 0 {
        t0 *= t0
        gi := sn.perm[(i+sn.perm[j&255])&255] % 12
        n0 = t0 * t0 * dot2D(grad3[gi], x0, y0)
    }
    
    t1 := 0.5 - x1*x1 - y1*y1
    if t1 > 0 {
        t1 *= t1
        gi := sn.perm[(i+i1+sn.perm[(j+j1)&255])&255] % 12
        n1 = t1 * t1 * dot2D(grad3[gi], x1, y1)
    }
    
    t2 := 0.5 - x2*x2 - y2*y2
    if t2 > 0 {
        t2 *= t2
        gi := sn.perm[(i+1+sn.perm[(j+1)&255])&255] % 12
        n2 = t2 * t2 * dot2D(grad3[gi], x2, y2)
    }
    
    return 70.0 * (n0 + n1 + n2)
}

var grad3 = [][3]float64{
    {1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0},
    {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
    {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1},
}

func dot2D(g [3]float64, x, y float64) float64 {
    return g[0]*x + g[1]*y
}
```

## 4. Hydraulic Erosion

Simulate water erosion for realistic terrain.

```go
func ApplyHydraulicErosion(hm *Heightmap, iterations int, rainAmount, evaporation, erosionRate, depositionRate float64) {
    water := NewHeightmap(hm.width, hm.height)
    sediment := NewHeightmap(hm.width, hm.height)
    
    for iter := 0; iter < iterations; iter++ {
        // Add rain
        for y := 0; y < hm.height; y++ {
            for x := 0; x < hm.width; x++ {
                water.data[y][x] += rainAmount
            }
        }
        
        // Flow water
        for y := 0; y < hm.height; y++ {
            for x := 0; x < hm.width; x++ {
                if water.Get(x, y) <= 0 {
                    continue
                }
                
                // Find lowest neighbor
                lowestHeight := hm.Get(x, y)
                lowestX, lowestY := x, y
                
                neighbors := [][2]int{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}
                for _, n := range neighbors {
                    nx, ny := x+n[0], y+n[1]
                    if nx >= 0 && nx < hm.width && ny >= 0 && ny < hm.height {
                        neighborHeight := hm.Get(nx, ny)
                        if neighborHeight < lowestHeight {
                            lowestHeight = neighborHeight
                            lowestX, lowestY = nx, ny
                        }
                    }
                }
                
                // Calculate height difference
                heightDiff := hm.Get(x, y) - lowestHeight
                
                if heightDiff > 0 {
                    // Erode
                    erosionAmount := math.Min(heightDiff, water.Get(x, y)*erosionRate)
                    hm.data[y][x] -= erosionAmount
                    sediment.data[y][x] += erosionAmount
                    
                    // Move water and sediment
                    if lowestX != x || lowestY != y {
                        waterToMove := water.Get(x, y) * 0.5
                        water.data[y][x] -= waterToMove
                        water.data[lowestY][lowestX] += waterToMove
                        
                        sedimentToMove := sediment.Get(x, y) * 0.5
                        sediment.data[y][x] -= sedimentToMove
                        sediment.data[lowestY][lowestX] += sedimentToMove
                    }
                } else {
                    // Deposit sediment
                    depositAmount := sediment.Get(x, y) * depositionRate
                    hm.data[y][x] += depositAmount
                    sediment.data[y][x] -= depositAmount
                }
            }
        }
        
        // Evaporate water
        for y := 0; y < hm.height; y++ {
            for x := 0; x < hm.width; x++ {
                water.data[y][x] *= (1.0 - evaporation)
            }
        }
    }
}
```

## 5. Thermal Erosion

Simulate rock/soil sliding down slopes.

```go
func ApplyThermalErosion(hm *Heightmap, iterations int, talusAngle float64) {
    for iter := 0; iter < iterations; iter++ {
        for y := 0; y < hm.height; y++ {
            for x := 0; x < hm.width; x++ {
                currentHeight := hm.Get(x, y)
                maxDiff := 0.0
                targetX, targetY := x, y
                
                // Find steepest neighbor
                neighbors := [][2]int{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}
                for _, n := range neighbors {
                    nx, ny := x+n[0], y+n[1]
                    if nx >= 0 && nx < hm.width && ny >= 0 && ny < hm.height {
                        diff := currentHeight - hm.Get(nx, ny)
                        if diff > maxDiff {
                            maxDiff = diff
                            targetX, targetY = nx, ny
                        }
                    }
                }
                
                // If slope exceeds talus angle, move material
                if maxDiff > talusAngle {
                    amount := 0.5 * (maxDiff - talusAngle)
                    hm.data[y][x] -= amount
                    hm.data[targetY][targetX] += amount
                }
            }
        }
    }
}
```

## Complete Terrain Generation Pipeline

```go
func GenerateRealisticTerrain(width, height int, seed int64) *Heightmap {
    // 1. Base terrain with Perlin noise
    hm := GeneratePerlinHeightmap(width, height, seed, 100, 6, 0.5, 2.0)
    
    // 2. Add mountain ranges with higher frequency
    mountains := GeneratePerlinHeightmap(width, height, seed+1, 50, 4, 0.6, 2.5)
    for y := 0; y < height; y++ {
        for x := 0; x < width; x++ {
            hm.data[y][x] = hm.data[y][x]*0.7 + mountains.data[y][x]*0.3
        }
    }
    
    // 3. Apply hydraulic erosion
    ApplyHydraulicErosion(hm, 50, 0.01, 0.5, 0.3, 0.1)
    
    // 4. Apply thermal erosion
    ApplyThermalErosion(hm, 10, 0.1)
    
    // 5. Normalize
    hm.Normalize()
    
    return hm
}
```

## Biome Assignment

```go
type Biome int

const (
    Ocean Biome = iota
    Beach
    Plains
    Forest
    Hills
    Mountains
    Snow
)

func AssignBiomes(hm *Heightmap, moisture *Heightmap) [][]Biome {
    biomes := make([][]Biome, hm.height)
    for i := range biomes {
        biomes[i] = make([]Biome, hm.width)
    }
    
    for y := 0; y < hm.height; y++ {
        for x := 0; x < hm.width; x++ {
            height := hm.Get(x, y)
            m := moisture.Get(x, y)
            
            if height < 0.3 {
                biomes[y][x] = Ocean
            } else if height < 0.35 {
                biomes[y][x] = Beach
            } else if height < 0.5 {
                if m < 0.3 {
                    biomes[y][x] = Plains
                } else {
                    biomes[y][x] = Forest
                }
            } else if height < 0.7 {
                biomes[y][x] = Hills
            } else if height < 0.85 {
                biomes[y][x] = Mountains
            } else {
                biomes[y][x] = Snow
            }
        }
    }
    
    return biomes
}
```

## Export to Image

```go
import "image"
import "image/color"
import "image/png"

func (hm *Heightmap) ToGrayscale() *image.Gray {
    img := image.NewGray(image.Rect(0, 0, hm.width, hm.height))
    
    for y := 0; y < hm.height; y++ {
        for x := 0; x < hm.width; x++ {
            value := uint8(hm.Get(x, y) * 255)
            img.SetGray(x, y, color.Gray{Y: value})
        }
    }
    
    return img
}

func (hm *Heightmap) ToColoredTerrain() *image.RGBA {
    img := image.NewRGBA(image.Rect(0, 0, hm.width, hm.height))
    
    for y := 0; y < hm.height; y++ {
        for x := 0; x < hm.width; x++ {
            h := hm.Get(x, y)
            var c color.RGBA
            
            if h < 0.3 {
                c = color.RGBA{30, 144, 255, 255} // Deep water
            } else if h < 0.35 {
                c = color.RGBA{238, 214, 175, 255} // Beach
            } else if h < 0.5 {
                c = color.RGBA{34, 139, 34, 255} // Grass
            } else if h < 0.7 {
                c = color.RGBA{107, 142, 35, 255} // Hills
            } else if h < 0.85 {
                c = color.RGBA{139, 137, 137, 255} // Mountains
            } else {
                c = color.RGBA{255, 250, 250, 255} // Snow
            }
            
            img.Set(x, y, c)
        }
    }
    
    return img
}

func SaveHeightmap(hm *Heightmap, filename string) error {
    img := hm.ToColoredTerrain()
    
    f, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer f.Close()
    
    return png.Encode(f, img)
}
```

## Applications

1. **Game Terrain**: Open-world games, flight simulators
2. **Map Generation**: Strategy games, roguelikes
3. **Visualization**: Geographic data, scientific simulations
4. **Art**: Procedural landscapes, wallpapers

## When to Use Each Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| Perlin Noise | Natural terrain | Fast, organic | Can look too smooth |
| Diamond-Square | Rough terrain | Simple, fast | Square artifacts |
| Simplex Noise | High quality | No artifacts, fast | More complex |
| Hydraulic Erosion | Realism | Very realistic | Slow |
| Thermal Erosion | Rocky terrain | Natural slopes | Requires base terrain |

✅ **Combine multiple methods** for best results!

