---
title: "Wave Function Collapse"
date: 2025-12-13
tags: ["wave-function-collapse", "procedural-generation", "algorithms", "game-dev"]
---

**Wave Function Collapse (WFC)** is a procedural generation algorithm that creates patterns by propagating constraints. Inspired by quantum mechanics, it generates coherent outputs from a set of tiles and their adjacency rules.

## Core Concept

1. Start with a grid where each cell can be any tile (superposition)
2. Collapse one cell to a specific tile (observation)
3. Propagate constraints to neighbors (wave propagation)
4. Repeat until all cells are collapsed

## Basic Structure

```go
type Tile struct {
    ID      int
    Pattern [][]int // Visual pattern
}

type Cell struct {
    Collapsed bool
    Options   []int // Possible tile IDs
    Entropy   float64
}

type WFC struct {
    width, height int
    grid          [][]Cell
    tiles         []Tile
    rules         map[int]map[Direction][]int // Adjacency rules
}

type Direction int

const (
    North Direction = iota
    East
    South
    West
)

func NewWFC(width, height int, tiles []Tile) *WFC {
    wfc := &WFC{
        width:  width,
        height: height,
        tiles:  tiles,
        rules:  make(map[int]map[Direction][]int),
        grid:   make([][]Cell, height),
    }
    
    // Initialize grid with all possibilities
    for y := 0; y < height; y++ {
        wfc.grid[y] = make([]Cell, width)
        for x := 0; x < width; x++ {
            options := make([]int, len(tiles))
            for i := range tiles {
                options[i] = i
            }
            wfc.grid[y][x] = Cell{
                Collapsed: false,
                Options:   options,
                Entropy:   float64(len(tiles)),
            }
        }
    }
    
    return wfc
}
```

## Adjacency Rules

Define which tiles can be adjacent to each other:

```go
func (wfc *WFC) AddRule(tileID int, direction Direction, allowedNeighbors []int) {
    if wfc.rules[tileID] == nil {
        wfc.rules[tileID] = make(map[Direction][]int)
    }
    wfc.rules[tileID][direction] = allowedNeighbors
}

// Example: Define tile compatibility
func SetupSimpleRules(wfc *WFC) {
    // Tile 0: Grass
    // Tile 1: Water
    // Tile 2: Sand (transition)
    
    // Grass can be next to grass or sand
    wfc.AddRule(0, North, []int{0, 2})
    wfc.AddRule(0, East, []int{0, 2})
    wfc.AddRule(0, South, []int{0, 2})
    wfc.AddRule(0, West, []int{0, 2})
    
    // Water can be next to water or sand
    wfc.AddRule(1, North, []int{1, 2})
    wfc.AddRule(1, East, []int{1, 2})
    wfc.AddRule(1, South, []int{1, 2})
    wfc.AddRule(1, West, []int{1, 2})
    
    // Sand can be next to anything
    wfc.AddRule(2, North, []int{0, 1, 2})
    wfc.AddRule(2, East, []int{0, 1, 2})
    wfc.AddRule(2, South, []int{0, 1, 2})
    wfc.AddRule(2, West, []int{0, 1, 2})
}
```

## Core Algorithm

### 1. Find Lowest Entropy Cell

```go
func (wfc *WFC) FindLowestEntropyCell() (int, int, bool) {
    minEntropy := math.MaxFloat64
    var minX, minY int
    found := false
    
    for y := 0; y < wfc.height; y++ {
        for x := 0; x < wfc.width; x++ {
            cell := &wfc.grid[y][x]
            
            if !cell.Collapsed && cell.Entropy < minEntropy && len(cell.Options) > 0 {
                // Add small random noise to break ties
                entropy := cell.Entropy + rand.Float64()*0.1
                if entropy < minEntropy {
                    minEntropy = entropy
                    minX, minY = x, y
                    found = true
                }
            }
        }
    }
    
    return minX, minY, found
}
```

### 2. Collapse Cell

```go
func (wfc *WFC) CollapseCell(x, y int) bool {
    cell := &wfc.grid[y][x]
    
    if len(cell.Options) == 0 {
        return false // Contradiction!
    }
    
    // Choose random option (can be weighted)
    chosenTile := cell.Options[rand.Intn(len(cell.Options))]
    
    cell.Options = []int{chosenTile}
    cell.Collapsed = true
    cell.Entropy = 0
    
    return true
}
```

### 3. Propagate Constraints

```go
func (wfc *WFC) Propagate(startX, startY int) bool {
    stack := []struct{ x, y int }{{startX, startY}}
    
    for len(stack) > 0 {
        // Pop from stack
        curr := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        x, y := curr.x, curr.y
        cell := &wfc.grid[y][x]
        
        // Check all four neighbors
        neighbors := []struct {
            x, y int
            dir  Direction
        }{
            {x, y - 1, North},
            {x + 1, y, East},
            {x, y + 1, South},
            {x - 1, y, West},
        }
        
        for _, n := range neighbors {
            if n.x < 0 || n.x >= wfc.width || n.y < 0 || n.y >= wfc.height {
                continue
            }
            
            neighbor := &wfc.grid[n.y][n.x]
            if neighbor.Collapsed {
                continue
            }
            
            // Constrain neighbor based on current cell's options
            if wfc.ConstrainCell(n.x, n.y, x, y, n.dir) {
                // If neighbor changed, add to stack
                stack = append(stack, struct{ x, y int }{n.x, n.y})
            }
        }
    }
    
    return true
}

func (wfc *WFC) ConstrainCell(x, y, sourceX, sourceY int, direction Direction) bool {
    cell := &wfc.grid[y][x]
    sourceCell := &wfc.grid[sourceY][sourceX]
    
    // Get opposite direction
    oppositeDir := wfc.OppositeDirection(direction)
    
    // Collect all valid options
    validOptions := make(map[int]bool)
    for _, sourceTileID := range sourceCell.Options {
        if allowedNeighbors, exists := wfc.rules[sourceTileID][direction]; exists {
            for _, neighborID := range allowedNeighbors {
                validOptions[neighborID] = true
            }
        }
    }
    
    // Filter cell's options
    newOptions := []int{}
    for _, option := range cell.Options {
        if validOptions[option] {
            newOptions = append(newOptions, option)
        }
    }
    
    // Check if options changed
    if len(newOptions) != len(cell.Options) {
        cell.Options = newOptions
        cell.Entropy = float64(len(newOptions))
        
        if len(newOptions) == 0 {
            return false // Contradiction!
        }
        
        return true // Changed
    }
    
    return false // No change
}

func (wfc *WFC) OppositeDirection(dir Direction) Direction {
    switch dir {
    case North:
        return South
    case South:
        return North
    case East:
        return West
    case West:
        return East
    }
    return North
}
```

### 4. Main Generation Loop

```go
func (wfc *WFC) Generate() bool {
    for {
        // Find cell with lowest entropy
        x, y, found := wfc.FindLowestEntropyCell()
        if !found {
            // All cells collapsed!
            return true
        }
        
        // Collapse the cell
        if !wfc.CollapseCell(x, y) {
            return false // Contradiction
        }
        
        // Propagate constraints
        if !wfc.Propagate(x, y) {
            return false // Contradiction
        }
    }
}

// Generate with retry on failure
func (wfc *WFC) GenerateWithRetry(maxAttempts int) bool {
    for attempt := 0; attempt < maxAttempts; attempt++ {
        // Reset grid
        wfc.Reset()
        
        if wfc.Generate() {
            return true
        }
    }
    return false
}

func (wfc *WFC) Reset() {
    for y := 0; y < wfc.height; y++ {
        for x := 0; x < wfc.width; x++ {
            options := make([]int, len(wfc.tiles))
            for i := range wfc.tiles {
                options[i] = i
            }
            wfc.grid[y][x] = Cell{
                Collapsed: false,
                Options:   options,
                Entropy:   float64(len(wfc.tiles)),
            }
        }
    }
}
```

## Advanced Features

### Weighted Tile Selection

```go
type WeightedTile struct {
    Tile
    Weight float64
}

func (wfc *WFC) CollapseCellWeighted(x, y int, weights map[int]float64) bool {
    cell := &wfc.grid[y][x]
    
    if len(cell.Options) == 0 {
        return false
    }
    
    // Calculate total weight
    totalWeight := 0.0
    for _, tileID := range cell.Options {
        totalWeight += weights[tileID]
    }
    
    // Choose weighted random
    r := rand.Float64() * totalWeight
    cumulative := 0.0
    
    for _, tileID := range cell.Options {
        cumulative += weights[tileID]
        if r <= cumulative {
            cell.Options = []int{tileID}
            cell.Collapsed = true
            cell.Entropy = 0
            return true
        }
    }
    
    return false
}
```

### Seeded Generation

```go
func (wfc *WFC) SetSeed(x, y, tileID int) {
    cell := &wfc.grid[y][x]
    cell.Options = []int{tileID}
    cell.Collapsed = true
    cell.Entropy = 0
    wfc.Propagate(x, y)
}

// Example: Force water in center
func GenerateWithLake(wfc *WFC) {
    centerX := wfc.width / 2
    centerY := wfc.height / 2
    
    // Seed a 3x3 lake
    for dy := -1; dy <= 1; dy++ {
        for dx := -1; dx <= 1; dx++ {
            wfc.SetSeed(centerX+dx, centerY+dy, 1) // 1 = water tile
        }
    }
    
    wfc.Generate()
}
```

### Backtracking on Contradiction

```go
type State struct {
    grid [][]Cell
}

func (wfc *WFC) SaveState() *State {
    state := &State{
        grid: make([][]Cell, wfc.height),
    }
    
    for y := 0; y < wfc.height; y++ {
        state.grid[y] = make([]Cell, wfc.width)
        for x := 0; x < wfc.width; x++ {
            // Deep copy options
            options := make([]int, len(wfc.grid[y][x].Options))
            copy(options, wfc.grid[y][x].Options)
            
            state.grid[y][x] = Cell{
                Collapsed: wfc.grid[y][x].Collapsed,
                Options:   options,
                Entropy:   wfc.grid[y][x].Entropy,
            }
        }
    }
    
    return state
}

func (wfc *WFC) RestoreState(state *State) {
    for y := 0; y < wfc.height; y++ {
        for x := 0; x < wfc.width; x++ {
            options := make([]int, len(state.grid[y][x].Options))
            copy(options, state.grid[y][x].Options)
            
            wfc.grid[y][x] = Cell{
                Collapsed: state.grid[y][x].Collapsed,
                Options:   options,
                Entropy:   state.grid[y][x].Entropy,
            }
        }
    }
}

func (wfc *WFC) GenerateWithBacktracking() bool {
    stack := []*State{}
    
    for {
        x, y, found := wfc.FindLowestEntropyCell()
        if !found {
            return true // Success!
        }
        
        // Save state before collapse
        stack = append(stack, wfc.SaveState())
        
        if !wfc.CollapseCell(x, y) || !wfc.Propagate(x, y) {
            // Contradiction! Backtrack
            if len(stack) == 0 {
                return false // No solution
            }
            
            // Restore previous state
            wfc.RestoreState(stack[len(stack)-1])
            stack = stack[:len(stack)-1]
            
            // Remove the option that caused contradiction
            cell := &wfc.grid[y][x]
            if len(cell.Options) > 1 {
                cell.Options = cell.Options[1:]
                cell.Entropy = float64(len(cell.Options))
            }
        }
    }
}
```

## Practical Example: Tilemap Generation

```go
func GenerateTilemap() {
    // Define tiles
    tiles := []Tile{
        {ID: 0, Pattern: [][]int{{0, 0}, {0, 0}}}, // Grass
        {ID: 1, Pattern: [][]int{{1, 1}, {1, 1}}}, // Water
        {ID: 2, Pattern: [][]int{{2, 2}, {2, 2}}}, // Sand
        {ID: 3, Pattern: [][]int{{3, 3}, {3, 3}}}, // Forest
    }
    
    // Create WFC
    wfc := NewWFC(20, 20, tiles)
    
    // Setup rules
    SetupSimpleRules(wfc)
    
    // Add weighted preferences
    weights := map[int]float64{
        0: 10.0, // Grass is common
        1: 3.0,  // Water is less common
        2: 5.0,  // Sand is medium
        3: 7.0,  // Forest is fairly common
    }
    
    // Generate
    if wfc.GenerateWithRetry(10) {
        // Export to image or game map
        ExportToImage(wfc)
    }
}

func ExportToImage(wfc *WFC) {
    // Create image
    img := image.NewRGBA(image.Rect(0, 0, wfc.width*16, wfc.height*16))
    
    colors := map[int]color.RGBA{
        0: {34, 139, 34, 255},   // Grass green
        1: {30, 144, 255, 255},  // Water blue
        2: {238, 214, 175, 255}, // Sand beige
        3: {0, 100, 0, 255},     // Forest dark green
    }
    
    for y := 0; y < wfc.height; y++ {
        for x := 0; x < wfc.width; x++ {
            tileID := wfc.grid[y][x].Options[0]
            c := colors[tileID]
            
            // Fill 16x16 block
            for dy := 0; dy < 16; dy++ {
                for dx := 0; dx < 16; dx++ {
                    img.Set(x*16+dx, y*16+dy, c)
                }
            }
        }
    }
    
    // Save image
    f, _ := os.Create("tilemap.png")
    defer f.Close()
    png.Encode(f, img)
}
```

## Applications

1. **Procedural Level Generation**: Dungeons, overworld maps
2. **Texture Synthesis**: Generate seamless textures
3. **Terrain Generation**: Biome placement, feature distribution
4. **Puzzle Generation**: Sudoku, crosswords
5. **Art Generation**: Pixel art, patterns

## Complexity

**Time**: $O(n^2 \times t \times r)$ where:
- $n$ = grid size
- $t$ = number of tile types
- $r$ = number of rules

**Space**: $O(n^2 \times t)$

## When to Use WFC

✅ **Use when**:
- Need coherent pattern generation
- Have well-defined tile adjacency rules
- Want deterministic results (with seed)
- Need local constraint satisfaction

❌ **Don't use when**:
- Need global structure (use grammar-based generation)
- Rules are too complex (may not converge)
- Need real-time generation (can be slow)
- Want more randomness (use noise-based generation)

