---
title: "Binary Search"
date: 2025-12-13
tags: ["binary-search", "searching", "algorithms", "divide-and-conquer"]
---

**Binary search** is an efficient algorithm for finding a target value in a **sorted** array by repeatedly dividing the search interval in half.

## Basic Algorithm

### Concept

Compare the target with the middle element:
- If equal: found
- If target < middle: search left half
- If target > middle: search right half

### Mathematical Formulation

For a sorted array $A[0 \ldots n-1]$ and target $x$:

$$
\text{BinarySearch}(A, x, \text{left}, \text{right}) = \begin{cases}
-1 & \text{if left} > \text{right} \\\\
\text{mid} & \text{if } A[\text{mid}] = x \\\\
\text{BinarySearch}(A, x, \text{left}, \text{mid}-1) & \text{if } A[\text{mid}] > x \\\\
\text{BinarySearch}(A, x, \text{mid}+1, \text{right}) & \text{if } A[\text{mid}] < x
\end{cases}
$$

where:
$$
\text{mid} = \text{left} + \lfloor (\text{right} - \text{left}) / 2 \rfloor
$$

**Note**: Use `left + (right - left) / 2` instead of `(left + right) / 2` to avoid integer overflow.

### Go Implementation: Basic Binary Search

```go
// BinarySearch finds target in sorted array, returns index or -1
func BinarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    
    for left <= right {
        // Avoid overflow: left + (right-left)/2
        mid := left + (right-left)/2
        
        if arr[mid] == target {
            return mid // Found
        } else if arr[mid] < target {
            left = mid + 1 // Search right half
        } else {
            right = mid - 1 // Search left half
        }
    }
    
    return -1 // Not found
}

// Example usage
func main() {
    arr := []int{1, 3, 5, 7, 9, 11, 13, 15}
    fmt.Println(BinarySearch(arr, 7))  // Output: 3
    fmt.Println(BinarySearch(arr, 6))  // Output: -1
}
```

### Recursive Implementation

```go
func BinarySearchRecursive(arr []int, target, left, right int) int {
    if left > right {
        return -1
    }
    
    mid := left + (right-left)/2
    
    if arr[mid] == target {
        return mid
    } else if arr[mid] < target {
        return BinarySearchRecursive(arr, target, mid+1, right)
    } else {
        return BinarySearchRecursive(arr, target, left, mid-1)
    }
}
```

## Complexity Analysis

### Time Complexity

**Recurrence relation**:
$$
T(n) = T(n/2) + O(1)
$$

**Solution** (by Master Theorem):
$$
T(n) = O(\log n)
$$

**Proof by iteration count**:
After $k$ iterations, search space is $n / 2^k$. When search space becomes 1:
$$
\frac{n}{2^k} = 1 \implies k = \log_2 n
$$

### Space Complexity

- **Iterative**: $O(1)$
- **Recursive**: $O(\log n)$ due to call stack

## Implementation Variants

### 1. Find Exact Match

Returns index if found, -1 otherwise.

**Invariant**: If target exists, it's in `[left, right]`

### 2. Find First Occurrence (Lower Bound)

Find the leftmost position where target appears or could be inserted.

**Invariant**: `A[left-1] < target` and `A[right+1] >= target`

```go
// LowerBound finds first position where arr[i] >= target
// Returns insertion point if target not found
func LowerBound(arr []int, target int) int {
    left, right := 0, len(arr)
    
    for left < right {
        mid := left + (right-left)/2
        
        if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return left
}

// Example
func main() {
    arr := []int{1, 2, 2, 2, 3, 4, 5}
    fmt.Println(LowerBound(arr, 2))  // Output: 1 (first occurrence)
    fmt.Println(LowerBound(arr, 6))  // Output: 7 (insertion point)
}
```

**Result**:
- If `left < n` and `A[left] == target`: first occurrence at `left`
- Otherwise: insertion point is `left`

### 3. Find Last Occurrence (Upper Bound)

Find the rightmost position where target appears.

```go
// UpperBound finds first position where arr[i] > target
func UpperBound(arr []int, target int) int {
    left, right := 0, len(arr)
    
    for left < right {
        mid := left + (right-left)/2
        
        if arr[mid] <= target {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return left
}

// FindLastOccurrence returns index of last occurrence or -1
func FindLastOccurrence(arr []int, target int) int {
    pos := UpperBound(arr, target) - 1
    if pos >= 0 && pos < len(arr) && arr[pos] == target {
        return pos
    }
    return -1
}

// Example
func main() {
    arr := []int{1, 2, 2, 2, 3, 4, 5}
    fmt.Println(FindLastOccurrence(arr, 2))  // Output: 3 (last occurrence)
}
```

### 4. Count Occurrences

```go
// CountOccurrences counts how many times target appears
func CountOccurrences(arr []int, target int) int {
    first := LowerBound(arr, target)
    last := UpperBound(arr, target)
    
    if first < len(arr) && arr[first] == target {
        return last - first
    }
    return 0
}

// Example
func main() {
    arr := []int{1, 2, 2, 2, 3, 4, 5}
    fmt.Println(CountOccurrences(arr, 2))  // Output: 3
}
```

## Advanced Variations

### 1. Search in Rotated Sorted Array

Array is sorted but rotated at some pivot.

**Example**: `[4, 5, 6, 7, 0, 1, 2]`

```go
func SearchRotated(arr []int, target int) int {
    left, right := 0, len(arr)-1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if arr[mid] == target {
            return mid
        }
        
        // Determine which half is sorted
        if arr[left] <= arr[mid] {
            // Left half is sorted
            if arr[left] <= target && target < arr[mid] {
                right = mid - 1 // Target in left half
            } else {
                left = mid + 1 // Target in right half
            }
        } else {
            // Right half is sorted
            if arr[mid] < target && target <= arr[right] {
                left = mid + 1 // Target in right half
            } else {
                right = mid - 1 // Target in left half
            }
        }
    }
    
    return -1
}

// Example
func main() {
    arr := []int{4, 5, 6, 7, 0, 1, 2}
    fmt.Println(SearchRotated(arr, 0))  // Output: 4
}
```

**Time**: $O(\log n)$

### 2. Find Minimum in Rotated Array

```go
func FindMin(arr []int) int {
    left, right := 0, len(arr)-1
    
    for left < right {
        mid := left + (right-left)/2
        
        if arr[mid] > arr[right] {
            // Minimum is in right half
            left = mid + 1
        } else {
            // Minimum is in left half (including mid)
            right = mid
        }
    }
    
    return arr[left]
}

// Example
func main() {
    arr := []int{4, 5, 6, 7, 0, 1, 2}
    fmt.Println(FindMin(arr))  // Output: 0
}
```

**Time**: $O(\log n)$

### 3. Find Peak Element

Element greater than its neighbors.

```go
func FindPeakElement(arr []int) int {
    left, right := 0, len(arr)-1
    
    for left < right {
        mid := left + (right-left)/2
        
        if arr[mid] < arr[mid+1] {
            // Peak is to the right
            left = mid + 1
        } else {
            // Peak is to the left (including mid)
            right = mid
        }
    }
    
    return left
}

// Example
func main() {
    arr := []int{1, 2, 3, 1}
    fmt.Println(FindPeakElement(arr))  // Output: 2 (value 3)
    
    arr2 := []int{1, 2, 1, 3, 5, 6, 4}
    fmt.Println(FindPeakElement(arr2))  // Output: 5 (value 6)
}
```

**Time**: $O(\log n)$

### 4. Search in 2D Matrix

```go
// Search2DMatrix searches in row-sorted and column-sorted matrix
func Search2DMatrix(matrix [][]int, target int) bool {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return false
    }
    
    m, n := len(matrix), len(matrix[0])
    left, right := 0, m*n-1
    
    for left <= right {
        mid := left + (right-left)/2
        // Convert 1D index to 2D coordinates
        row, col := mid/n, mid%n
        midVal := matrix[row][col]
        
        if midVal == target {
            return true
        } else if midVal < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return false
}

// Example
func main() {
    matrix := [][]int{
        {1, 3, 5, 7},
        {10, 11, 16, 20},
        {23, 30, 34, 60},
    }
    fmt.Println(Search2DMatrix(matrix, 3))   // Output: true
    fmt.Println(Search2DMatrix(matrix, 13))  // Output: false
}
```

**Time**: $O(\log(m \times n))$

## Binary Search on Answer

When the answer space is monotonic, binary search can find optimal value.

### Pattern

1. Define search space `[left, right]`
2. Define feasibility function `canAchieve(mid)`
3. Binary search on the answer

### Example: Capacity To Ship Packages Within D Days

**Problem**: Find minimum ship capacity to ship all packages in D days.

```go
func ShipWithinDays(weights []int, days int) int {
    // Search space: [max(weights), sum(weights)]
    left, right := 0, 0
    for _, w := range weights {
        if w > left {
            left = w
        }
        right += w
    }
    
    // Binary search on capacity
    for left < right {
        mid := left + (right-left)/2
        
        if canShip(weights, days, mid) {
            right = mid // Try smaller capacity
        } else {
            left = mid + 1 // Need larger capacity
        }
    }
    
    return left
}

// canShip checks if we can ship with given capacity in days
func canShip(weights []int, days, capacity int) bool {
    daysNeeded, currentLoad := 1, 0
    
    for _, w := range weights {
        if currentLoad+w > capacity {
            daysNeeded++
            currentLoad = w
            if daysNeeded > days {
                return false
            }
        } else {
            currentLoad += w
        }
    }
    
    return true
}

// Example
func main() {
    weights := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    days := 5
    fmt.Println(ShipWithinDays(weights, days))  // Output: 15
}
```

**Time**: $O(n \log(\text{sum} - \text{max}))$

### Example: Koko Eating Bananas

```go
func MinEatingSpeed(piles []int, h int) int {
    left, right := 1, 0
    for _, p := range piles {
        if p > right {
            right = p
        }
    }
    
    for left < right {
        mid := left + (right-left)/2
        
        if canFinish(piles, h, mid) {
            right = mid // Try slower speed
        } else {
            left = mid + 1 // Need faster speed
        }
    }
    
    return left
}

func canFinish(piles []int, h, speed int) bool {
    hours := 0
    for _, p := range piles {
        hours += (p + speed - 1) / speed // Ceiling division
    }
    return hours <= h
}

// Example
func main() {
    piles := []int{3, 6, 7, 11}
    h := 8
    fmt.Println(MinEatingSpeed(piles, h))  // Output: 4
}
```

**Time**: $O(n \log(\text{max}))$

## Mathematical Properties

### Number of Comparisons

**Best case**: 1 (target is at middle)

**Worst case**: $\lceil \log_2(n+1) \rceil$

**Average case**: $\log_2 n - 1$

### Decision Tree Height

Binary search corresponds to a decision tree of height:
$$
h = \lceil \log_2(n+1) \rceil
$$

This is optimal for comparison-based search (information-theoretic lower bound).

### Search Space Reduction

After $k$ iterations:
$$
\text{remaining elements} = \frac{n}{2^k}
$$

## Common Pitfalls

### 1. Integer Overflow

**Wrong**:
```go
mid := (left + right) / 2  // Can overflow!
```

**Correct**:
```go
mid := left + (right-left)/2  // Safe from overflow
```

### 2. Infinite Loop

**Problem**: `mid` calculation can cause infinite loop.

**Wrong**:
```go
// When left = 1, right = 2, mid = 1 forever
for left < right {
    mid := (left + right) / 2
    if arr[mid] <= target {
        left = mid  // Infinite loop!
    }
}
```

**Correct**:
```go
for left < right {
    mid := left + (right-left+1)/2  // Rounds up
    if arr[mid] <= target {
        left = mid
    } else {
        right = mid - 1
    }
}
```

### 3. Off-by-One Errors

**Common mistakes**:
- Using `left <= right` vs. `left < right`
- Updating with `mid` vs. `mid ± 1`
- Return value off by one

**Solution**: Maintain clear invariants and test boundary cases.

## Comparison with Other Search Algorithms

| Algorithm | Time (Average) | Time (Worst) | Space | Requirement |
|-----------|---------------|--------------|-------|-------------|
| Binary Search | $O(\log n)$ | $O(\log n)$ | $O(1)$ | Sorted array |
| Linear Search | $O(n)$ | $O(n)$ | $O(1)$ | None |
| Jump Search | $O(\sqrt{n})$ | $O(\sqrt{n})$ | $O(1)$ | Sorted array |
| Interpolation Search | $O(\log \log n)$ | $O(n)$ | $O(1)$ | Uniformly distributed |
| Exponential Search | $O(\log n)$ | $O(\log n)$ | $O(1)$ | Unbounded sorted array |

## Applications

1. **Dictionary lookup**: Fast word search
2. **Database indexing**: B-tree search
3. **Version control**: Git bisect for finding bugs
4. **Debugging**: Finding first failing test
5. **Game development**: AI decision trees
6. **Numerical methods**: Root finding, optimization
7. **Computer graphics**: Ray tracing, collision detection

## When to Use Binary Search

✅ **Use when**:
- Data is sorted (or has monotonic property)
- Need $O(\log n)$ search time
- Random access is available (arrays)
- Search space is large

❌ **Don't use when**:
- Data is unsorted and sorting cost > search benefit
- Data structure doesn't support random access (linked lists)
- Search space is small (linear search is simpler)
- Need to find all occurrences (may need linear scan anyway)
