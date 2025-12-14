---
title: "Interview Questions - Medium"
date: 2025-12-13
tags: ["interview", "algorithms", "medium", "leetcode"]
---

Medium-level algorithm interview questions with detailed approach explanations and solutions.

## Problem 1: Longest Substring Without Repeating Characters

**Problem**: Find length of longest substring without repeating characters.

**Example**:
```
Input: s = "abcabcbb"
Output: 3
Explanation: "abc" is longest without repeats
```

### Approach

**Key Insight**: Sliding window with hash map to track last seen position.

**Pattern**: Expand window when no repeat, contract when repeat found.

**Steps**:
1. Use map to store character -> last seen index
2. Expand right pointer
3. If character seen and within window, move left pointer
4. Track maximum length

### Solution

```go
func lengthOfLongestSubstring(s string) int {
    charIndex := make(map[byte]int)
    left, maxLen := 0, 0
    
    for right := 0; right < len(s); right++ {
        char := s[right]
        
        // If char seen and within current window
        if lastIndex, exists := charIndex[char]; exists && lastIndex >= left {
            left = lastIndex + 1
        }
        
        charIndex[char] = right
        maxLen = max(maxLen, right - left + 1)
    }
    
    return maxLen
}
```

**Time**: $O(n)$, **Space**: $O(min(n, m))$ where m is charset size

**Why this works**: We maintain a valid window [left, right] with no repeats.

---

## Problem 2: 3Sum

**Problem**: Find all unique triplets that sum to zero.

**Example**:
```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```

### Approach

**Key Insight**: Sort array, fix one number, use two pointers for remaining two.

**Pattern**: Reduce to 2Sum problem.

**Steps**:
1. Sort array
2. For each number, find pairs that sum to -number
3. Skip duplicates

### Solution

```go
func threeSum(nums []int) [][]int {
    sort.Ints(nums)
    result := [][]int{}
    
    for i := 0; i < len(nums)-2; i++ {
        // Skip duplicates for first number
        if i > 0 && nums[i] == nums[i-1] {
            continue
        }
        
        target := -nums[i]
        left, right := i+1, len(nums)-1
        
        for left < right {
            sum := nums[left] + nums[right]
            
            if sum == target {
                result = append(result, []int{nums[i], nums[left], nums[right]})
                
                // Skip duplicates
                for left < right && nums[left] == nums[left+1] {
                    left++
                }
                for left < right && nums[right] == nums[right-1] {
                    right--
                }
                
                left++
                right--
            } else if sum < target {
                left++
            } else {
                right--
            }
        }
    }
    
    return result
}
```

**Time**: $O(n^2)$, **Space**: $O(1)$ excluding output

**Critical**: Must skip duplicates to avoid duplicate triplets.

---

## Problem 3: Group Anagrams

**Problem**: Group strings that are anagrams of each other.

**Example**:
```
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

### Approach

**Key Insight**: Anagrams have same sorted characters or same character frequency.

**Pattern**: Use hash map with sorted string as key.

### Solution

```go
import "sort"
import "strings"

func groupAnagrams(strs []string) [][]string {
    groups := make(map[string][]string)
    
    for _, str := range strs {
        // Sort characters to create key
        chars := []rune(str)
        sort.Slice(chars, func(i, j int) bool {
            return chars[i] < chars[j]
        })
        key := string(chars)
        
        groups[key] = append(groups[key], str)
    }
    
    result := [][]string{}
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}

// Alternative: Use character count as key
func groupAnagramsCount(strs []string) [][]string {
    groups := make(map[[26]int][]string)
    
    for _, str := range strs {
        var count [26]int
        for _, ch := range str {
            count[ch-'a']++
        }
        groups[count] = append(groups[count], str)
    }
    
    result := [][]string{}
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}
```

**Time**: $O(n \times k \log k)$ where k is max string length  
**Space**: $O(n \times k)$

**Alternative**: Character count approach is $O(n \times k)$ time.

---

## Problem 4: Product of Array Except Self

**Problem**: Return array where each element is product of all others (no division).

**Example**:
```
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
```

### Approach

**Key Insight**: Product = (product of all left) × (product of all right).

**Pattern**: Two passes - left to right, then right to left.

### Solution

```go
func productExceptSelf(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    
    // Pass 1: Calculate left products
    result[0] = 1
    for i := 1; i < n; i++ {
        result[i] = result[i-1] * nums[i-1]
    }
    
    // Pass 2: Multiply by right products
    rightProduct := 1
    for i := n-1; i >= 0; i-- {
        result[i] *= rightProduct
        rightProduct *= nums[i]
    }
    
    return result
}
```

**Time**: $O(n)$, **Space**: $O(1)$ excluding output

**Why this works**: result[i] = (nums[0]×...×nums[i-1]) × (nums[i+1]×...×nums[n-1])

---

## Problem 5: Coin Change

**Problem**: Find minimum coins needed to make amount (infinite supply of each coin).

**Example**:
```
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
```

### Approach

**Key Insight**: Dynamic programming - build up from smaller amounts.

**Recurrence**: $dp[i] = \min(dp[i], dp[i - coin] + 1)$ for each coin

### Solution

```go
func coinChange(coins []int, amount int) int {
    dp := make([]int, amount+1)
    
    // Initialize with "impossible" value
    for i := 1; i <= amount; i++ {
        dp[i] = amount + 1
    }
    dp[0] = 0
    
    for i := 1; i <= amount; i++ {
        for _, coin := range coins {
            if coin <= i {
                dp[i] = min(dp[i], dp[i-coin] + 1)
            }
        }
    }
    
    if dp[amount] > amount {
        return -1 // Impossible
    }
    return dp[amount]
}
```

**Time**: $O(amount \times coins)$, **Space**: $O(amount)$

**Why DP**: Optimal substructure - optimal solution uses optimal solutions to subproblems.

---

## Problem 6: Number of Islands

**Problem**: Count number of islands in 2D grid ('1' = land, '0' = water).

**Example**:
```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

### Approach

**Key Insight**: DFS/BFS to mark connected components.

**Pattern**: Graph traversal, mark visited cells.

### Solution

```go
func numIslands(grid [][]byte) int {
    if len(grid) == 0 {
        return 0
    }
    
    count := 0
    
    for i := 0; i < len(grid); i++ {
        for j := 0; j < len(grid[0]); j++ {
            if grid[i][j] == '1' {
                dfs(grid, i, j)
                count++
            }
        }
    }
    
    return count
}

func dfs(grid [][]byte, i, j int) {
    // Boundary check
    if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[0]) || grid[i][j] != '1' {
        return
    }
    
    grid[i][j] = '0' // Mark as visited
    
    // Explore 4 directions
    dfs(grid, i+1, j)
    dfs(grid, i-1, j)
    dfs(grid, i, j+1)
    dfs(grid, i, j-1)
}
```

**Time**: $O(m \times n)$, **Space**: $O(m \times n)$ for recursion stack

**Alternative**: BFS using queue instead of recursion.

---

## Problem 7: Course Schedule (Cycle Detection)

**Problem**: Can you finish all courses given prerequisites? (Detect cycle in directed graph)

**Example**:
```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: Take course 0, then course 1
```

### Approach

**Key Insight**: Build dependency graph, detect cycles using DFS.

**Pattern**: Graph cycle detection with 3 states (unvisited, visiting, visited).

### Solution

```go
func canFinish(numCourses int, prerequisites [][]int) bool {
    // Build adjacency list
    graph := make([][]int, numCourses)
    for _, pre := range prerequisites {
        graph[pre[1]] = append(graph[pre[1]], pre[0])
    }
    
    // 0: unvisited, 1: visiting, 2: visited
    state := make([]int, numCourses)
    
    var hasCycle func(int) bool
    hasCycle = func(course int) bool {
        if state[course] == 1 {
            return true // Cycle detected
        }
        if state[course] == 2 {
            return false // Already processed
        }
        
        state[course] = 1 // Mark as visiting
        
        for _, next := range graph[course] {
            if hasCycle(next) {
                return true
            }
        }
        
        state[course] = 2 // Mark as visited
        return false
    }
    
    for i := 0; i < numCourses; i++ {
        if hasCycle(i) {
            return false
        }
    }
    
    return true
}
```

**Time**: $O(V + E)$, **Space**: $O(V + E)$

**Why 3 states**: Distinguish between "not yet visited" and "currently in recursion stack".

---

## Problem 8: Lowest Common Ancestor of Binary Tree

**Problem**: Find lowest common ancestor of two nodes in binary tree.

**Example**:
```
Input: root = [3,5,1,6,2,0,8], p = 5, q = 1
Output: 3
```

### Approach

**Key Insight**: If p and q are in different subtrees, current node is LCA.

**Pattern**: Recursive tree traversal with bottom-up information passing.

### Solution

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if root == nil || root == p || root == q {
        return root
    }
    
    left := lowestCommonAncestor(root.Left, p, q)
    right := lowestCommonAncestor(root.Right, p, q)
    
    // If both found in different subtrees, current node is LCA
    if left != nil && right != nil {
        return root
    }
    
    // Return whichever is not nil
    if left != nil {
        return left
    }
    return right
}
```

**Time**: $O(n)$, **Space**: $O(h)$ where h is height

**Why this works**: First node where paths to p and q diverge is the LCA.

---

## Problem 9: Kth Largest Element (QuickSelect)

**Problem**: Find k-th largest element in unsorted array.

**Example**:
```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
```

### Approach

**Key Insight**: Use QuickSelect (like QuickSort but only recurse on one side).

**Pattern**: Partitioning with early termination.

### Solution

```go
func findKthLargest(nums []int, k int) int {
    return quickSelect(nums, 0, len(nums)-1, len(nums)-k)
}

func quickSelect(nums []int, left, right, k int) int {
    if left == right {
        return nums[left]
    }
    
    pivotIndex := partition(nums, left, right)
    
    if k == pivotIndex {
        return nums[k]
    } else if k < pivotIndex {
        return quickSelect(nums, left, pivotIndex-1, k)
    } else {
        return quickSelect(nums, pivotIndex+1, right, k)
    }
}

func partition(nums []int, left, right int) int {
    pivot := nums[right]
    i := left
    
    for j := left; j < right; j++ {
        if nums[j] <= pivot {
            nums[i], nums[j] = nums[j], nums[i]
            i++
        }
    }
    
    nums[i], nums[right] = nums[right], nums[i]
    return i
}
```

**Time**: $O(n)$ average, $O(n^2)$ worst, **Space**: $O(1)$

**Alternative**: Use heap ($O(n \log k)$ time, $O(k)$ space).

---

## Problem 10: Word Break

**Problem**: Determine if string can be segmented into words from dictionary.

**Example**:
```
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
```

### Approach

**Key Insight**: Dynamic programming - can we break s[0:i]?

**Recurrence**: $dp[i] = \text{true if } \exists j < i: dp[j] \land s[j:i] \in dict$

### Solution

```go
func wordBreak(s string, wordDict []string) bool {
    wordSet := make(map[string]bool)
    for _, word := range wordDict {
        wordSet[word] = true
    }
    
    n := len(s)
    dp := make([]bool, n+1)
    dp[0] = true // Empty string
    
    for i := 1; i <= n; i++ {
        for j := 0; j < i; j++ {
            if dp[j] && wordSet[s[j:i]] {
                dp[i] = true
                break
            }
        }
    }
    
    return dp[n]
}
```

**Time**: $O(n^2 \times m)$ where m is max word length, **Space**: $O(n)$

**Optimization**: Check only valid word lengths instead of all j.

---

## Common Patterns in Medium Problems

1. **Sliding Window**: Longest substring, subarray problems
2. **Two Pointers**: 3Sum, container with most water
3. **Hash Map**: Group anagrams, subarray sum
4. **Dynamic Programming**: Coin change, word break
5. **DFS/BFS**: Islands, course schedule
6. **Binary Search**: Search in rotated array
7. **Backtracking**: Permutations, combinations

## Problem-Solving Framework

1. **Identify pattern**: Which category does this fit?
2. **Consider brute force**: What's the naive solution?
3. **Optimize**: Can we use hash map? Two pointers? DP?
4. **Edge cases**: Empty input, single element, all same
5. **Test**: Walk through example step-by-step

## Time Complexity Goals

- **$O(n)$**: Single pass with hash map/set
- **$O(n \log n)$**: Sorting, heap operations
- **$O(n^2)$**: Nested loops (often optimizable)
- **$O(2^n)$**: Backtracking, subsets

Aim for better than brute force!

