---
title: "Interview Questions - Hard"
date: 2025-12-13
tags: ["interview", "algorithms", "hard", "leetcode"]
---

Hard-level algorithm interview questions with detailed approach explanations and solutions.

## Problem 1: Median of Two Sorted Arrays

**Problem**: Find median of two sorted arrays in $O(\log(m+n))$ time.

**Example**:
```
Input: nums1 = [1,3], nums2 = [2]
Output: 2.0
Explanation: merged = [1,2,3], median = 2
```

### Approach

**Key Insight**: Binary search on smaller array to partition both arrays.

**Goal**: Partition arrays so left half ≤ right half.

**Steps**:
1. Binary search on smaller array
2. Calculate partition in second array
3. Check if valid partition (maxLeft ≤ minRight)
4. Adjust binary search bounds

### Solution

```go
func findMedianSortedArrays(nums1, nums2 []int) float64 {
    // Ensure nums1 is smaller
    if len(nums1) > len(nums2) {
        nums1, nums2 = nums2, nums1
    }
    
    m, n := len(nums1), len(nums2)
    left, right := 0, m
    
    for left <= right {
        partition1 := (left + right) / 2
        partition2 := (m + n + 1) / 2 - partition1
        
        maxLeft1 := math.MinInt32
        if partition1 > 0 {
            maxLeft1 = nums1[partition1-1]
        }
        
        minRight1 := math.MaxInt32
        if partition1 < m {
            minRight1 = nums1[partition1]
        }
        
        maxLeft2 := math.MinInt32
        if partition2 > 0 {
            maxLeft2 = nums2[partition2-1]
        }
        
        minRight2 := math.MaxInt32
        if partition2 < n {
            minRight2 = nums2[partition2]
        }
        
        if maxLeft1 <= minRight2 && maxLeft2 <= minRight1 {
            // Found correct partition
            if (m + n) % 2 == 0 {
                return float64(max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2.0
            }
            return float64(max(maxLeft1, maxLeft2))
        } else if maxLeft1 > minRight2 {
            right = partition1 - 1
        } else {
            left = partition1 + 1
        }
    }
    
    return 0.0
}
```

**Time**: $O(\log(\min(m, n)))$, **Space**: $O(1)$

**Why binary search**: We're finding the correct partition point.

---

## Problem 2: Trapping Rain Water

**Problem**: Calculate how much water can be trapped after raining.

**Example**:
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

### Approach

**Key Insight**: Water at position i = min(maxLeft, maxRight) - height[i]

**Pattern**: Two pointers from both ends, track max heights.

### Solution

```go
func trap(height []int) int {
    if len(height) == 0 {
        return 0
    }
    
    left, right := 0, len(height)-1
    leftMax, rightMax := 0, 0
    water := 0
    
    for left < right {
        if height[left] < height[right] {
            if height[left] >= leftMax {
                leftMax = height[left]
            } else {
                water += leftMax - height[left]
            }
            left++
        } else {
            if height[right] >= rightMax {
                rightMax = height[right]
            } else {
                water += rightMax - height[right]
            }
            right--
        }
    }
    
    return water
}
```

**Time**: $O(n)$, **Space**: $O(1)$

**Why this works**: Process from lower side, guarantees water is trapped by higher side.

---

## Problem 3: Regular Expression Matching

**Problem**: Implement regex matching with '.' and '*'.

**Example**:
```
Input: s = "aa", p = "a*"
Output: true
Explanation: '*' means zero or more of preceding element
```

### Approach

**Key Insight**: Dynamic programming with pattern matching logic.

**DP State**: $dp[i][j]$ = does s[0:i] match p[0:j]?

### Solution

```go
func isMatch(s string, p string) bool {
    m, n := len(s), len(p)
    dp := make([][]bool, m+1)
    for i := range dp {
        dp[i] = make([]bool, n+1)
    }
    
    dp[0][0] = true
    
    // Handle patterns like a*, a*b*, etc.
    for j := 2; j <= n; j++ {
        if p[j-1] == '*' {
            dp[0][j] = dp[0][j-2]
        }
    }
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if p[j-1] == '*' {
                // Zero occurrences or one+ occurrences
                dp[i][j] = dp[i][j-2] || 
                          (dp[i-1][j] && (s[i-1] == p[j-2] || p[j-2] == '.'))
            } else if p[j-1] == '.' || s[i-1] == p[j-1] {
                dp[i][j] = dp[i-1][j-1]
            }
        }
    }
    
    return dp[m][n]
}
```

**Time**: $O(m \times n)$, **Space**: $O(m \times n)$

**Critical**: '*' matches zero or more of **preceding** element.

---

## Problem 4: Merge K Sorted Lists

**Problem**: Merge k sorted linked lists into one sorted list.

**Example**:
```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
```

### Approach

**Key Insight**: Use min heap to track smallest element from each list.

**Pattern**: K-way merge using priority queue.

### Solution

```go
import "container/heap"

type MinHeap []*ListNode

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].Val < h[j].Val }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MinHeap) Push(x interface{}) {
    *h = append(*h, x.(*ListNode))
}

func (h *MinHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func mergeKLists(lists []*ListNode) *ListNode {
    h := &MinHeap{}
    heap.Init(h)
    
    // Add first node from each list
    for _, list := range lists {
        if list != nil {
            heap.Push(h, list)
        }
    }
    
    dummy := &ListNode{}
    curr := dummy
    
    for h.Len() > 0 {
        node := heap.Pop(h).(*ListNode)
        curr.Next = node
        curr = curr.Next
        
        if node.Next != nil {
            heap.Push(h, node.Next)
        }
    }
    
    return dummy.Next
}
```

**Time**: $O(N \log k)$ where N is total nodes, k is number of lists  
**Space**: $O(k)$ for heap

**Alternative**: Divide and conquer (merge pairs recursively).

---

## Problem 5: Longest Valid Parentheses

**Problem**: Find length of longest valid parentheses substring.

**Example**:
```
Input: s = "(()"
Output: 2
Explanation: "()" is longest valid
```

### Approach

**Key Insight**: Use stack to track indices, or two-pass scan.

**Pattern**: Stack-based matching with index tracking.

### Solution (Stack)

```go
func longestValidParentheses(s string) int {
    stack := []int{-1} // Base for valid substring
    maxLen := 0
    
    for i, char := range s {
        if char == '(' {
            stack = append(stack, i)
        } else {
            stack = stack[:len(stack)-1] // Pop
            
            if len(stack) == 0 {
                stack = append(stack, i) // New base
            } else {
                maxLen = max(maxLen, i - stack[len(stack)-1])
            }
        }
    }
    
    return maxLen
}
```

**Time**: $O(n)$, **Space**: $O(n)$

### Solution (Two-Pass, O(1) Space)

```go
func longestValidParentheses(s string) int {
    left, right, maxLen := 0, 0, 0
    
    // Left to right
    for i := 0; i < len(s); i++ {
        if s[i] == '(' {
            left++
        } else {
            right++
        }
        
        if left == right {
            maxLen = max(maxLen, 2*right)
        } else if right > left {
            left, right = 0, 0
        }
    }
    
    left, right = 0, 0
    
    // Right to left
    for i := len(s)-1; i >= 0; i-- {
        if s[i] == '(' {
            left++
        } else {
            right++
        }
        
        if left == right {
            maxLen = max(maxLen, 2*left)
        } else if left > right {
            left, right = 0, 0
        }
    }
    
    return maxLen
}
```

**Time**: $O(n)$, **Space**: $O(1)$

---

## Problem 6: Word Ladder

**Problem**: Find shortest transformation sequence from beginWord to endWord.

**Example**:
```
Input: beginWord = "hit", endWord = "cog", 
       wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: "hit" -> "hot" -> "dot" -> "dog" -> "cog"
```

### Approach

**Key Insight**: BFS on word graph (words differ by one letter are connected).

**Pattern**: Shortest path in unweighted graph.

### Solution

```go
func ladderLength(beginWord string, endWord string, wordList []string) int {
    wordSet := make(map[string]bool)
    for _, word := range wordList {
        wordSet[word] = true
    }
    
    if !wordSet[endWord] {
        return 0
    }
    
    queue := []string{beginWord}
    visited := make(map[string]bool)
    visited[beginWord] = true
    level := 1
    
    for len(queue) > 0 {
        size := len(queue)
        
        for i := 0; i < size; i++ {
            word := queue[0]
            queue = queue[1:]
            
            if word == endWord {
                return level
            }
            
            // Try all possible one-letter changes
            for j := 0; j < len(word); j++ {
                for ch := 'a'; ch <= 'z'; ch++ {
                    if rune(word[j]) == ch {
                        continue
                    }
                    
                    newWord := word[:j] + string(ch) + word[j+1:]
                    
                    if wordSet[newWord] && !visited[newWord] {
                        queue = append(queue, newWord)
                        visited[newWord] = true
                    }
                }
            }
        }
        
        level++
    }
    
    return 0
}
```

**Time**: $O(M^2 \times N)$ where M is word length, N is number of words  
**Space**: $O(M \times N)$

**Optimization**: Bidirectional BFS from both ends.

---

## Problem 7: Minimum Window Substring

**Problem**: Find minimum window in s containing all characters of t.

**Example**:
```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
```

### Approach

**Key Insight**: Sliding window with character frequency tracking.

**Pattern**: Expand right to include all chars, contract left to minimize.

### Solution

```go
func minWindow(s string, t string) string {
    if len(s) < len(t) {
        return ""
    }
    
    need := make(map[byte]int)
    for i := 0; i < len(t); i++ {
        need[t[i]]++
    }
    
    window := make(map[byte]int)
    left, right := 0, 0
    valid := 0
    start, length := 0, len(s)+1
    
    for right < len(s) {
        c := s[right]
        right++
        
        if _, exists := need[c]; exists {
            window[c]++
            if window[c] == need[c] {
                valid++
            }
        }
        
        for valid == len(need) {
            if right - left < length {
                start = left
                length = right - left
            }
            
            d := s[left]
            left++
            
            if _, exists := need[d]; exists {
                if window[d] == need[d] {
                    valid--
                }
                window[d]--
            }
        }
    }
    
    if length == len(s)+1 {
        return ""
    }
    return s[start : start+length]
}
```

**Time**: $O(m + n)$, **Space**: $O(m + n)$

---

## Common Patterns in Hard Problems

1. **Binary Search on Answer**: Median of sorted arrays, capacity problems
2. **Advanced DP**: Edit distance, regex matching
3. **Heap/Priority Queue**: Merge K lists, sliding window maximum
4. **Graph Algorithms**: Word ladder, shortest path with obstacles
5. **Sliding Window**: Minimum window substring
6. **Backtracking with Pruning**: N-Queens, Sudoku solver
7. **Divide and Conquer**: Merge K lists

## Approach Strategy for Hard Problems

1. **Break down**: Identify subproblems
2. **Pattern recognition**: Which technique applies?
3. **Start simple**: Solve easier version first
4. **Optimize incrementally**: Don't aim for optimal immediately
5. **Consider multiple approaches**: DP vs. greedy vs. graph

## Interview Tips for Hard Problems

✅ **Do**:
- Discuss multiple approaches
- Explain trade-offs
- Start with working solution, then optimize
- Ask if optimization is needed

❌ **Don't**:
- Panic if you don't see solution immediately
- Code without discussing approach
- Ignore hints from interviewer
- Give up

**Remember**: Process matters more than perfect solution!

