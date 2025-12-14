---
title: "Common Interview Challenges"
date: 2025-12-13
tags: ["interview", "algorithms", "coding-challenges", "leetcode"]
---

A curated collection of the most common algorithm interview problems with optimal Go solutions.

## Array Problems

### 1. Two Sum

**Problem**: Find two numbers that add up to target.

```go
func TwoSum(nums []int, target int) []int {
    seen := make(map[int]int)
    
    for i, num := range nums {
        complement := target - num
        if j, exists := seen[complement]; exists {
            return []int{j, i}
        }
        seen[num] = i
    }
    
    return nil
}
```

**Time**: $O(n)$, **Space**: $O(n)$

### 2. Best Time to Buy and Sell Stock

**Problem**: Maximize profit from one buy and one sell.

```go
func MaxProfit(prices []int) int {
    minPrice := prices[0]
    maxProfit := 0
    
    for _, price := range prices {
        if price < minPrice {
            minPrice = price
        } else if price - minPrice > maxProfit {
            maxProfit = price - minPrice
        }
    }
    
    return maxProfit
}
```

**Time**: $O(n)$, **Space**: $O(1)$

### 3. Container With Most Water

**Problem**: Find two lines that form container with most water.

```go
func MaxArea(height []int) int {
    left, right := 0, len(height)-1
    maxArea := 0
    
    for left < right {
        width := right - left
        h := min(height[left], height[right])
        area := width * h
        
        if area > maxArea {
            maxArea = area
        }
        
        // Move pointer with smaller height
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }
    
    return maxArea
}
```

**Time**: $O(n)$, **Space**: $O(1)$

## String Problems

### 4. Longest Substring Without Repeating Characters

**Problem**: Find length of longest substring without repeating characters.

```go
func LengthOfLongestSubstring(s string) int {
    charMap := make(map[byte]int)
    left, maxLen := 0, 0
    
    for right := 0; right < len(s); right++ {
        if idx, exists := charMap[s[right]]; exists && idx >= left {
            left = idx + 1
        }
        
        charMap[s[right]] = right
        if right - left + 1 > maxLen {
            maxLen = right - left + 1
        }
    }
    
    return maxLen
}
```

**Time**: $O(n)$, **Space**: $O(min(n, m))$ where m is charset size

### 5. Valid Parentheses

**Problem**: Check if string has valid parentheses.

```go
func IsValid(s string) bool {
    stack := []rune{}
    pairs := map[rune]rune{')': '(', '}': '{', ']': '['}
    
    for _, char := range s {
        if char == '(' || char == '{' || char == '[' {
            stack = append(stack, char)
        } else {
            if len(stack) == 0 || stack[len(stack)-1] != pairs[char] {
                return false
            }
            stack = stack[:len(stack)-1]
        }
    }
    
    return len(stack) == 0
}
```

**Time**: $O(n)$, **Space**: $O(n)$

## Linked List Problems

### 6. Merge Two Sorted Lists

**Problem**: Merge two sorted linked lists.

```go
func MergeTwoLists(l1, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    curr := dummy
    
    for l1 != nil && l2 != nil {
        if l1.Val <= l2.Val {
            curr.Next = l1
            l1 = l1.Next
        } else {
            curr.Next = l2
            l2 = l2.Next
        }
        curr = curr.Next
    }
    
    if l1 != nil {
        curr.Next = l1
    } else {
        curr.Next = l2
    }
    
    return dummy.Next
}
```

**Time**: $O(m + n)$, **Space**: $O(1)$

### 7. Reverse Linked List

**Problem**: Reverse a linked list.

```go
func ReverseList(head *ListNode) *ListNode {
    var prev *ListNode
    curr := head
    
    for curr != nil {
        next := curr.Next
        curr.Next = prev
        prev = curr
        curr = next
    }
    
    return prev
}
```

**Time**: $O(n)$, **Space**: $O(1)$

## Tree Problems

### 8. Maximum Depth of Binary Tree

**Problem**: Find maximum depth of binary tree.

```go
func MaxDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    leftDepth := MaxDepth(root.Left)
    rightDepth := MaxDepth(root.Right)
    
    return 1 + max(leftDepth, rightDepth)
}
```

**Time**: $O(n)$, **Space**: $O(h)$ where h is height

### 9. Validate Binary Search Tree

**Problem**: Check if tree is valid BST.

```go
func IsValidBST(root *TreeNode) bool {
    return validate(root, nil, nil)
}

func validate(node *TreeNode, min, max *int) bool {
    if node == nil {
        return true
    }
    
    if (min != nil && node.Val <= *min) || (max != nil && node.Val >= *max) {
        return false
    }
    
    return validate(node.Left, min, &node.Val) && 
           validate(node.Right, &node.Val, max)
}
```

**Time**: $O(n)$, **Space**: $O(h)$

### 10. Lowest Common Ancestor

**Problem**: Find LCA of two nodes in BST.

```go
func LowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    // If both nodes are in left subtree
    if p.Val < root.Val && q.Val < root.Val {
        return LowestCommonAncestor(root.Left, p, q)
    }
    
    // If both nodes are in right subtree
    if p.Val > root.Val && q.Val > root.Val {
        return LowestCommonAncestor(root.Right, p, q)
    }
    
    // Current node is LCA
    return root
}
```

**Time**: $O(h)$, **Space**: $O(h)$

## Dynamic Programming

### 11. Climbing Stairs

**Problem**: How many ways to climb n stairs (1 or 2 steps at a time).

```go
func ClimbStairs(n int) int {
    if n <= 2 {
        return n
    }
    
    prev2, prev1 := 1, 2
    
    for i := 3; i <= n; i++ {
        curr := prev1 + prev2
        prev2 = prev1
        prev1 = curr
    }
    
    return prev1
}
```

**Time**: $O(n)$, **Space**: $O(1)$

### 12. House Robber

**Problem**: Maximize money robbed from non-adjacent houses.

```go
func Rob(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    if len(nums) == 1 {
        return nums[0]
    }
    
    prev2, prev1 := 0, nums[0]
    
    for i := 1; i < len(nums); i++ {
        curr := max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = curr
    }
    
    return prev1
}
```

**Time**: $O(n)$, **Space**: $O(1)$

### 13. Coin Change

**Problem**: Minimum coins needed to make amount.

```go
func CoinChange(coins []int, amount int) int {
    dp := make([]int, amount+1)
    for i := 1; i <= amount; i++ {
        dp[i] = amount + 1 // Infinity
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
        return -1
    }
    return dp[amount]
}
```

**Time**: $O(n \times amount)$, **Space**: $O(amount)$

## Graph Problems

### 14. Number of Islands

**Problem**: Count number of islands in 2D grid.

```go
func NumIslands(grid [][]byte) int {
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
    if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[0]) || grid[i][j] != '1' {
        return
    }
    
    grid[i][j] = '0' // Mark as visited
    
    dfs(grid, i+1, j)
    dfs(grid, i-1, j)
    dfs(grid, i, j+1)
    dfs(grid, i, j-1)
}
```

**Time**: $O(m \times n)$, **Space**: $O(m \times n)$ for recursion

### 15. Course Schedule (Cycle Detection)

**Problem**: Can you finish all courses given prerequisites?

```go
func CanFinish(numCourses int, prerequisites [][]int) bool {
    graph := make([][]int, numCourses)
    for _, pre := range prerequisites {
        graph[pre[1]] = append(graph[pre[1]], pre[0])
    }
    
    visited := make([]int, numCourses) // 0: unvisited, 1: visiting, 2: visited
    
    var hasCycle func(int) bool
    hasCycle = func(course int) bool {
        if visited[course] == 1 {
            return true // Cycle detected
        }
        if visited[course] == 2 {
            return false // Already processed
        }
        
        visited[course] = 1 // Mark as visiting
        
        for _, next := range graph[course] {
            if hasCycle(next) {
                return true
            }
        }
        
        visited[course] = 2 // Mark as visited
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

## Backtracking

### 16. Generate Parentheses

**Problem**: Generate all valid combinations of n pairs of parentheses.

```go
func GenerateParenthesis(n int) []string {
    result := []string{}
    backtrack(&result, "", 0, 0, n)
    return result
}

func backtrack(result *[]string, current string, open, close, max int) {
    if len(current) == max*2 {
        *result = append(*result, current)
        return
    }
    
    if open < max {
        backtrack(result, current+"(", open+1, close, max)
    }
    if close < open {
        backtrack(result, current+")", open, close+1, max)
    }
}
```

**Time**: $O(\frac{4^n}{\sqrt{n}})$ (Catalan number), **Space**: $O(n)$

### 17. Permutations

**Problem**: Generate all permutations of array.

```go
func Permute(nums []int) [][]int {
    result := [][]int{}
    backtrackPermute(&result, nums, 0)
    return result
}

func backtrackPermute(result *[][]int, nums []int, start int) {
    if start == len(nums) {
        perm := make([]int, len(nums))
        copy(perm, nums)
        *result = append(*result, perm)
        return
    }
    
    for i := start; i < len(nums); i++ {
        nums[start], nums[i] = nums[i], nums[start]
        backtrackPermute(result, nums, start+1)
        nums[start], nums[i] = nums[i], nums[start] // Backtrack
    }
}
```

**Time**: $O(n!)$, **Space**: $O(n)$

## Bit Manipulation

### 18. Single Number

**Problem**: Find the number that appears once (others appear twice).

```go
func SingleNumber(nums []int) int {
    result := 0
    for _, num := range nums {
        result ^= num // XOR cancels out duplicates
    }
    return result
}
```

**Time**: $O(n)$, **Space**: $O(1)$

### 19. Number of 1 Bits

**Problem**: Count number of 1 bits in integer.

```go
func HammingWeight(num uint32) int {
    count := 0
    for num != 0 {
        count++
        num &= (num - 1) // Remove rightmost 1 bit
    }
    return count
}
```

**Time**: $O(\log n)$, **Space**: $O(1)$

## Sliding Window

### 20. Minimum Window Substring

**Problem**: Find minimum window in s that contains all characters of t.

```go
func MinWindow(s string, t string) string {
    if len(s) < len(t) {
        return ""
    }
    
    need := make(map[byte]int)
    for i := 0; i < len(t); i++ {
        need[t[i]]++
    }
    
    left, right := 0, 0
    valid := 0
    start, length := 0, len(s)+1
    window := make(map[byte]int)
    
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

## Helper Functions

```go
func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

## Problem Categories

### Easy (Good for Warm-up)
- Two Sum
- Valid Parentheses
- Merge Two Sorted Lists
- Maximum Depth of Binary Tree
- Climbing Stairs
- Single Number

### Medium (Core Interview Questions)
- Longest Substring Without Repeating
- Container With Most Water
- Reverse Linked List
- Validate BST
- Coin Change
- Number of Islands
- Generate Parentheses

### Hard (Advanced)
- Minimum Window Substring
- Merge K Sorted Lists
- Median of Two Sorted Arrays
- Trapping Rain Water

## Study Strategy

1. **Master the patterns**: Two pointers, sliding window, DFS/BFS, DP
2. **Practice daily**: 1-2 problems per day
3. **Time yourself**: 30-45 minutes per problem
4. **Understand, don't memorize**: Focus on why solutions work
5. **Review mistakes**: Keep a log of problems you struggled with

