---
title: "Interview Questions - Easy"
date: 2025-12-13
tags: ["interview", "algorithms", "easy", "leetcode"]
---

Easy-level algorithm interview questions with detailed approach explanations and solutions.

## How to Approach Interview Problems

1. **Clarify requirements**: Ask about edge cases, constraints, input/output format
2. **Think out loud**: Explain your thought process
3. **Start with brute force**: Then optimize
4. **Consider trade-offs**: Time vs. space complexity
5. **Test with examples**: Walk through your solution
6. **Handle edge cases**: Empty input, single element, duplicates

## Problem 1: Two Sum

**Problem**: Given an array of integers and a target, return indices of two numbers that add up to target.

**Example**:
```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9
```

### Approach

**Brute Force** ($O(n^2)$): Check all pairs
```go
for i := 0; i < len(nums); i++ {
    for j := i+1; j < len(nums); j++ {
        if nums[i] + nums[j] == target {
            return []int{i, j}
        }
    }
}
```

**Optimized** ($O(n)$): Use hash map to store complements

**Key Insight**: For each number, check if its complement (target - num) exists in map.

### Solution

```go
func twoSum(nums []int, target int) []int {
    seen := make(map[int]int) // value -> index
    
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

**Why this works**: We build the map as we go, so when we find a complement, we know its index.

---

## Problem 2: Valid Parentheses

**Problem**: Determine if string of brackets is valid (every opening has matching closing in correct order).

**Example**:
```
Input: s = "()[]{}"
Output: true

Input: s = "([)]"
Output: false
```

### Approach

**Key Insight**: Use stack - opening brackets push, closing brackets must match top of stack.

**Steps**:
1. For each character:
   - If opening bracket: push to stack
   - If closing bracket: check if matches stack top, pop if yes
2. Stack should be empty at end

### Solution

```go
func isValid(s string) bool {
    stack := []rune{}
    pairs := map[rune]rune{')': '(', '}': '{', ']': '['}
    
    for _, char := range s {
        // Opening bracket
        if char == '(' || char == '{' || char == '[' {
            stack = append(stack, char)
        } else {
            // Closing bracket
            if len(stack) == 0 || stack[len(stack)-1] != pairs[char] {
                return false
            }
            stack = stack[:len(stack)-1] // Pop
        }
    }
    
    return len(stack) == 0
}
```

**Time**: $O(n)$, **Space**: $O(n)$

**Edge cases**: Empty string (valid), unmatched opening, unmatched closing

---

## Problem 3: Merge Two Sorted Lists

**Problem**: Merge two sorted linked lists into one sorted list.

**Example**:
```
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
```

### Approach

**Key Insight**: Use two pointers, always pick smaller value.

**Technique**: Dummy node to simplify edge cases.

### Solution

```go
func mergeTwoLists(l1, l2 *ListNode) *ListNode {
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
    
    // Attach remaining nodes
    if l1 != nil {
        curr.Next = l1
    } else {
        curr.Next = l2
    }
    
    return dummy.Next
}
```

**Time**: $O(m + n)$, **Space**: $O(1)$

**Why dummy node**: Avoids special case for first node.

---

## Problem 4: Best Time to Buy and Sell Stock

**Problem**: Find maximum profit from one buy and one sell (buy before sell).

**Example**:
```
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy at 1, sell at 6, profit = 5
```

### Approach

**Key Insight**: Track minimum price seen so far, calculate profit at each step.

**Pattern**: Single pass, keep running minimum.

### Solution

```go
func maxProfit(prices []int) int {
    if len(prices) == 0 {
        return 0
    }
    
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

**Why this works**: At each point, we know the best buy price before this point.

---

## Problem 5: Valid Palindrome

**Problem**: Check if string is palindrome (ignoring non-alphanumeric, case-insensitive).

**Example**:
```
Input: s = "A man, a plan, a canal: Panama"
Output: true
```

### Approach

**Key Insight**: Two pointers from both ends, skip non-alphanumeric.

### Solution

```go
import "unicode"

func isPalindrome(s string) bool {
    left, right := 0, len(s)-1
    
    for left < right {
        // Skip non-alphanumeric from left
        for left < right && !isAlphaNumeric(rune(s[left])) {
            left++
        }
        
        // Skip non-alphanumeric from right
        for left < right && !isAlphaNumeric(rune(s[right])) {
            right--
        }
        
        // Compare (case-insensitive)
        if unicode.ToLower(rune(s[left])) != unicode.ToLower(rune(s[right])) {
            return false
        }
        
        left++
        right--
    }
    
    return true
}

func isAlphaNumeric(ch rune) bool {
    return unicode.IsLetter(ch) || unicode.IsDigit(ch)
}
```

**Time**: $O(n)$, **Space**: $O(1)$

---

## Problem 6: Reverse Linked List

**Problem**: Reverse a singly linked list.

**Example**:
```
Input: 1 -> 2 -> 3 -> 4 -> 5
Output: 5 -> 4 -> 3 -> 2 -> 1
```

### Approach

**Key Insight**: Three pointers - prev, curr, next.

**Steps**:
1. Save next node
2. Reverse current node's pointer
3. Move all pointers forward

### Solution

```go
func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    curr := head
    
    for curr != nil {
        next := curr.Next    // Save next
        curr.Next = prev     // Reverse pointer
        prev = curr          // Move prev forward
        curr = next          // Move curr forward
    }
    
    return prev // New head
}
```

**Time**: $O(n)$, **Space**: $O(1)$

**Common mistake**: Forgetting to save next before changing curr.Next.

---

## Problem 7: Contains Duplicate

**Problem**: Check if array contains any duplicate values.

**Example**:
```
Input: nums = [1,2,3,1]
Output: true
```

### Approach

**Key Insight**: Use hash set to track seen values.

### Solution

```go
func containsDuplicate(nums []int) bool {
    seen := make(map[int]bool)
    
    for _, num := range nums {
        if seen[num] {
            return true
        }
        seen[num] = true
    }
    
    return false
}
```

**Time**: $O(n)$, **Space**: $O(n)$

**Alternative**: Sort first ($O(n \log n)$ time, $O(1)$ space), then check adjacent elements.

---

## Problem 8: Maximum Subarray (Kadane's Algorithm)

**Problem**: Find contiguous subarray with largest sum.

**Example**:
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has sum 6
```

### Approach

**Key Insight**: At each position, decide: extend current subarray or start new one.

**Kadane's Algorithm**: Keep running sum, reset if it goes negative.

### Solution

```go
func maxSubArray(nums []int) int {
    maxSum := nums[0]
    currentSum := nums[0]
    
    for i := 1; i < len(nums); i++ {
        // Either extend current or start new
        currentSum = max(nums[i], currentSum + nums[i])
        maxSum = max(maxSum, currentSum)
    }
    
    return maxSum
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**Time**: $O(n)$, **Space**: $O(1)$

**Why this works**: If current sum becomes negative, it can't help future sums.

---

## Problem 9: Climbing Stairs

**Problem**: How many distinct ways to climb n stairs (1 or 2 steps at a time)?

**Example**:
```
Input: n = 3
Output: 3
Explanation: 1+1+1, 1+2, 2+1
```

### Approach

**Key Insight**: To reach step n, you came from step n-1 or n-2.

**Recurrence**: $f(n) = f(n-1) + f(n-2)$ (Fibonacci!)

### Solution

```go
func climbStairs(n int) int {
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

**Common mistake**: Using recursion without memoization ($O(2^n)$ time).

---

## Problem 10: Binary Tree Maximum Depth

**Problem**: Find maximum depth of binary tree.

**Example**:
```
Input: root = [3,9,20,null,null,15,7]
Output: 3
```

### Approach

**Key Insight**: Depth = 1 + max(left depth, right depth).

**Pattern**: Recursive tree traversal.

### Solution

```go
func maxDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    leftDepth := maxDepth(root.Left)
    rightDepth := maxDepth(root.Right)
    
    return 1 + max(leftDepth, rightDepth)
}
```

**Time**: $O(n)$, **Space**: $O(h)$ where h is height (recursion stack)

**Iterative version**: Use level-order traversal (BFS) and count levels.

---

## Common Patterns in Easy Problems

1. **Hash Map/Set**: Two sum, contains duplicate
2. **Two Pointers**: Valid palindrome, merge lists
3. **Stack**: Valid parentheses
4. **Sliding Window**: Maximum subarray
5. **Dynamic Programming**: Climbing stairs
6. **Tree Recursion**: Maximum depth
7. **Linked List**: Reverse, merge

## Interview Tips

✅ **Do**:
- Ask clarifying questions
- Start with brute force
- Explain your thinking
- Test with examples
- Consider edge cases

❌ **Don't**:
- Jump to code immediately
- Assume constraints
- Stay silent
- Ignore edge cases
- Give up if stuck

## Time Management

- **5 min**: Understand problem, ask questions
- **10 min**: Discuss approach, consider alternatives
- **20 min**: Code solution
- **5 min**: Test and debug

**Total**: ~40 minutes per problem

