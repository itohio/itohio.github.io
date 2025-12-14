---
title: "String Similarity Algorithms"
date: 2025-12-13
tags: ["string-algorithms", "edit-distance", "lcs", "dynamic-programming", "algorithms"]
---

Algorithms for measuring similarity between strings, used in spell checking, DNA sequencing, plagiarism detection, and fuzzy matching.

## 1. Levenshtein Distance (Edit Distance)

Minimum number of single-character edits (insertions, deletions, substitutions) to transform one string into another.

### How It Works

**Core Idea**: Build a table where `dp[i][j]` represents the minimum edits needed to transform the first `i` characters of string 1 into the first `j` characters of string 2.

**Intuition**:
- If characters match: no edit needed, copy diagonal value
- If different: take minimum of:
  - Delete from s1: `dp[i-1][j] + 1`
  - Insert into s1: `dp[i][j-1] + 1`
  - Substitute: `dp[i-1][j-1] + 1`

**Example**: Transform "kitten" → "sitting"
```
    ""  s  i  t  t  i  n  g
""   0  1  2  3  4  5  6  7
k    1  1  2  3  4  5  6  7
i    2  2  1  2  3  4  5  6
t    3  3  2  1  2  3  4  5
t    4  4  3  2  1  2  3  4
e    5  5  4  3  2  2  3  4
n    6  6  5  4  3  3  2  3
```
Result: 3 edits (k→s, e→i, insert g)

### Dynamic Programming Solution

```go
func LevenshteinDistance(s1, s2 string) int {
    m, n := len(s1), len(s2)
    
    // Create DP table
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    // Initialize base cases
    for i := 0; i <= m; i++ {
        dp[i][0] = i // Delete all characters from s1
    }
    for j := 0; j <= n; j++ {
        dp[0][j] = j // Insert all characters from s2
    }
    
    // Fill DP table
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1] // No operation needed
            } else {
                dp[i][j] = 1 + min(
                    dp[i-1][j],   // Delete from s1
                    dp[i][j-1],   // Insert into s1
                    dp[i-1][j-1], // Substitute
                )
            }
        }
    }
    
    return dp[m][n]
}

func min(a, b, c int) int {
    if a < b {
        if a < c {
            return a
        }
        return c
    }
    if b < c {
        return b
    }
    return c
}

// Example
func main() {
    s1 := "kitten"
    s2 := "sitting"
    dist := LevenshteinDistance(s1, s2)
    fmt.Printf("Distance between '%s' and '%s': %d\n", s1, s2, dist)
    // Output: Distance between 'kitten' and 'sitting': 3
    // (k→s, e→i, insert g)
}
```

**Time**: $O(m \times n)$  
**Space**: $O(m \times n)$

### Space-Optimized Version

```go
func LevenshteinDistanceOptimized(s1, s2 string) int {
    m, n := len(s1), len(s2)
    
    // Only need two rows
    prev := make([]int, n+1)
    curr := make([]int, n+1)
    
    // Initialize first row
    for j := 0; j <= n; j++ {
        prev[j] = j
    }
    
    for i := 1; i <= m; i++ {
        curr[0] = i
        
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                curr[j] = prev[j-1]
            } else {
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
            }
        }
        
        prev, curr = curr, prev
    }
    
    return prev[n]
}
```

**Space**: $O(n)$

## 2. Longest Common Subsequence (LCS)

Find the longest subsequence present in both strings (not necessarily contiguous).

### How It Works

**Core Idea**: Build a table where `dp[i][j]` represents the length of LCS of first `i` characters of s1 and first `j` characters of s2.

**Intuition**:
- If characters match: extend previous LCS by 1
- If different: take maximum from either excluding current char from s1 or s2

**Example**: "ABCDGH" and "AEDFHR"
```
    ""  A  E  D  F  H  R
""   0  0  0  0  0  0  0
A    0  1  1  1  1  1  1
B    0  1  1  1  1  1  1
C    0  1  1  1  1  1  1
D    0  1  1  2  2  2  2
G    0  1  1  2  2  2  2
H    0  1  1  2  2  3  3
```
LCS: "ADH" (length 3)

**Key Difference from Edit Distance**: We're finding common elements, not transforming one to another.

### Dynamic Programming Solution

```go
func LongestCommonSubsequence(s1, s2 string) int {
    m, n := len(s1), len(s2)
    
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }
    
    return dp[m][n]
}

// Get the actual LCS string
func GetLCS(s1, s2 string) string {
    m, n := len(s1), len(s2)
    
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    // Fill DP table
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }
    
    // Backtrack to find LCS
    lcs := []byte{}
    i, j := m, n
    
    for i > 0 && j > 0 {
        if s1[i-1] == s2[j-1] {
            lcs = append([]byte{s1[i-1]}, lcs...)
            i--
            j--
        } else if dp[i-1][j] > dp[i][j-1] {
            i--
        } else {
            j--
        }
    }
    
    return string(lcs)
}

// Example
func main() {
    s1 := "ABCDGH"
    s2 := "AEDFHR"
    fmt.Printf("LCS length: %d\n", LongestCommonSubsequence(s1, s2))
    fmt.Printf("LCS: %s\n", GetLCS(s1, s2))
    // Output:
    // LCS length: 3
    // LCS: ADH
}
```

**Time**: $O(m \times n)$  
**Space**: $O(m \times n)$

## 3. Longest Common Substring

Find the longest contiguous substring present in both strings.

### How It Works

**Core Idea**: Similar to LCS but requires consecutive matches. Reset to 0 when characters don't match.

**Intuition**:
- If characters match: extend current substring length
- If different: reset to 0 (must be contiguous)
- Track maximum length and ending position

**Example**: "OldSite:GeeksforGeeks.org" and "NewSite:GeeksQuiz.com"
- Common substrings: "Site:", "Geeks"
- Longest: "Site:Geeks" (length 11)

**Key Difference from LCS**: Must be consecutive characters, not just subsequence.

```go
func LongestCommonSubstring(s1, s2 string) string {
    m, n := len(s1), len(s2)
    
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    maxLen := 0
    endIndex := 0
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
                
                if dp[i][j] > maxLen {
                    maxLen = dp[i][j]
                    endIndex = i
                }
            }
        }
    }
    
    if maxLen == 0 {
        return ""
    }
    
    return s1[endIndex-maxLen : endIndex]
}

// Example
func main() {
    s1 := "OldSite:GeeksforGeeks.org"
    s2 := "NewSite:GeeksQuiz.com"
    fmt.Printf("Longest common substring: %s\n", LongestCommonSubstring(s1, s2))
    // Output: Longest common substring: Site:Geeks
}
```

**Time**: $O(m \times n)$  
**Space**: $O(m \times n)$

## 4. Hamming Distance

Number of positions at which corresponding symbols differ (strings must be same length).

### How It Works

**Core Idea**: Simply count positions where characters differ.

**Intuition**: Used for fixed-length codes (error detection, DNA sequences).

**Example**: "karolin" vs "kathrin"
```
k a r o l i n
k a t h r i n
✓ ✓ ✗ ✗ ✗ ✓ ✓
```
Distance = 3 (positions 2, 3, 4 differ)

**Use Case**: Error-correcting codes, detecting bit flips.

```go
func HammingDistance(s1, s2 string) int {
    if len(s1) != len(s2) {
        return -1 // Invalid
    }
    
    distance := 0
    for i := 0; i < len(s1); i++ {
        if s1[i] != s2[i] {
            distance++
        }
    }
    
    return distance
}

// Example
func main() {
    s1 := "karolin"
    s2 := "kathrin"
    fmt.Printf("Hamming distance: %d\n", HammingDistance(s1, s2))
    // Output: Hamming distance: 3
    // (k-k, a-a, r-t, o-h, l-r, i-i, n-n)
}
```

**Time**: $O(n)$  
**Space**: $O(1)$

## 5. Jaro-Winkler Distance

Measures similarity between two strings, giving more weight to common prefixes.

### How It Works

**Core Idea**: Jaro distance considers matching characters within a window, then Winkler adds bonus for common prefix.

**Jaro Distance Formula**:
$$
\text{Jaro} = \frac{1}{3}\left(\frac{m}{|s_1|} + \frac{m}{|s_2|} + \frac{m-t}{m}\right)
$$
where:
- $m$ = number of matching characters
- $t$ = number of transpositions (matching chars in wrong order)
- Match window = $\max(|s_1|, |s_2|) / 2 - 1$

**Jaro-Winkler**: Adds prefix bonus
$$
\text{JW} = \text{Jaro} + (l \times p \times (1 - \text{Jaro}))
$$
where:
- $l$ = length of common prefix (max 4)
- $p$ = scaling factor (typically 0.1)

**Example**: "MARTHA" vs "MARHTA"
- Matches: M, A, R, H, T, A (all 6)
- Transpositions: T and H swapped (1 transposition)
- Common prefix: "MAR" (length 3)
- Jaro ≈ 0.944, Jaro-Winkler ≈ 0.961

**Use Case**: Record linkage, name matching (tolerates typos).

```go
func JaroDistance(s1, s2 string) float64 {
    if s1 == s2 {
        return 1.0
    }
    
    len1, len2 := len(s1), len(s2)
    if len1 == 0 || len2 == 0 {
        return 0.0
    }
    
    // Maximum distance for matches
    matchDistance := max(len1, len2)/2 - 1
    if matchDistance < 0 {
        matchDistance = 0
    }
    
    s1Matches := make([]bool, len1)
    s2Matches := make([]bool, len2)
    
    matches := 0
    transpositions := 0
    
    // Find matches
    for i := 0; i < len1; i++ {
        start := max(0, i-matchDistance)
        end := min(i+matchDistance+1, len2)
        
        for j := start; j < end; j++ {
            if s2Matches[j] || s1[i] != s2[j] {
                continue
            }
            s1Matches[i] = true
            s2Matches[j] = true
            matches++
            break
        }
    }
    
    if matches == 0 {
        return 0.0
    }
    
    // Count transpositions
    k := 0
    for i := 0; i < len1; i++ {
        if !s1Matches[i] {
            continue
        }
        for !s2Matches[k] {
            k++
        }
        if s1[i] != s2[k] {
            transpositions++
        }
        k++
    }
    
    jaro := (float64(matches)/float64(len1) +
        float64(matches)/float64(len2) +
        float64(matches-transpositions/2)/float64(matches)) / 3.0
    
    return jaro
}

func JaroWinklerDistance(s1, s2 string) float64 {
    jaro := JaroDistance(s1, s2)
    
    // Find common prefix length (max 4)
    prefixLen := 0
    for i := 0; i < min(len(s1), len(s2), 4); i++ {
        if s1[i] == s2[i] {
            prefixLen++
        } else {
            break
        }
    }
    
    // Jaro-Winkler = Jaro + (prefix_length * 0.1 * (1 - Jaro))
    return jaro + float64(prefixLen)*0.1*(1.0-jaro)
}

// Example
func main() {
    s1 := "MARTHA"
    s2 := "MARHTA"
    fmt.Printf("Jaro-Winkler distance: %.4f\n", JaroWinklerDistance(s1, s2))
    // Output: Jaro-Winkler distance: 0.9611
}
```

**Time**: $O(m \times n)$  
**Space**: $O(m + n)$

## 6. Damerau-Levenshtein Distance

Edit distance allowing insertions, deletions, substitutions, and **transpositions** of adjacent characters.

### How It Works

**Core Idea**: Like Levenshtein but also allows swapping adjacent characters (transposition).

**Intuition**: Accounts for common typo - swapping two adjacent letters.

**Example**: "CA" → "ABC"
- Levenshtein: 3 (delete C, delete A, insert A, B, C)
- Damerau-Levenshtein: 2 (insert A before, insert B after)

**Operations**:
1. Insert
2. Delete
3. Substitute
4. **Transpose** (swap adjacent) - this is the addition

**Use Case**: Spell checkers (80% of typos are single-character edits or transpositions).

```go
func DamerauLevenshteinDistance(s1, s2 string) int {
    m, n := len(s1), len(s2)
    
    // Create DP table with extra row/column
    dp := make([][]int, m+2)
    for i := range dp {
        dp[i] = make([]int, n+2)
    }
    
    maxDist := m + n
    dp[0][0] = maxDist
    
    for i := 0; i <= m; i++ {
        dp[i+1][0] = maxDist
        dp[i+1][1] = i
    }
    for j := 0; j <= n; j++ {
        dp[0][j+1] = maxDist
        dp[1][j+1] = j
    }
    
    da := make(map[byte]int)
    
    for i := 1; i <= m; i++ {
        db := 0
        
        for j := 1; j <= n; j++ {
            k := da[s2[j-1]]
            l := db
            cost := 1
            
            if s1[i-1] == s2[j-1] {
                cost = 0
                db = j
            }
            
            dp[i+1][j+1] = min(
                dp[i][j]+cost,           // Substitution
                dp[i+1][j]+1,            // Insertion
                dp[i][j+1]+1,            // Deletion
                dp[k][l]+(i-k-1)+1+(j-l-1), // Transposition
            )
        }
        
        da[s1[i-1]] = i
    }
    
    return dp[m+1][n+1]
}

func min(vals ...int) int {
    minVal := vals[0]
    for _, v := range vals[1:] {
        if v < minVal {
            minVal = v
        }
    }
    return minVal
}

// Example
func main() {
    s1 := "CA"
    s2 := "ABC"
    fmt.Printf("Damerau-Levenshtein distance: %d\n", DamerauLevenshteinDistance(s1, s2))
    // Output: Damerau-Levenshtein distance: 2
}
```

**Time**: $O(m \times n)$  
**Space**: $O(m \times n)$

## 7. Cosine Similarity

Measures similarity based on character frequency (bag of words).

### How It Works

**Core Idea**: Treat strings as vectors of character frequencies, measure angle between vectors.

**Intuition**: Ignores order, focuses on character distribution.

**Formula**:
$$
\text{Cosine} = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \times ||\mathbf{B}||} = \frac{\sum A_i B_i}{\sqrt{\sum A_i^2} \times \sqrt{\sum B_i^2}}
$$

**Example**: "hello" vs "olleh"
- Vector for "hello": {h:1, e:1, l:2, o:1}
- Vector for "olleh": {o:1, l:2, e:1, h:1}
- Same frequencies → Cosine = 1.0 (identical)

**Key Property**: Order-independent, perfect for anagrams.

**Use Case**: Document similarity, plagiarism detection.

```go
import "math"

func CosineSimilarity(s1, s2 string) float64 {
    // Build frequency maps
    freq1 := make(map[rune]int)
    freq2 := make(map[rune]int)
    
    for _, ch := range s1 {
        freq1[ch]++
    }
    for _, ch := range s2 {
        freq2[ch]++
    }
    
    // Calculate dot product and magnitudes
    dotProduct := 0.0
    mag1 := 0.0
    mag2 := 0.0
    
    // Get all unique characters
    allChars := make(map[rune]bool)
    for ch := range freq1 {
        allChars[ch] = true
    }
    for ch := range freq2 {
        allChars[ch] = true
    }
    
    for ch := range allChars {
        f1 := float64(freq1[ch])
        f2 := float64(freq2[ch])
        
        dotProduct += f1 * f2
        mag1 += f1 * f1
        mag2 += f2 * f2
    }
    
    if mag1 == 0 || mag2 == 0 {
        return 0.0
    }
    
    return dotProduct / (math.Sqrt(mag1) * math.Sqrt(mag2))
}

// Example
func main() {
    s1 := "hello world"
    s2 := "hello there"
    fmt.Printf("Cosine similarity: %.4f\n", CosineSimilarity(s1, s2))
}
```

**Time**: $O(m + n)$  
**Space**: $O(m + n)$

## 8. Jaccard Similarity

Measures similarity as intersection over union of character sets.

### How It Works

**Core Idea**: Compare unique character sets using set operations.

**Formula**:
$$
\text{Jaccard} = \frac{|A \cap B|}{|A \cup B|}
$$

**Intuition**: What fraction of all unique characters appear in both strings?

**Example**: "night" vs "nacht"
- Set A: {n, i, g, h, t}
- Set B: {n, a, c, h, t}
- Intersection: {n, h, t} (3 chars)
- Union: {n, i, g, h, t, a, c} (7 chars)
- Jaccard = 3/7 ≈ 0.43

**Properties**:
- Range: [0, 1]
- 0 = no common characters
- 1 = identical character sets

**Use Case**: Set similarity, recommendation systems.

```go
func JaccardSimilarity(s1, s2 string) float64 {
    set1 := make(map[rune]bool)
    set2 := make(map[rune]bool)
    
    for _, ch := range s1 {
        set1[ch] = true
    }
    for _, ch := range s2 {
        set2[ch] = true
    }
    
    // Calculate intersection
    intersection := 0
    for ch := range set1 {
        if set2[ch] {
            intersection++
        }
    }
    
    // Calculate union
    union := len(set1) + len(set2) - intersection
    
    if union == 0 {
        return 0.0
    }
    
    return float64(intersection) / float64(union)
}

// Example
func main() {
    s1 := "night"
    s2 := "nacht"
    fmt.Printf("Jaccard similarity: %.4f\n", JaccardSimilarity(s1, s2))
    // Output: Jaccard similarity: 0.2857 (2 common chars / 7 unique chars)
}
```

**Time**: $O(m + n)$  
**Space**: $O(m + n)$

## Comparison Table

| Algorithm | Time | Space | Use Case |
|-----------|------|-------|----------|
| Levenshtein | $O(mn)$ | $O(mn)$ or $O(n)$ | General edit distance, spell check |
| LCS | $O(mn)$ | $O(mn)$ | Diff tools, DNA sequencing |
| LCS Substring | $O(mn)$ | $O(mn)$ | Find common patterns |
| Hamming | $O(n)$ | $O(1)$ | Equal-length strings, error detection |
| Jaro-Winkler | $O(mn)$ | $O(m+n)$ | Record linkage, name matching |
| Damerau-Levenshtein | $O(mn)$ | $O(mn)$ | Typo detection (transpositions) |
| Cosine | $O(m+n)$ | $O(m+n)$ | Document similarity |
| Jaccard | $O(m+n)$ | $O(m+n)$ | Set similarity |

## Practical Applications

### Spell Checker

```go
func FindClosestWord(word string, dictionary []string, threshold int) []string {
    suggestions := []string{}
    
    for _, dictWord := range dictionary {
        dist := LevenshteinDistance(word, dictWord)
        if dist <= threshold {
            suggestions = append(suggestions, dictWord)
        }
    }
    
    return suggestions
}
```

### Fuzzy String Matching

```go
func FuzzyMatch(query string, candidates []string, threshold float64) []string {
    matches := []string{}
    
    for _, candidate := range candidates {
        similarity := JaroWinklerDistance(query, candidate)
        if similarity >= threshold {
            matches = append(matches, candidate)
        }
    }
    
    return matches
}
```

### DNA Sequence Alignment

```go
func AlignSequences(seq1, seq2 string) int {
    return LongestCommonSubsequence(seq1, seq2)
}
```

## When to Use Each Algorithm

✅ **Levenshtein**: General purpose, spell checking  
✅ **LCS**: Version control diffs, plagiarism detection  
✅ **Hamming**: Error detection in equal-length codes  
✅ **Jaro-Winkler**: Name matching, record linkage  
✅ **Damerau-Levenshtein**: Typo detection with transpositions  
✅ **Cosine**: Document similarity, text classification  
✅ **Jaccard**: Set similarity, recommendation systems

