---
title: "General Tree"
date: 2025-12-13
tags: ["tree", "data-structures", "n-ary-tree", "graph"]
---

A **tree** is a hierarchical data structure consisting of nodes connected by edges, with one node designated as the root. Unlike binary trees, general trees allow nodes to have any number of children.

## Definitions

### Basic Terminology

**Root**: The topmost node with no parent

**Parent**: A node that has one or more child nodes

**Child**: A node descended from another node

**Siblings**: Nodes with the same parent

**Leaf (External Node)**: A node with no children

**Internal Node**: A node with at least one child

**Edge**: Connection between two nodes

**Path**: Sequence of nodes connected by edges

**Depth of Node**: Number of edges from root to the node

**Height of Node**: Number of edges on longest path from node to a leaf

**Height of Tree**: Height of the root node

**Level**: All nodes at the same depth

**Degree of Node**: Number of children of the node

**Degree of Tree**: Maximum degree among all nodes

**Subtree**: A tree formed by a node and all its descendants

## Properties

### Fundamental Properties

For a tree with \( n \) nodes:

**Number of edges**:
\[
\text{edges} = n - 1
\]

**Relationship between nodes and edges**:
\[
n = e + 1
\]

where \( e \) is the number of edges.

### Height and Depth

**Minimum height** (when tree is maximally wide):
\[
h_{\min} = 1
\]

**Maximum height** (when tree is a chain):
\[
h_{\max} = n - 1
\]

For a tree where each node has at most \( k \) children:

**Minimum height** (complete k-ary tree):
\[
h_{\min} = \lceil \log_k n \rceil
\]

**Maximum nodes** at height \( h \):
\[
n_{\max} = \frac{k^{h+1} - 1}{k - 1}
\]

## Types of Trees

### 1. N-ary Tree (K-ary Tree)
Each node has at most \( k \) children.

```
       1
    /  |  \
   2   3   4
  /|\  |
 5 6 7 8
```

**Properties**:
- Generalization of binary tree (\( k = 2 \))
- Height: \( O(\log_k n) \) for balanced tree

### 2. Trie (Prefix Tree)
Tree where each path from root represents a string.

**Applications**:
- Autocomplete
- Spell checking
- IP routing tables

**Complexity**:
- Insert/Search: \( O(m) \) where \( m \) is string length
- Space: \( O(\text{ALPHABET\_SIZE} \times m \times n) \)

### 3. Suffix Tree
Compressed trie of all suffixes of a string.

**Applications**:
- Pattern matching
- Longest common substring
- Genome analysis

**Properties**:
- Build time: \( O(n) \) with Ukkonen's algorithm
- Space: \( O(n) \)
- Pattern search: \( O(m) \) where \( m \) is pattern length

### 4. B-Tree
Self-balancing tree optimized for disk access.

**Properties**:
- Each node can have multiple keys and children
- All leaves at same level
- Minimum degree \( t \): each node has at least \( t-1 \) keys

**Applications**:
- Database indexing
- File systems

**Complexity**:
- Search/Insert/Delete: \( O(\log_t n) \)

### 5. Segment Tree
Tree for storing intervals or segments.

**Applications**:
- Range queries (sum, min, max)
- Range updates

**Properties**:
- Height: \( O(\log n) \)
- Space: \( O(4n) \approx O(n) \)
- Query/Update: \( O(\log n) \)

### 6. Fenwick Tree (Binary Indexed Tree)
Efficient structure for cumulative frequency tables.

**Applications**:
- Prefix sums
- Frequency counting

**Properties**:
- Space: \( O(n) \)
- Update/Query: \( O(\log n) \)
- More space-efficient than segment tree

## Tree Representations

### 1. Array of Children (Adjacency List)
Each node stores list of its children.

```python
class Node:
    value: any
    children: List[Node]
```

**Space**: \( O(n) \)

### 2. Left-Child Right-Sibling
Convert general tree to binary tree representation.

```python
class Node:
    value: any
    left_child: Node      # First child
    right_sibling: Node   # Next sibling
```

**Space**: \( O(n) \)

**Advantage**: Uniform structure, easier to implement

### 3. Parent Pointer
Each node stores reference to its parent.

```python
class Node:
    value: any
    parent: Node
```

**Use case**: Union-Find, upward traversal

### 4. Array Representation (for complete k-ary tree)
Similar to binary heap representation.

For node at index \( i \):
- **Parent**: \( \lfloor (i-1)/k \rfloor \)
- **j-th child** (0-indexed): \( ki + j + 1 \)

## Tree Traversals

### 1. Depth-First Search (DFS)

#### Pre-order (Root → Children)
\[
\text{PreOrder}(node) = \text{visit}(node) \to \text{PreOrder}(\text{child}_1) \to \cdots \to \text{PreOrder}(\text{child}_k)
\]

**Use case**: Copy tree, serialize tree

#### Post-order (Children → Root)
\[
\text{PostOrder}(node) = \text{PostOrder}(\text{child}_1) \to \cdots \to \text{PostOrder}(\text{child}_k) \to \text{visit}(node)
\]

**Use case**: Delete tree, calculate directory sizes, evaluate expression trees

**Time**: \( O(n) \)  
**Space**: \( O(h) \) for recursion stack

### 2. Breadth-First Search (BFS)

#### Level-order
Visit nodes level by level.

**Implementation**: Use queue

**Time**: \( O(n) \)  
**Space**: \( O(w) \) where \( w \) is maximum width

## Common Operations

### 1. Height Calculation

\[
\text{height}(node) = \begin{cases}
0 & \text{if node is leaf} \\
1 + \max_{c \in \text{children}} \text{height}(c) & \text{otherwise}
\end{cases}
\]

**Time**: \( O(n) \)

### 2. Size (Count Nodes)

\[
\text{size}(node) = 1 + \sum_{c \in \text{children}} \text{size}(c)
\]

**Time**: \( O(n) \)

### 3. Depth of Node

\[
\text{depth}(node) = \begin{cases}
0 & \text{if node is root} \\
1 + \text{depth}(\text{parent}) & \text{otherwise}
\end{cases}
\]

**Time**: \( O(h) \)

### 4. Lowest Common Ancestor (LCA)

Find the deepest node that is ancestor of both given nodes.

**Approaches**:
1. **Path comparison**: Find paths from root to both nodes, compare
   - Time: \( O(n) \)
2. **Binary lifting**: Preprocess with parent pointers at powers of 2
   - Preprocess: \( O(n \log n) \)
   - Query: \( O(\log n) \)
3. **Euler tour + RMQ**: Convert to range minimum query
   - Preprocess: \( O(n) \)
   - Query: \( O(1) \)

### 5. Diameter

Longest path between any two nodes.

**Approach**:
1. Run DFS from any node to find farthest node \( u \)
2. Run DFS from \( u \) to find farthest node \( v \)
3. Distance from \( u \) to \( v \) is diameter

**Time**: \( O(n) \)

## Complexity Summary

| Operation | Time | Space |
|-----------|------|-------|
| Traversal (DFS/BFS) | \( O(n) \) | \( O(h) \) or \( O(w) \) |
| Height | \( O(n) \) | \( O(h) \) |
| Size | \( O(n) \) | \( O(h) \) |
| Search | \( O(n) \) | \( O(h) \) |
| Insert | \( O(1) \)* | \( O(1) \) |
| Delete | \( O(n) \)** | \( O(h) \) |

\* If you have reference to parent node  
\** Need to find node first, then remove and reconnect children

## Common Interview Problems

### 1. Serialize and Deserialize N-ary Tree
Convert tree to string and reconstruct.

**Approach**: Pre-order with level markers or child count.

**Time**: \( O(n) \)

### 2. Maximum Depth
Find the deepest level.

**Approach**: Recursive DFS or iterative BFS.

**Time**: \( O(n) \)

### 3. Level Order Traversal
Return nodes grouped by level.

**Approach**: BFS with level tracking.

**Time**: \( O(n) \)

### 4. Zigzag Level Order
Alternate left-to-right and right-to-left by level.

**Approach**: BFS with alternating reversal.

**Time**: \( O(n) \)

### 5. Diameter of N-ary Tree
Longest path between any two nodes.

**Approach**: For each node, find two longest paths to leaves through different children.

**Time**: \( O(n) \)

### 6. Clone Tree
Create deep copy of tree.

**Approach**: Pre-order traversal with hash map for node mapping.

**Time**: \( O(n) \)

### 7. Find All Paths from Root to Leaves
Return all root-to-leaf paths.

**Approach**: DFS with path tracking.

**Time**: \( O(n) \)

### 8. Vertical Order Traversal
Group nodes by horizontal distance from root.

**Approach**: DFS/BFS with horizontal distance tracking, use hash map.

**Time**: \( O(n \log n) \)

## Tree vs. Graph

| Property | Tree | Graph |
|----------|------|-------|
| Cycles | No cycles | May have cycles |
| Edges | \( n - 1 \) | Any number |
| Root | Has one root | No specific root |
| Path | Unique path between any two nodes | Multiple paths possible |
| Connectivity | Always connected | May be disconnected |
| Hierarchy | Hierarchical | Non-hierarchical |

**Note**: A tree is a special case of a graph (connected acyclic graph).

## Applications

### File Systems
Directory structure is a tree where:
- Root is the root directory
- Directories are internal nodes
- Files are leaves

### Organization Charts
Company hierarchy:
- CEO at root
- Departments and employees as nodes

### XML/HTML DOM
Document structure:
- Root element
- Nested elements as children

### Decision Trees
Machine learning:
- Internal nodes: decision rules
- Leaves: predictions

### Game Trees
AI game playing:
- Root: current state
- Children: possible moves
- Minimax algorithm for optimal play

### Expression Trees
Mathematical expressions:
- Operators as internal nodes
- Operands as leaves

### Routing Tables
Network routing:
- Trie for IP prefix matching

## Advanced Concepts

### Tree Isomorphism
Two trees are isomorphic if they have the same structure.

**Check**: Canonical form comparison or hash-based approach.

**Time**: \( O(n) \) with proper hashing

### Centroid Decomposition
Recursively partition tree by removing centroids.

**Applications**:
- Path queries
- Subtree queries

**Time**: \( O(n \log n) \) preprocessing

### Heavy-Light Decomposition
Partition tree into heavy and light edges.

**Applications**:
- Path queries with updates
- LCA queries

**Time**: \( O(\log n) \) per query after \( O(n) \) preprocessing

### Euler Tour Technique
Convert tree to array using DFS.

**Applications**:
- Subtree queries
- Path queries with segment trees

**Time**: \( O(n) \) preprocessing, \( O(\log n) \) queries

