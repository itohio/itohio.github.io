---
title: "Graphviz DOT Diagrams"
date: 2024-12-12T19:30:00Z
draft: false
description: "Create complex graph layouts with Graphviz DOT language"
tags: ["graphviz", "dot", "diagram", "graph", "network", "visualization"]
category: "diagrams"
---



Graphviz uses the DOT language to create sophisticated graph layouts. Excellent for complex network diagrams, dependency graphs, state machines, and any graph-based visualization where automatic layout is beneficial.

## Use Case

Use Graphviz when you need to:
- Visualize complex networks or dependencies
- Create state machines
- Show hierarchical structures
- Generate automatic graph layouts
- Visualize data structures (trees, graphs)

## Code

````markdown
```viz-dot
digraph G {
    A -> B;
    B -> C;
    C -> A;
}
```
````

**Result:**

```viz-dot
digraph G {
    A -> B;
    B -> C;
    C -> A;
}
```

## Explanation

- `digraph` - Directed graph (use `graph` for undirected)
- `->` - Directed edge (use `--` for undirected)
- Node names are identifiers
- Attributes in square brackets: `[label="text"]`
- Graph/node/edge attributes control appearance

## Examples

### Example 1: Simple Directed Graph

````markdown
```viz-dot
digraph Dependencies {
    rankdir=LR;
    node [shape=box, style=rounded];
    
    Main -> Parser;
    Main -> Executor;
    Parser -> Lexer;
    Parser -> AST;
    Executor -> AST;
    Executor -> Runtime;
}
```
````

**Result:**

```viz-dot
digraph Dependencies {
    rankdir=LR;
    node [shape=box, style=rounded];
    
    Main -> Parser;
    Main -> Executor;
    Parser -> Lexer;
    Parser -> AST;
    Executor -> AST;
    Executor -> Runtime;
}
```

### Example 2: State Machine

````markdown
```viz-dot
digraph StateMachine {
    rankdir=LR;
    node [shape=circle];
    
    start [shape=point];
    end [shape=doublecircle];
    
    start -> Idle;
    Idle -> Processing [label="start"];
    Processing -> Success [label="complete"];
    Processing -> Error [label="fail"];
    Success -> end;
    Error -> Retry [label="retry"];
    Retry -> Processing;
    Error -> end [label="abort"];
}
```
````

**Result:**

```viz-dot
digraph StateMachine {
    rankdir=LR;
    node [shape=circle];
    
    start [shape=point];
    end [shape=doublecircle];
    
    start -> Idle;
    Idle -> Processing [label="start"];
    Processing -> Success [label="complete"];
    Processing -> Error [label="fail"];
    Success -> end;
    Error -> Retry [label="retry"];
    Retry -> Processing;
    Error -> end [label="abort"];
}
```

### Example 3: Hierarchical Structure

````markdown
```viz-dot
digraph Hierarchy {
    node [shape=box, style="rounded,filled", fillcolor=lightblue];
    
    CEO [fillcolor=gold];
    CTO [fillcolor=lightgreen];
    CFO [fillcolor=lightgreen];
    
    CEO -> CTO;
    CEO -> CFO;
    
    CTO -> "Dev Team Lead";
    CTO -> "QA Team Lead";
    CFO -> "Accounting";
    CFO -> "Finance";
    
    "Dev Team Lead" -> "Developer 1";
    "Dev Team Lead" -> "Developer 2";
    "QA Team Lead" -> "QA Engineer";
}
```
````

**Result:**

```viz-dot
digraph Hierarchy {
    node [shape=box, style="rounded,filled", fillcolor=lightblue];
    
    CEO [fillcolor=gold];
    CTO [fillcolor=lightgreen];
    CFO [fillcolor=lightgreen];
    
    CEO -> CTO;
    CEO -> CFO;
    
    CTO -> "Dev Team Lead";
    CTO -> "QA Team Lead";
    CFO -> "Accounting";
    CFO -> "Finance";
    
    "Dev Team Lead" -> "Developer 1";
    "Dev Team Lead" -> "Developer 2";
    "QA Team Lead" -> "QA Engineer";
}
```

### Example 4: Network Topology

````markdown
```viz-dot
graph Network {
    layout=neato;  // Force-directed layout
    node [shape=box];
    
    Internet [shape=cloud, fillcolor=lightblue, style=filled];
    Router [shape=triangle];
    Switch [shape=box];
    
    Server1 [fillcolor=lightgreen, style=filled];
    Server2 [fillcolor=lightgreen, style=filled];
    PC1 [fillcolor=lightyellow, style=filled];
    PC2 [fillcolor=lightyellow, style=filled];
    PC3 [fillcolor=lightyellow, style=filled];
    
    Internet -- Router [label="WAN"];
    Router -- Switch [label="LAN"];
    Switch -- Server1 [label="1Gbps"];
    Switch -- Server2 [label="1Gbps"];
    Switch -- PC1 [label="100Mbps"];
    Switch -- PC2 [label="100Mbps"];
    Switch -- PC3 [label="100Mbps"];
}
```
````

**Result:**

```viz-dot
graph Network {
    layout=neato;
    node [shape=box];
    
    Internet [shape=cloud, fillcolor=lightblue, style=filled];
    Router [shape=triangle];
    Switch [shape=box];
    
    Server1 [fillcolor=lightgreen, style=filled];
    Server2 [fillcolor=lightgreen, style=filled];
    PC1 [fillcolor=lightyellow, style=filled];
    PC2 [fillcolor=lightyellow, style=filled];
    PC3 [fillcolor=lightyellow, style=filled];
    
    Internet -- Router [label="WAN"];
    Router -- Switch [label="LAN"];
    Switch -- Server1 [label="1Gbps"];
    Switch -- Server2 [label="1Gbps"];
    Switch -- PC1 [label="100Mbps"];
    Switch -- PC2 [label="100Mbps"];
    Switch -- PC3 [label="100Mbps"];
}
```

### Example 5: Data Structure (Binary Tree)

````markdown
```viz-dot
digraph BinaryTree {
    node [shape=circle, style=filled, fillcolor=lightblue];
    
    50 -> 30;
    50 -> 70;
    30 -> 20;
    30 -> 40;
    70 -> 60;
    70 -> 80;
    
    null1 [shape=point];
    null2 [shape=point];
    null3 [shape=point];
    null4 [shape=point];
    
    20 -> null1 [style=dashed];
    20 -> null2 [style=dashed];
    40 -> null3 [style=dashed];
    60 -> null4 [style=dashed];
}
```
````

**Result:**

```viz-dot
digraph BinaryTree {
    node [shape=circle, style=filled, fillcolor=lightblue];
    
    50 -> 30;
    50 -> 70;
    30 -> 20;
    30 -> 40;
    70 -> 60;
    70 -> 80;
    
    null1 [shape=point];
    null2 [shape=point];
    null3 [shape=point];
    null4 [shape=point];
    
    20 -> null1 [style=dashed];
    20 -> null2 [style=dashed];
    40 -> null3 [style=dashed];
    60 -> null4 [style=dashed];
}
```

## Common Attributes

### Node Shapes
- `box`, `circle`, `ellipse`, `triangle`, `diamond`
- `point`, `plaintext`, `record`, `Mrecord`
- `doublecircle`, `house`, `hexagon`, `octagon`

### Layout Engines
- `dot` - Hierarchical (default)
- `neato` - Force-directed
- `fdp` - Force-directed with springs
- `circo` - Circular
- `twopi` - Radial

### Edge Styles
- `solid`, `dashed`, `dotted`, `bold`
- `dir=both` - Bidirectional arrow
- `dir=none` - No arrow

### Colors
- Named colors: `red`, `blue`, `lightgreen`, etc.
- Hex colors: `"#FF0000"`
- RGB: `"#FF0000"`

## Notes

- Use `rankdir=LR` for left-to-right layout (default is top-to-bottom)
- Subgraphs create visual groupings: `subgraph cluster_name {}`
- Use quotes for labels with spaces: `[label="My Node"]`
- Test complex graphs at https://dreampuf.github.io/GraphvizOnline/

## Gotchas/Warnings

- ⚠️ **Layout**: Automatic layout may not match your mental model - iterate
- ⚠️ **Complexity**: Very large graphs can be slow to render
- ⚠️ **Quotes**: Use quotes for identifiers with spaces or special characters
- ⚠️ **Semicolons**: Optional but recommended for clarity