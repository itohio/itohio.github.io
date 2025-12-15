---
title: "Mermaid Arbitrary Graphs"
date: 2024-12-12T21:00:00Z
draft: false
description: "Create arbitrary graphs and networks with Mermaid"
tags: ["mermaid", "graph", "network", "diagram", "visualization", "diagrams"]
category: "diagrams"
---

Mermaid supports creating arbitrary graphs (both directed and undirected) for visualizing networks, relationships, and complex graph structures. Perfect for network topologies, social graphs, dependency graphs, and any graph-based visualization.

## Use Case

Use Mermaid graphs when you need to:
- Visualize network topologies
- Show relationships between entities
- Create dependency graphs
- Model social networks
- Display graph data structures
- Show arbitrary connections between nodes

## Basic Syntax

### Directed Graph

````markdown
```mermaid
graph TD
    A --> B
    B --> C
    C --> A
```
````

**Result:**

```mermaid
graph TD
    A --> B
    B --> C
    C --> A
```

### Undirected Graph

````markdown
```mermaid
graph LR
    A --- B
    B --- C
    C --- A
```
````

**Result:**

```mermaid
graph LR
    A --- B
    B --- C
    C --- A
```

## Examples

### Example 1: Network Topology

````markdown
```mermaid
graph TB
    Internet[Internet]
    Router[Router]
    Switch[Switch]
    Server1[Server 1]
    Server2[Server 2]
    PC1[PC 1]
    PC2[PC 2]
    
    Internet --> Router
    Router --> Switch
    Switch --> Server1
    Switch --> Server2
    Switch --> PC1
    Switch --> PC2
```
````

**Result:**

```mermaid
graph TB
    Internet[Internet]
    Router[Router]
    Switch[Switch]
    Server1[Server 1]
    Server2[Server 2]
    PC1[PC 1]
    PC2[PC 2]
    
    Internet --> Router
    Router --> Switch
    Switch --> Server1
    Switch --> Server2
    Switch --> PC1
    Switch --> PC2
```

### Example 2: Social Network Graph

````markdown
```mermaid
graph LR
    Alice[Alice]
    Bob[Bob]
    Charlie[Charlie]
    Diana[Diana]
    Eve[Eve]
    
    Alice --- Bob
    Alice --- Charlie
    Bob --- Diana
    Charlie --- Eve
    Diana --- Eve
    Bob --- Eve
```
````

**Result:**

```mermaid
graph LR
    Alice[Alice]
    Bob[Bob]
    Charlie[Charlie]
    Diana[Diana]
    Eve[Eve]
    
    Alice --- Bob
    Alice --- Charlie
    Bob --- Diana
    Charlie --- Eve
    Diana --- Eve
    Bob --- Eve
```

### Example 3: Dependency Graph

````markdown
```mermaid
graph TD
    A[Module A] --> B[Module B]
    A --> C[Module C]
    B --> D[Module D]
    C --> D
    D --> E[Module E]
    B --> F[Module F]
    C --> F
```
````

**Result:**

```mermaid
graph TD
    A[Module A] --> B[Module B]
    A --> C[Module C]
    B --> D[Module D]
    C --> D
    D --> E[Module E]
    B --> F[Module F]
    C --> F
```

### Example 4: Weighted Graph

````markdown
```mermaid
graph LR
    A[A] -->|5| B[B]
    A -->|3| C[C]
    B -->|2| D[D]
    C -->|4| D
    B -->|1| E[E]
    D -->|6| E
```
````

**Result:**

```mermaid
graph LR
    A[A] -->|5| B[B]
    A -->|3| C[C]
    B -->|2| D[D]
    C -->|4| D
    B -->|1| E[E]
    D -->|6| E
```

### Example 5: Complex Graph with Styling

````markdown
```mermaid
graph TB
    Start([Start]) --> A{Decision}
    A -->|Yes| B[Process 1]
    A -->|No| C[Process 2]
    B --> D[End]
    C --> D
    
    style Start fill:#90EE90
    style D fill:#FFB6C1
    style A fill:#87CEEB
```
````

**Result:**

```mermaid
graph TB
    Start([Start]) --> A{Decision}
    A -->|Yes| B[Process 1]
    A -->|No| C[Process 2]
    B --> D[End]
    C --> D
    
    style Start fill:#90EE90
    style D fill:#FFB6C1
    style A fill:#87CEEB
```

### Example 6: Graph with Subgraphs

````markdown
```mermaid
graph TB
    subgraph Cluster1[Cluster 1]
        A1[A1]
        A2[A2]
        A3[A3]
    end
    
    subgraph Cluster2[Cluster 2]
        B1[B1]
        B2[B2]
    end
    
    A1 --> A2
    A2 --> A3
    B1 --> B2
    A3 --> B1
```
````

**Result:**

```mermaid
graph TB
    subgraph Cluster1[Cluster 1]
        A1[A1]
        A2[A2]
        A3[A3]
    end
    
    subgraph Cluster2[Cluster 2]
        B1[B1]
        B2[B2]
    end
    
    A1 --> A2
    A2 --> A3
    B1 --> B2
    A3 --> B1
```

### Example 7: Multi-directional Graph

````markdown
```mermaid
graph LR
    A[A] <--> B[B]
    B --> C[C]
    C -.->|dashed| D[D]
    D ==>|thick| E[E]
    E -->|normal| A
```
````

**Result:**

```mermaid
graph LR
    A[A] <--> B[B]
    B --> C[C]
    C -.->|dashed| D[D]
    D ==>|thick| E[E]
    E -->|normal| A
```

## Edge Types

### Directed Edges

- `A --> B` - Solid arrow
- `A ==> B` - Thick arrow
- `A -.-> B` - Dashed arrow
- `A -..-> B` - Dotted arrow

### Undirected Edges

- `A --- B` - Solid line
- `A === B` - Thick line
- `A -.- B` - Dashed line
- `A -..- B` - Dotted line

### Bidirectional

- `A <--> B` - Bidirectional solid
- `A <==> B` - Bidirectional thick

### With Labels

- `A -->|label| B` - Edge with label
- `A ---|label| B` - Undirected edge with label

## Node Shapes

- `A[Rectangle]` - Rectangle
- `A(Rounded)` - Rounded rectangle
- `A([Stadium])` - Stadium shape
- `A[[Subroutine]]` - Double rectangle
- `A[(Database)]` - Cylinder
- `A((Circle))` - Circle
- `A>Asymmetric]` - Asymmetric shape
- `A{Diamond}` - Diamond
- `A{{Hexagon}}` - Hexagon

## Graph Directions

- `graph TD` - Top to bottom
- `graph TB` - Top to bottom (same as TD)
- `graph BT` - Bottom to top
- `graph LR` - Left to right
- `graph RL` - Right to left

## Styling

### Node Styling

````markdown
```mermaid
graph LR
    A[A] --> B[B]
    C[C] --> D[D]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:4px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbf,stroke:#333,stroke-width:2px
```
````

**Result:**

```mermaid
graph LR
    A[A] --> B[B]
    C[C] --> D[D]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:4px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbf,stroke:#333,stroke-width:2px
```

### Class-based Styling

````markdown
```mermaid
graph TD
    A[Node A] --> B[Node B]
    C[Node C] --> D[Node D]
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px
    classDef highlight fill:#ff6,stroke:#333,stroke-width:4px
    
    class A,C highlight
```
````

**Result:**

```mermaid
graph TD
    A[Node A] --> B[Node B]
    C[Node C] --> D[Node D]
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px
    classDef highlight fill:#ff6,stroke:#333,stroke-width:4px
    
    class A,C highlight
```

## Notes

- Graphs automatically layout nodes - you can't control exact positions
- Use subgraphs to group related nodes visually
- Edge labels can contain text and some HTML
- Styling supports CSS color formats (hex, rgb, named colors)
- Complex graphs may need optimization for readability

## Gotchas/Warnings

- ⚠️ **Layout**: Automatic layout - nodes position themselves
- ⚠️ **Complexity**: Very large graphs can be slow to render
- ⚠️ **Node IDs**: Must be unique and simple (alphanumeric, no spaces)
- ⚠️ **Edge Labels**: Keep labels short for readability
- ⚠️ **Subgraphs**: Must start with `subgraph` keyword
- ⚠️ **Styling**: CSS color names work, but hex is more reliable

