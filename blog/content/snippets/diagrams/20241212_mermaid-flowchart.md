---
title: "Mermaid Flowchart"
date: 2024-12-12T18:00:00Z
draft: false
description: "Create flowcharts and decision trees with Mermaid"
type: "snippet"
tags: ["mermaid", "flowchart", "diagram", "visualization", "diagrams"]
category: "diagrams"
---



Mermaid flowcharts are perfect for visualizing processes, algorithms, and decision trees. They're text-based, version-control friendly, and render beautifully.

## Use Case

Use flowcharts when you need to:
- Document algorithms or processes
- Show decision logic
- Map out workflows
- Visualize state transitions

## Code

````markdown
```mermaid
graph TD
    A[Start] --> B{Decision?}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E
```
````

## Explanation

- `graph TD` - Top-down flowchart (use `LR` for left-right, `BT` for bottom-top)
- `[]` - Rectangle node
- `{}` - Diamond (decision) node
- `()` - Rounded rectangle
- `-->` - Arrow
- `-->|text|` - Arrow with label

## Examples

### Example 1: Research Process

````markdown
```mermaid
graph TD
    A[Research Question] --> B{Literature<br/>Review Done?}
    B -->|No| C[Read Papers]
    C --> B
    B -->|Yes| D[Form Hypothesis]
    D --> E[Design Experiment]
    E --> F[Run Experiment]
    F --> G{Results<br/>Significant?}
    G -->|Yes| H[Document Findings]
    G -->|No| I[Refine Approach]
    I --> E
    H --> J[Publish]
```
````

**Result:**

```mermaid
graph TD
    A[Research Question] --> B{Literature<br/>Review Done?}
    B -->|No| C[Read Papers]
    C --> B
    B -->|Yes| D[Form Hypothesis]
    D --> E[Design Experiment]
    E --> F[Run Experiment]
    F --> G{Results<br/>Significant?}
    G -->|Yes| H[Document Findings]
    G -->|No| I[Refine Approach]
    I --> E
    H --> J[Publish]
```

### Example 2: Algorithm Flow

````markdown
```mermaid
graph LR
    A[Input Data] --> B[Preprocess]
    B --> C[Feature Extraction]
    C --> D[Model Training]
    D --> E[Validation]
    E --> F{Accuracy > 95%?}
    F -->|No| G[Tune Hyperparameters]
    G --> D
    F -->|Yes| H[Deploy Model]
```
````

**Result:**

```mermaid
graph LR
    A[Input Data] --> B[Preprocess]
    B --> C[Feature Extraction]
    C --> D[Model Training]
    D --> E[Validation]
    E --> F{Accuracy > 95%?}
    F -->|No| G[Tune Hyperparameters]
    G --> D
    F -->|Yes| H[Deploy Model]
```

### Example 3: Node Shapes

````markdown
```mermaid
graph TD
    A[Rectangle] --> B(Rounded)
    B --> C([Stadium])
    C --> D[[Subroutine]]
    D --> E[(Database)]
    E --> F((Circle))
    F --> G>Asymmetric]
    G --> H{Diamond}
    H --> I{{Hexagon}}
```
````

**Result:**

```mermaid
graph TD
    A[Rectangle] --> B(Rounded)
    B --> C([Stadium])
    C --> D[[Subroutine]]
    D --> E[(Database)]
    E --> F((Circle))
    F --> G>Asymmetric]
    G --> H{Diamond}
    H --> I{{Hexagon}}
```

## Notes

- Use `<br/>` for line breaks in node text
- Keep node IDs simple (A, B, C or descriptive names)
- Use subgraphs for grouping related nodes
- Test complex diagrams at https://mermaid.live/

## Gotchas/Warnings

- ⚠️ **Special characters**: Avoid quotes in node text, use `#quot;` if needed
- ⚠️ **Spacing**: Extra spaces in node IDs can cause issues
- ⚠️ **Circular references**: Be careful with loops, they can make diagrams hard to read
- ⚠️ **Complexity**: Very large flowcharts become hard to maintain - consider breaking them up