---
title: "Mermaid Treemap Diagrams"
date: 2024-12-12T23:40:00Z
draft: false
description: "Create treemap diagrams for hierarchical data visualization with Mermaid"
tags: ["mermaid", "treemap", "hierarchy", "data-visualization", "diagram", "diagrams"]
category: "diagrams"
---

Treemaps visualize hierarchical data as nested rectangles, with size proportional to value. Perfect for showing proportions, file sizes, and hierarchical structures.

## Use Case

Use treemaps when you need to:
- Show hierarchical proportions
- Visualize file/directory sizes
- Display category breakdowns
- Show nested data structures
- Compare relative sizes

## Code

````markdown
```mermaid
treemap-beta
"Project Structure"
    "Frontend": 40
    "Backend": 35
    "Database": 25
```
````

**Result:**

```mermaid
treemap-beta
"Project Structure"
    "Frontend": 40
    "Backend": 35
    "Database": 25
```

## Examples

### Example 1: Project Structure

````markdown
```mermaid
treemap-beta
"Codebase Structure"
    "Frontend"
        "Components": 20
        "Pages": 15
        "Utils": 10
    "Backend"
        "API": 15
        "Services": 12
        "Models": 8
    "Tests"
        "Unit": 12
        "Integration": 8
```
````

**Result:**

```mermaid
treemap-beta
"Codebase Structure"
    "Frontend"
        "Components": 20
        "Pages": 15
        "Utils": 10
    "Backend"
        "API": 15
        "Services": 12
        "Models": 8
    "Tests"
        "Unit": 12
        "Integration": 8
```

### Example 2: Budget Allocation

````markdown
```mermaid
treemap-beta
"Total Budget"
    "Development"
        "Backend": 25
        "Frontend": 20
        "DevOps": 5
    "Marketing"
        "Advertising": 15
        "Content": 10
        "Events": 5
    "Operations"
        "Infrastructure": 12
        "Support": 8
```
````

**Result:**

```mermaid
treemap-beta
"Total Budget"
    "Development"
        "Backend": 25
        "Frontend": 20
        "DevOps": 5
    "Marketing"
        "Advertising": 15
        "Content": 10
        "Events": 5
    "Operations"
        "Infrastructure": 12
        "Support": 8
```

## Notes

- Keyword is `treemap-beta` (Mermaid 11.8.0+)
- Every label is double-quoted; leaves are `"Label": value`
- Indentation creates hierarchy; parents carry no value (their size is the sum of children)
- Can nest multiple levels

## Gotchas/Warnings

- ⚠️ **Indentation**: Must be consistent for hierarchy
- ⚠️ **Values**: Only on leaves, positive numbers
- ⚠️ **Nesting**: Can nest multiple levels deep
- ⚠️ **Proportions**: Rectangle sizes proportional to values


