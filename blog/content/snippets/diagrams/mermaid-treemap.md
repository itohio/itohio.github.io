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
treemap
    title Project Structure
    Root
        Frontend : 40
        Backend : 35
        Database : 25
```
````

**Result:**

```mermaid
treemap
    title Project Structure
    Root
        Frontend : 40
        Backend : 35
        Database : 25
```

## Examples

### Example 1: Project Structure

````markdown
```mermaid
treemap
    title Codebase Structure
    Root
        Frontend : 45
            Components : 20
            Pages : 15
            Utils : 10
        Backend : 35
            API : 15
            Services : 12
            Models : 8
        Tests : 20
            Unit : 12
            Integration : 8
```
````

**Result:**

```mermaid
treemap
    title Codebase Structure
    Root
        Frontend : 45
            Components : 20
            Pages : 15
            Utils : 10
        Backend : 35
            API : 15
            Services : 12
            Models : 8
        Tests : 20
            Unit : 12
            Integration : 8
```

### Example 2: Budget Allocation

````markdown
```mermaid
treemap
    title Budget Breakdown
    Total Budget
        Development : 50
            Backend : 25
            Frontend : 20
            DevOps : 5
        Marketing : 30
            Advertising : 15
            Content : 10
            Events : 5
        Operations : 20
            Infrastructure : 12
            Support : 8
```
````

**Result:**

```mermaid
treemap
    title Budget Breakdown
    Total Budget
        Development : 50
            Backend : 25
            Frontend : 20
            DevOps : 5
        Marketing : 30
            Advertising : 15
            Content : 10
            Events : 5
        Operations : 20
            Infrastructure : 12
            Support : 8
```

## Notes

- `title` - Optional treemap title
- Indentation creates hierarchy
- Values determine rectangle size
- Can nest multiple levels

## Gotchas/Warnings

- ⚠️ **Indentation**: Must be consistent for hierarchy
- ⚠️ **Values**: Should be positive numbers
- ⚠️ **Nesting**: Can nest multiple levels deep
- ⚠️ **Proportions**: Rectangle sizes proportional to values

