---
title: "Mermaid Quadrant Charts"
date: 2024-12-12T23:20:00Z
draft: false
description: "Create quadrant charts for prioritization with Mermaid"
tags: ["mermaid", "quadrant", "prioritization", "matrix", "diagram", "diagrams"]
category: "diagrams"
---

Quadrant charts divide items into four quadrants based on two criteria. Perfect for prioritization matrices, feature planning, and decision-making frameworks.

## Use Case

Use quadrant charts when you need to:
- Prioritize features or tasks
- Create 2x2 decision matrices
- Classify items by two criteria
- Visualize trade-offs
- Make strategic decisions

## Code

````markdown
```mermaid
quadrantChart
    title Prioritization Matrix
    x-axis Low --> High
    y-axis Low --> High
    quadrant-1 High Value
    quadrant-2 Quick Wins
    quadrant-3 Low Priority
    quadrant-4 High Effort
    Feature A: [0.3, 0.6]
    Feature B: [0.7, 0.8]
    Feature C: [0.2, 0.3]
```
````

**Result:**

```mermaid
quadrantChart
    title Prioritization Matrix
    x-axis Low --> High
    y-axis Low --> High
    quadrant-1 High Value
    quadrant-2 Quick Wins
    quadrant-3 Low Priority
    quadrant-4 High Effort
    Feature A: [0.3, 0.6]
    Feature B: [0.7, 0.8]
    Feature C: [0.2, 0.3]
```

## Examples

### Example 1: Feature Prioritization

````markdown
```mermaid
quadrantChart
    title Feature Prioritization
    x-axis Low Effort --> High Effort
    y-axis Low Value --> High Value
    quadrant-1 High Value, High Effort
    quadrant-2 High Value, Low Effort
    quadrant-3 Low Value, Low Effort
    quadrant-4 Low Value, High Effort
    Login: [0.2, 0.8]
    Dashboard: [0.6, 0.9]
    Reports: [0.8, 0.7]
    Themes: [0.3, 0.4]
```
````

**Result:**

```mermaid
quadrantChart
    title Feature Prioritization
    x-axis Low Effort --> High Effort
    y-axis Low Value --> High Value
    quadrant-1 High Value, High Effort
    quadrant-2 High Value, Low Effort
    quadrant-3 Low Value, Low Effort
    quadrant-4 Low Value, High Effort
    Login: [0.2, 0.8]
    Dashboard: [0.6, 0.9]
    Reports: [0.8, 0.7]
    Themes: [0.3, 0.4]
```

### Example 2: Risk vs Impact

````markdown
```mermaid
quadrantChart
    title Risk vs Impact Analysis
    x-axis Low Risk --> High Risk
    y-axis Low Impact --> High Impact
    quadrant-1 High Impact, High Risk
    quadrant-2 High Impact, Low Risk
    quadrant-3 Low Impact, Low Risk
    quadrant-4 Low Impact, High Risk
    Migration: [0.8, 0.9]
    New Feature: [0.3, 0.8]
    Bug Fix: [0.2, 0.6]
    Refactor: [0.5, 0.4]
```
````

**Result:**

```mermaid
quadrantChart
    title Risk vs Impact Analysis
    x-axis Low Risk --> High Risk
    y-axis Low Impact --> High Impact
    quadrant-1 High Impact, High Risk
    quadrant-2 High Impact, Low Risk
    quadrant-3 Low Impact, Low Risk
    quadrant-4 Low Impact, High Risk
    Migration: [0.8, 0.9]
    New Feature: [0.3, 0.8]
    Bug Fix: [0.2, 0.6]
    Refactor: [0.5, 0.4]
```

## Notes

- Coordinates: `[x, y]` where values are 0.0 to 1.0
- Quadrants numbered 1-4 (top-right, top-left, bottom-left, bottom-right)
- Axis labels define the criteria
- Quadrant labels describe each section

## Gotchas/Warnings

- ⚠️ **Coordinates**: Values must be 0.0 to 1.0
- ⚠️ **Quadrants**: Numbered 1-4 starting top-right, clockwise
- ⚠️ **Axes**: Must define both x-axis and y-axis
- ⚠️ **Labels**: Keep quadrant labels concise

