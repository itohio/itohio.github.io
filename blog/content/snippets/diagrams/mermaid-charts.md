---
title: "Mermaid Charts (Pie, Bar, Line)"
date: 2025-12-14T20:00:00Z
draft: false
description: "Create pie charts, bar charts, and line charts with Mermaid"
tags: ["mermaid", "charts", "pie", "bar", "line", "data-visualization", "diagrams"]
category: "diagrams"
---

Mermaid supports various chart types including pie charts, bar charts, and line charts. Perfect for visualizing data and statistics in documentation.

## Use Case

Use Mermaid charts when you need to:
- Visualize data distributions (pie charts)
- Compare values across categories (bar charts)
- Show trends over time (line charts)
- Create simple, text-based data visualizations

## Pie Charts

Pie charts show proportional data as slices of a circle.

### Basic Pie Chart

````markdown
```mermaid
pie title Distribution of Languages
    "Python" : 35
    "JavaScript" : 25
    "Go" : 20
    "Rust" : 15
    "Other" : 5
```
````

**Result:**

```mermaid
pie title Distribution of Languages
    "Python" : 35
    "JavaScript" : 25
    "Go" : 20
    "Rust" : 15
    "Other" : 5
```

### Example 1: Research Methods Distribution

````markdown
```mermaid
pie title Research Methods Used
    "Experimental" : 45
    "Theoretical" : 30
    "Simulation" : 15
    "Survey" : 10
```
````

**Result:**

```mermaid
pie title Research Methods Used
    "Experimental" : 45
    "Theoretical" : 30
    "Simulation" : 15
    "Survey" : 10
```

### Example 2: Performance Metrics

````markdown
```mermaid
pie title System Performance Breakdown
    "CPU Usage" : 40
    "Memory" : 30
    "Network I/O" : 20
    "Disk I/O" : 10
```
````

**Result:**

```mermaid
pie title System Performance Breakdown
    "CPU Usage" : 40
    "Memory" : 30
    "Network I/O" : 20
    "Disk I/O" : 10
```

## Bar Charts (GitGraph Style)

Mermaid bar charts use the `gitgraph` syntax for creating bar-like visualizations, or you can use `xychart-beta` for more traditional bar charts.

### XY Chart Bar

````markdown
```mermaid
xychart-beta
    title "Monthly Sales Data"
    x-axis [Jan, Feb, Mar, Apr, May, Jun]
    y-axis "Revenue ($)" 0 --> 10000
    bar [5000, 6000, 7500, 8200, 9500, 11000]
```
````

**Result:**

```mermaid
xychart-beta
    title "Monthly Sales Data"
    x-axis [Jan, Feb, Mar, Apr, May, Jun]
    y-axis "Revenue ($)" 0 --> 10000
    bar [5000, 6000, 7500, 8200, 9500, 11000]
```

### Example 1: Performance Comparison

````markdown
```mermaid
xychart-beta
    title "Algorithm Performance Comparison"
    x-axis [Algorithm A, Algorithm B, Algorithm C, Algorithm D]
    y-axis "Execution Time (ms)" 0 --> 500
    bar [450, 320, 280, 195]
```
````

**Result:**

```mermaid
xychart-beta
    title "Algorithm Performance Comparison"
    x-axis [Algorithm A, Algorithm B, Algorithm C, Algorithm D]
    y-axis "Execution Time (ms)" 0 --> 500
    bar [450, 320, 280, 195]
```

## Line Charts

Line charts show trends over time using the `xychart-beta` syntax.

### Basic Line Chart

````markdown
```mermaid
xychart-beta
    title "Temperature Over Time"
    x-axis [Jan, Feb, Mar, Apr, May, Jun]
    y-axis "Temperature (°C)" 0 --> 30
    line [5, 8, 12, 18, 22, 25]
```
````

**Result:**

```mermaid
xychart-beta
    title "Temperature Over Time"
    x-axis [Jan, Feb, Mar, Apr, May, Jun]
    y-axis "Temperature (°C)" 0 --> 30
    line [5, 8, 12, 18, 22, 25]
```

### Example 1: Multiple Series

````markdown
```mermaid
xychart-beta
    title "Accuracy Over Epochs"
    x-axis [Epoch 1, Epoch 5, Epoch 10, Epoch 15, Epoch 20]
    y-axis "Accuracy (%)" 0 --> 100
    line [45, 65, 78, 85, 92]
    line [40, 60, 72, 80, 88]
```
````

**Result:**

```mermaid
xychart-beta
    title "Accuracy Over Epochs"
    x-axis [Epoch 1, Epoch 5, Epoch 10, Epoch 15, Epoch 20]
    y-axis "Accuracy (%)" 0 --> 100
    line [45, 65, 78, 85, 92]
    line [40, 60, 72, 80, 88]
```

### Example 2: Research Progress

````markdown
```mermaid
xychart-beta
    title "Research Progress Over Time"
    x-axis [Week 1, Week 2, Week 3, Week 4, Week 5, Week 6]
    y-axis "Completion (%)" 0 --> 100
    line [10, 25, 40, 55, 75, 90]
```
````

**Result:**

```mermaid
xychart-beta
    title "Research Progress Over Time"
    x-axis [Week 1, Week 2, Week 3, Week 4, Week 5, Week 6]
    y-axis "Completion (%)" 0 --> 100
    line [10, 25, 40, 55, 75, 90]
```

## Combined Charts

You can combine bar and line charts in a single visualization.

### Example: Bar and Line Combination

````markdown
```mermaid
xychart-beta
    title "Revenue and Growth Rate"
    x-axis [Q1, Q2, Q3, Q4]
    y-axis "Revenue ($)" 0 --> 50000
    y-axis "Growth (%)" 0 --> 20
    bar [30000, 35000, 42000, 48000]
    line [5, 8, 12, 15]
```
````

**Result:**

```mermaid
xychart-beta
    title "Revenue and Growth Rate"
    x-axis [Q1, Q2, Q3, Q4]
    y-axis "Revenue ($)" 0 --> 50000
    y-axis "Growth (%)" 0 --> 20
    bar [30000, 35000, 42000, 48000]
    line [5, 8, 12, 15]
```

## Syntax Reference

### Pie Chart Syntax

```
pie title "Chart Title"
    "Label 1" : value1
    "Label 2" : value2
    "Label 3" : value3
```

### XY Chart Syntax

```
xychart-beta
    title "Chart Title"
    x-axis [label1, label2, label3, ...]
    y-axis "Y-axis Label" min --> max
    bar [value1, value2, value3, ...]
    line [value1, value2, value3, ...]
```

## Notes

- Pie charts automatically calculate percentages from values
- XY charts support multiple series (multiple `bar` or `line` declarations)
- Use descriptive titles and axis labels
- Ensure data arrays match x-axis labels in length
- XY charts are in beta - syntax may change

## Gotchas/Warnings

- ⚠️ **XY Chart Beta**: `xychart-beta` is experimental - check Mermaid docs for updates
- ⚠️ **Data Length**: X-axis labels and data arrays must have matching lengths
- ⚠️ **Pie Values**: Values are relative - Mermaid calculates percentages automatically
- ⚠️ **Multiple Series**: Each `bar` or `line` declaration creates a new series
- ⚠️ **Axis Range**: Set appropriate min/max for y-axis to show data clearly

