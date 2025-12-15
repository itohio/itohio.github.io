---
title: "Chart.js Bar Chart"
date: 2024-12-12T18:30:00Z
draft: false
description: "Create bar charts for data visualization"
tags: ["chartjs", "bar", "chart", "data", "visualization", "diagrams"]
category: "diagrams"
---



Bar charts are perfect for comparing values across categories. Chart.js provides interactive, responsive charts with hover effects.

## Use Case

Use bar charts when you need to:
- Compare values across categories
- Show performance metrics
- Display experimental results
- Visualize benchmark data

## Code

````markdown
```chart
{
  "type": "bar",
  "data": {
    "labels": ["Method A", "Method B", "Method C"],
    "datasets": [{
      "label": "Performance (ms)",
      "data": [120, 85, 95],
      "backgroundColor": [
        "rgba(255, 99, 132, 0.5)",
        "rgba(54, 162, 235, 0.5)",
        "rgba(255, 206, 86, 0.5)"
      ]
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true
      }
    }
  }
}
```
````

**Result:**

```chart
{
  "type": "bar",
  "data": {
    "labels": ["Method A", "Method B", "Method C"],
    "datasets": [{
      "label": "Performance (ms)",
      "data": [120, 85, 95],
      "backgroundColor": [
        "rgba(255, 99, 132, 0.5)",
        "rgba(54, 162, 235, 0.5)",
        "rgba(255, 206, 86, 0.5)"
      ]
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true
      }
    }
  }
}
```

## Explanation

- `type`: Chart type (bar, line, pie, etc.)
- `data.labels`: X-axis labels
- `data.datasets`: Data series with styling
- `options`: Chart configuration (scales, legend, etc.)

## Examples

### Example 1: Benchmark Comparison

````markdown
```chart
{
  "type": "bar",
  "data": {
    "labels": ["Baseline", "Optimized v1", "Optimized v2", "Final"],
    "datasets": [{
      "label": "Execution Time (ms)",
      "data": [450, 320, 280, 195],
      "backgroundColor": "rgba(75, 192, 192, 0.6)",
      "borderColor": "rgba(75, 192, 192, 1)",
      "borderWidth": 1
    }]
  },
  "options": {
    "responsive": true,
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "Time (milliseconds)"
        }
      }
    },
    "plugins": {
      "title": {
        "display": true,
        "text": "Performance Optimization Progress"
      }
    }
  }
}
```
````

**Result:**

```chart
{
  "type": "bar",
  "data": {
    "labels": ["Baseline", "Optimized v1", "Optimized v2", "Final"],
    "datasets": [{
      "label": "Execution Time (ms)",
      "data": [450, 320, 280, 195],
      "backgroundColor": "rgba(75, 192, 192, 0.6)",
      "borderColor": "rgba(75, 192, 192, 1)",
      "borderWidth": 1
    }]
  },
  "options": {
    "responsive": true,
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "Time (milliseconds)"
        }
      }
    },
    "plugins": {
      "title": {
        "display": true,
        "text": "Performance Optimization Progress"
      }
    }
  }
}
```

### Example 2: Multiple Datasets

````markdown
```chart
{
  "type": "bar",
  "data": {
    "labels": ["Test 1", "Test 2", "Test 3", "Test 4"],
    "datasets": [
      {
        "label": "Accuracy (%)",
        "data": [92, 95, 94, 97],
        "backgroundColor": "rgba(54, 162, 235, 0.6)"
      },
      {
        "label": "Precision (%)",
        "data": [89, 93, 91, 96],
        "backgroundColor": "rgba(255, 99, 132, 0.6)"
      }
    ]
  },
  "options": {
    "responsive": true,
    "scales": {
      "y": {
        "beginAtZero": true,
        "max": 100
      }
    }
  }
}
```
````

**Result:**

```chart
{
  "type": "bar",
  "data": {
    "labels": ["Test 1", "Test 2", "Test 3", "Test 4"],
    "datasets": [
      {
        "label": "Accuracy (%)",
        "data": [92, 95, 94, 97],
        "backgroundColor": "rgba(54, 162, 235, 0.6)"
      },
      {
        "label": "Precision (%)",
        "data": [89, 93, 91, 96],
        "backgroundColor": "rgba(255, 99, 132, 0.6)"
      }
    ]
  },
  "options": {
    "responsive": true,
    "scales": {
      "y": {
        "beginAtZero": true,
        "max": 100
      }
    }
  }
}
```

### Example 3: Horizontal Bar Chart

````markdown
```chart
{
  "type": "bar",
  "data": {
    "labels": ["Python", "Go", "JavaScript", "Rust"],
    "datasets": [{
      "label": "Lines of Code",
      "data": [1200, 800, 1500, 650],
      "backgroundColor": [
        "rgba(255, 159, 64, 0.6)",
        "rgba(75, 192, 192, 0.6)",
        "rgba(255, 205, 86, 0.6)",
        "rgba(201, 203, 207, 0.6)"
      ]
    }]
  },
  "options": {
    "indexAxis": "y",
    "responsive": true
  }
}
```
````

**Result:**

```chart
{
  "type": "bar",
  "data": {
    "labels": ["Python", "Go", "JavaScript", "Rust"],
    "datasets": [{
      "label": "Lines of Code",
      "data": [1200, 800, 1500, 650],
      "backgroundColor": [
        "rgba(255, 159, 64, 0.6)",
        "rgba(75, 192, 192, 0.6)",
        "rgba(255, 205, 86, 0.6)",
        "rgba(201, 203, 207, 0.6)"
      ]
    }]
  },
  "options": {
    "indexAxis": "y",
    "responsive": true
  }
}
```

## Notes

- Charts are interactive - hover to see values
- Use `indexAxis: "y"` for horizontal bars
- RGBA colors: `rgba(red, green, blue, alpha)`
- Multiple datasets create grouped bars

## Gotchas/Warnings

- ⚠️ **JSON format**: Must be valid JSON (double quotes, no trailing commas)
- ⚠️ **Data length**: Labels and data arrays must match in length
- ⚠️ **Colors**: Provide enough colors for all data points
- ⚠️ **Scale**: Set appropriate min/max for your data range