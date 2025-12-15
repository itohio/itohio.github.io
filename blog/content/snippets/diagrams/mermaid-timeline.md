---
title: "Mermaid Timeline Diagrams"
date: 2024-12-12T23:00:00Z
draft: false
description: "Create timeline diagrams for chronological events with Mermaid"
tags: ["mermaid", "timeline", "chronology", "history", "diagram", "diagrams"]
category: "diagrams"
---

Timeline diagrams show events in chronological order. Perfect for documenting history, project milestones, and event sequences.

## Use Case

Use timelines when you need to:
- Show chronological events
- Document project milestones
- Visualize history
- Track event sequences
- Display temporal relationships

## Code

````markdown
```mermaid
timeline
    title Project Timeline
    2024-01 : Planning
    2024-02 : Development
    2024-03 : Testing
    2024-04 : Launch
```
````

**Result:**

```mermaid
timeline
    title Project Timeline
    2024-01 : Planning
    2024-02 : Development
    2024-03 : Testing
    2024-04 : Launch
```

## Examples

### Example 1: Project Milestones

````markdown
```mermaid
timeline
    title Research Project
    2024-01 : Literature Review
    2024-02 : Hypothesis Formation
    2024-03 : Experiment Design
    2024-04 : Data Collection
    2024-05 : Analysis
    2024-06 : Publication
```
````

**Result:**

```mermaid
timeline
    title Research Project
    2024-01 : Literature Review
    2024-02 : Hypothesis Formation
    2024-03 : Experiment Design
    2024-04 : Data Collection
    2024-05 : Analysis
    2024-06 : Publication
```

### Example 2: Technology Evolution

````markdown
```mermaid
timeline
    title Web Development Evolution
    1990s : HTML/CSS
    2000s : JavaScript Frameworks
    2010s : Single Page Apps
    2020s : Modern Frameworks
        : React
        : Vue
        : Angular
```
````

**Result:**

```mermaid
timeline
    title Web Development Evolution
    1990s : HTML/CSS
    2000s : JavaScript Frameworks
    2010s : Single Page Apps
    2020s : Modern Frameworks
        : React
        : Vue
        : Angular
```

## Notes

- `title` - Optional timeline title
- Format: `Date/Time : Event`
- Can nest events under time periods
- Use indentation for sub-events

## Gotchas/Warnings

- ⚠️ **Format**: Must use `Date : Event` format
- ⚠️ **Chronology**: Events should be in chronological order
- ⚠️ **Nesting**: Use indentation for sub-events
- ⚠️ **Dates**: Can use various date formats

