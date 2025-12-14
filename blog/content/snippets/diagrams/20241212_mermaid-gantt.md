---
title: "Mermaid Gantt Chart"
date: 2024-12-12T18:20:00Z
draft: false
description: "Create Gantt charts for project timelines and scheduling"
type: "snippet"
tags: ["mermaid", "gantt", "timeline", "project", "schedule", "diagrams"]
category: "diagrams"
---



Gantt charts visualize project timelines, showing tasks, durations, and dependencies. Perfect for planning research phases and tracking progress.

## Use Case

Use Gantt charts when you need to:
- Plan research phases and milestones
- Visualize project timelines
- Show task dependencies
- Track progress over time

## Code

````markdown
```mermaid
gantt
    title Research Project Timeline
    dateFormat YYYY-MM-DD
    section Phase 1
    Literature Review    :done, 2024-01-01, 14d
    Hypothesis Formation :active, 2024-01-15, 7d
    section Phase 2
    Experiment Design    :2024-01-22, 10d
    Implementation       :2024-02-01, 21d
```
````

## Explanation

- `dateFormat` - Date format for tasks (YYYY-MM-DD recommended)
- `section` - Group related tasks
- Task format: `Name :status, start, duration`
- Status: `done`, `active`, `crit` (critical), or empty
- Duration: `Xd` (days), `Xw` (weeks), or end date

## Examples

### Example 1: Research Timeline

````markdown
```mermaid
gantt
    title Research Project Timeline
    dateFormat YYYY-MM-DD
    
    section Discovery
    Initial Questions     :done, disc1, 2024-12-01, 3d
    Literature Review     :done, disc2, after disc1, 10d
    Problem Definition    :done, disc3, after disc2, 5d
    
    section Exploration
    Baseline Implementation :active, exp1, 2024-12-19, 14d
    Experiments            :exp2, after exp1, 21d
    Analysis               :exp3, after exp2, 7d
    
    section Implementation
    POC Development        :crit, impl1, 2025-01-20, 28d
    Testing                :impl2, after impl1, 14d
    Documentation          :impl3, after impl2, 7d
```
````

**Result:**

```mermaid
gantt
    title Research Project Timeline
    dateFormat YYYY-MM-DD
    
    section Discovery
    Initial Questions     :done, disc1, 2024-12-01, 3d
    Literature Review     :done, disc2, after disc1, 10d
    Problem Definition    :done, disc3, after disc2, 5d
    
    section Exploration
    Baseline Implementation :active, exp1, 2024-12-19, 14d
    Experiments            :exp2, after exp1, 21d
    Analysis               :exp3, after exp2, 7d
    
    section Implementation
    POC Development        :crit, impl1, 2025-01-20, 28d
    Testing                :impl2, after impl1, 14d
    Documentation          :impl3, after impl2, 7d
```

### Example 2: Milestone-Based

````markdown
```mermaid
gantt
    title Development Milestones
    dateFormat YYYY-MM-DD
    
    section Setup
    Repository Setup      :milestone, 2024-12-01, 0d
    Environment Config    :done, 2024-12-01, 2d
    
    section Development
    Feature A             :done, 2024-12-03, 5d
    Feature B             :active, 2024-12-08, 7d
    Feature C             :2024-12-15, 5d
    
    section Release
    Testing               :2024-12-20, 5d
    Release v1.0          :milestone, 2024-12-25, 0d
```
````

**Result:**

```mermaid
gantt
    title Development Milestones
    dateFormat YYYY-MM-DD
    
    section Setup
    Repository Setup      :milestone, 2024-12-01, 0d
    Environment Config    :done, 2024-12-01, 2d
    
    section Development
    Feature A             :done, 2024-12-03, 5d
    Feature B             :active, 2024-12-08, 7d
    Feature C             :2024-12-15, 5d
    
    section Release
    Testing               :2024-12-20, 5d
    Release v1.0          :milestone, 2024-12-25, 0d
```

## Notes

- Use `milestone` for zero-duration events
- `crit` highlights critical path tasks
- Task IDs (like `disc1`) enable `after` dependencies
- Sections help organize related tasks

## Gotchas/Warnings

- ⚠️ **Date format**: Must match `dateFormat` exactly
- ⚠️ **Dependencies**: Use task IDs with `after` for dependencies
- ⚠️ **Duration**: Can use days (d), weeks (w), or end dates
- ⚠️ **Updates**: Remember to update status as project progresses