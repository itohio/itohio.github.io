---
title: "Mermaid Kanban Boards"
date: 2024-12-12T22:50:00Z
draft: false
description: "Create Kanban boards for project management with Mermaid"
tags: ["mermaid", "kanban", "project-management", "agile", "diagram", "diagrams"]
category: "diagrams"
---

Kanban boards visualize work items across different stages. Perfect for project management, task tracking, and workflow visualization.

## Use Case

Use Kanban boards when you need to:
- Track project tasks
- Visualize workflow stages
- Manage agile sprints
- Show work in progress
- Document process stages

## Code (Basic)

````markdown
```mermaid
kanban
    title Kanban Board
    section To Do
        Task 1
        Task 2
    section In Progress
        Task 3
    section Done
        Task 4
```
````

**Result:**

```mermaid
kanban
    title Kanban Board
    section To Do
        Task 1
        Task 2
    section In Progress
        Task 3
    section Done
        Task 4
```

## Examples

### Example 1: Development Workflow

````markdown
```mermaid
kanban
    title Development Pipeline
    section Backlog
        Design API
        Write Tests
    section In Progress
        Implement Feature
        Code Review
    section Testing
        Unit Tests
    section Done
        Deploy to Prod
        Documentation
```
````

**Result:**

```mermaid
kanban
    title Development Pipeline
    section Backlog
        Design API
        Write Tests
    section In Progress
        Implement Feature
        Code Review
    section Testing
        Unit Tests
    section Done
        Deploy to Prod
        Documentation
```

### Example 2: Bug Tracking (with IDs and metadata)

````markdown
```mermaid
kanban
    title Bug Tracking
    section Todo
        id1[Create Documentation]
        docs[Create Blog about the new diagram]
    section In progress
        id6[Create renderer so that it works in all cases. We also add some extra text here for testing purposes. And some more just for the extra flare.]
    section Ready for deploy
        id8[Design grammar]@{ assigned: 'knsv' }
    section Ready for test
        id4[Create parsing tests]@{ ticket: 2038, assigned: 'K.Sveidqvist', priority: 'High' }
        id66[last item]@{ priority: 'Very Low', assigned: 'knsv' }
    section Done
        id5[define getData]
        id2[Title of diagram is more than 100 chars when user duplicates diagram with 100 char]@{ ticket: 2036, priority: 'Very High'}
    section Can't reproduce
        id3[Weird flickering in Firefox]
```
````

**Result:**

```mermaid
kanban
    title Bug Tracking
    section Todo
        id1[Create Documentation]
        docs[Create Blog about the new diagram]
    section In progress
        id6[Create renderer so that it works in all cases. We also add some extra text here for testing purposes. And some more just for the extra flare.]
    section Ready for deploy
        id8[Design grammar]@{ assigned: 'knsv' }
    section Ready for test
        id4[Create parsing tests]@{ ticket: 2038, assigned: 'K.Sveidqvist', priority: 'High' }
        id66[last item]@{ priority: 'Very Low', assigned: 'knsv' }
    section Done
        id5[define getData]
        id2[Title of diagram is more than 100 chars when user duplicates diagram with 100 char]@{ ticket: 2036, priority: 'Very High'}
    section Can't reproduce
        id3[Weird flickering in Firefox]
```

## Notes

- `title` - Optional board title
- `section` - Defines a column/stage
- Tasks can be plain text or `id[Label]` with optional `@{ ... }` metadata
- Keep task names concise

## Gotchas/Warnings

- ⚠️ **Indentation**: Tasks must be indented under their `section`
- ⚠️ **Sections**: Must define sections before tasks
- ⚠️ **IDs**: Use unique IDs when you add metadata (`@{ ... }`)
- ⚠️ **Complexity**: Too many tasks can clutter the board

