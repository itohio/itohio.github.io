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
    todo[To Do]
        t1[Task 1]
        t2[Task 2]
    doing[In Progress]
        t3[Task 3]
    done[Done]
        t4[Task 4]
```
````

**Result:**

```mermaid
kanban
    todo[To Do]
        t1[Task 1]
        t2[Task 2]
    doing[In Progress]
        t3[Task 3]
    done[Done]
        t4[Task 4]
```

## Examples

### Example 1: Development Workflow

````markdown
```mermaid
kanban
    backlog[Backlog]
        t1[Design API]
        t2[Write Tests]
    doing[In Progress]
        t3[Implement Feature]
        t4[Code Review]
    testing[Testing]
        t5[Unit Tests]
    done[Done]
        t6[Deploy to Prod]
        t7[Documentation]
```
````

**Result:**

```mermaid
kanban
    backlog[Backlog]
        t1[Design API]
        t2[Write Tests]
    doing[In Progress]
        t3[Implement Feature]
        t4[Code Review]
    testing[Testing]
        t5[Unit Tests]
    done[Done]
        t6[Deploy to Prod]
        t7[Documentation]
```

### Example 2: Bug Tracking (with IDs and metadata)

````markdown
```mermaid
kanban
    todo[Todo]
        id1[Create Documentation]
        docs[Create Blog about the new diagram]
    doing[In progress]
        id6[Create renderer so that it works in all cases. We also add some extra text here for testing purposes. And some more just for the extra flare.]
    deploy[Ready for deploy]
        id8[Design grammar]@{ assigned: 'knsv' }
    test[Ready for test]
        id4[Create parsing tests]@{ ticket: 2038, assigned: 'K.Sveidqvist', priority: 'High' }
        id66[last item]@{ priority: 'Very Low', assigned: 'knsv' }
    done[Done]
        id5[define getData]
        id2[Title of diagram is more than 100 chars when user duplicates diagram with 100 char]@{ ticket: 2036, priority: 'Very High'}
    cnr[Can't reproduce]
        id3[Weird flickering in Firefox]
```
````

**Result:**

```mermaid
kanban
    todo[Todo]
        id1[Create Documentation]
        docs[Create Blog about the new diagram]
    doing[In progress]
        id6[Create renderer so that it works in all cases. We also add some extra text here for testing purposes. And some more just for the extra flare.]
    deploy[Ready for deploy]
        id8[Design grammar]@{ assigned: 'knsv' }
    test[Ready for test]
        id4[Create parsing tests]@{ ticket: 2038, assigned: 'K.Sveidqvist', priority: 'High' }
        id66[last item]@{ priority: 'Very Low', assigned: 'knsv' }
    done[Done]
        id5[define getData]
        id2[Title of diagram is more than 100 chars when user duplicates diagram with 100 char]@{ ticket: 2036, priority: 'Very High'}
    cnr[Can't reproduce]
        id3[Weird flickering in Firefox]
```

## Notes

- No `title` or `section` keywords: a column is `columnId[Column Title]`
- Tasks are `taskId[Label]`, indented under their column, with optional `@{ ... }` metadata
- Keep task names concise

## Gotchas/Warnings

- ⚠️ **Indentation**: Tasks must be indented under their column
- ⚠️ **Columns**: Define a column before its tasks
- ⚠️ **IDs**: Use unique IDs when you add metadata (`@{ ... }`)
- ⚠️ **Complexity**: Too many tasks can clutter the board

