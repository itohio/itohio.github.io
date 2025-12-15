---
title: "Mermaid Sequence Diagram"
date: 2024-12-12T18:10:00Z
draft: false
description: "Create sequence diagrams for interactions and API flows"
tags: ["mermaid", "sequence", "diagram", "api", "interaction", "diagrams"]
category: "diagrams"
---



Sequence diagrams show interactions between different components over time. Perfect for documenting API calls, system interactions, and message flows.

## Use Case

Use sequence diagrams when you need to:
- Document API interactions
- Show message passing between components
- Visualize request/response flows
- Map out system communication patterns

## Code

````markdown
```mermaid
sequenceDiagram
    participant A as Alice
    participant B as Bob
    
    A->>B: Hello Bob!
    B-->>A: Hi Alice!
```
````

**Result:**

```mermaid
sequenceDiagram
    participant A as Alice
    participant B as Bob
    
    A->>B: Hello Bob!
    B-->>A: Hi Alice!
```

## Explanation

- `participant` - Define participants (optional, auto-detected)
- `->>` - Solid arrow (synchronous call)
- `-->>` - Dashed arrow (asynchronous return)
- `->>+` - Activate lifeline
- `-->>-` - Deactivate lifeline

## Examples

### Example 1: API Request Flow

````markdown
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Database
    
    Client->>+API: POST /api/data
    API->>+Database: INSERT query
    Database-->>-API: Success
    API-->>-Client: 201 Created
```
````

**Result:**

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Database
    
    Client->>+API: POST /api/data
    API->>+Database: INSERT query
    Database-->>-API: Success
    API-->>-Client: 201 Created
```

### Example 2: With Loops and Alt

````markdown
```mermaid
sequenceDiagram
    participant User
    participant System
    participant Cache
    participant DB
    
    User->>System: Request Data
    System->>Cache: Check Cache
    
    alt Cache Hit
        Cache-->>System: Return Data
    else Cache Miss
        System->>DB: Query Database
        DB-->>System: Return Data
        System->>Cache: Update Cache
    end
    
    System-->>User: Return Data
```
````

**Result:**

```mermaid
sequenceDiagram
    participant User
    participant System
    participant Cache
    participant DB
    
    User->>System: Request Data
    System->>Cache: Check Cache
    
    alt Cache Hit
        Cache-->>System: Return Data
    else Cache Miss
        System->>DB: Query Database
        DB-->>System: Return Data
        System->>Cache: Update Cache
    end
    
    System-->>User: Return Data
```

### Example 3: With Notes

````markdown
```mermaid
sequenceDiagram
    participant A as Algorithm
    participant D as Data
    
    Note over A,D: Initialization Phase
    A->>D: Load Data
    D-->>A: Data Ready
    
    Note right of A: Processing
    loop For each item
        A->>D: Process Item
        D-->>A: Result
    end
    
    Note over A: Complete
```
````

**Result:**

```mermaid
sequenceDiagram
    participant A as Algorithm
    participant D as Data
    
    Note over A,D: Initialization Phase
    A->>D: Load Data
    D-->>A: Data Ready
    
    Note right of A: Processing
    loop For each item
        A->>D: Process Item
        D-->>A: Result
    end
    
    Note over A: Complete
```

## Notes

- Use `participant X as Name` for readable aliases
- `Note over A,B: Text` spans multiple participants
- `Note left/right of A: Text` for single participant notes
- Supports `loop`, `alt`, `opt`, `par` blocks

## Gotchas/Warnings

- ⚠️ **Arrow types**: `->>` (solid) vs `-->>` (dashed) - use consistently
- ⚠️ **Activation**: `+` and `-` must be balanced
- ⚠️ **Participant order**: Defined order determines left-to-right placement
- ⚠️ **Long text**: Break long messages into multiple lines with `<br/>`