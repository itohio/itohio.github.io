---
title: "Mermaid State Diagrams"
date: 2024-12-12T22:10:00Z
draft: false
description: "Create state diagrams and state machines with Mermaid"
tags: ["mermaid", "state", "state-machine", "uml", "diagram", "diagrams"]
category: "diagrams"
---

State diagrams visualize the states of a system and transitions between them. Perfect for modeling state machines, workflows, and system behavior.

## Use Case

Use state diagrams when you need to:
- Model state machines
- Document system states and transitions
- Show workflow states
- Visualize state transitions
- Design finite state automata

## Code

````markdown
```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing: start
    Processing --> Success: complete
    Processing --> Error: fail
    Error --> [*]
    Success --> [*]
```
````

**Result:**

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing: start
    Processing --> Success: complete
    Processing --> Error: fail
    Error --> [*]
    Success --> [*]
```

## Explanation

- `stateDiagram-v2` - Modern state diagram syntax
- `[*]` - Start/End state
- `-->` - State transition
- `: label` - Transition label
- States can be simple or composite

## Examples

### Example 1: Order Processing State Machine

````markdown
```mermaid
stateDiagram-v2
    [*] --> Pending
    Pending --> Processing: payment_received
    Processing --> Shipped: order_fulfilled
    Processing --> Cancelled: payment_failed
    Shipped --> Delivered: delivery_complete
    Delivered --> [*]
    Cancelled --> [*]
```
````

**Result:**

```mermaid
stateDiagram-v2
    [*] --> Pending
    Pending --> Processing: payment_received
    Processing --> Shipped: order_fulfilled
    Processing --> Cancelled: payment_failed
    Shipped --> Delivered: delivery_complete
    Delivered --> [*]
    Cancelled --> [*]
```

### Example 2: Composite States

````markdown
```mermaid
stateDiagram-v2
    [*] --> NotShooting
    
    state NotShooting {
        [*] --> Idle
        Idle --> Configuring: evConfig
        Configuring --> Idle: evConfig
    }
    
    NotShooting --> Shooting: evShutterRelease
    Shooting --> NotShooting: evShutterRelease
    
    state Shooting {
        [*] --> Idle
        Idle --> Processing: evImage
        Processing --> Idle: evDone
    }
```
````

**Result:**

```mermaid
stateDiagram-v2
    [*] --> NotShooting
    
    state NotShooting {
        [*] --> Idle
        Idle --> Configuring: evConfig
        Configuring --> Idle: evConfig
    }
    
    NotShooting --> Shooting: evShutterRelease
    Shooting --> NotShooting: evShutterRelease
    
    state Shooting {
        [*] --> Idle
        Idle --> Processing: evImage
        Processing --> Idle: evDone
    }
```

### Example 3: Concurrent States

````markdown
```mermaid
stateDiagram-v2
    [*] --> Active
    
    state Active {
        [*] --> NumLockOff
        NumLockOff --> NumLockOn : EvNumLockPressed
        NumLockOn --> NumLockOff : EvNumLockPressed
        --
        [*] --> CapsLockOff
        CapsLockOff --> CapsLockOn : EvCapsLockPressed
        CapsLockOn --> CapsLockOff : EvCapsLockPressed
        --
        [*] --> ScrollLockOff
        ScrollLockOff --> ScrollLockOn : EvScrollLockPressed
        ScrollLockOn --> ScrollLockOff : EvScrollLockPressed
    }
```
````

**Result:**

```mermaid
stateDiagram-v2
    [*] --> Active
    
    state Active {
        [*] --> NumLockOff
        NumLockOff --> NumLockOn : EvNumLockPressed
        NumLockOn --> NumLockOff : EvNumLockPressed
        --
        [*] --> CapsLockOff
        CapsLockOff --> CapsLockOn : EvCapsLockPressed
        CapsLockOn --> CapsLockOff : EvCapsLockPressed
        --
        [*] --> ScrollLockOff
        ScrollLockOff --> ScrollLockOn : EvScrollLockPressed
        ScrollLockOn --> ScrollLockOff : EvScrollLockPressed
    }
```

## Notes

- Use `stateDiagram-v2` for modern syntax (recommended)
- `[*]` represents start/end states
- Composite states use `state StateName { ... }`
- Concurrent regions use `--` separator
- Transition labels are optional

## Gotchas/Warnings

- ⚠️ **Syntax**: Use `stateDiagram-v2` not `stateDiagram` (deprecated)
- ⚠️ **Start/End**: `[*]` must be used for initial/final states
- ⚠️ **Labels**: Transition labels come after the colon
- ⚠️ **Complexity**: Deeply nested states can be hard to read

