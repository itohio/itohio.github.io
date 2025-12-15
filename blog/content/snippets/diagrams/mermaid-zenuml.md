---
title: "Mermaid ZenUML Diagrams"
date: 2024-12-13T00:20:00Z
draft: false
description: "Create ZenUML sequence diagrams with Mermaid"
tags: ["mermaid", "zenuml", "sequence", "uml", "diagram", "diagrams"]
category: "diagrams"
---

ZenUML provides an alternative syntax for sequence diagrams, offering more concise notation. Perfect for complex sequence diagrams with less verbosity.

## Use Case

Use ZenUML when you need to:
- Create concise sequence diagrams
- Show complex interactions
- Document API flows
- Visualize method calls
- Use alternative sequence syntax

## Code

````markdown
```mermaid
zenuml
    Client -> API: Request
    API -> Database: Query
    Database --> API: Result
    API --> Client: Response
```
````

**Result:**

```mermaid
zenuml
    Client -> API: Request
    API -> Database: Query
    Database --> API: Result
    API --> Client: Response
```

## Examples

### Example 1: API Flow

````markdown
```mermaid
zenuml
    User -> Frontend: Click Button
    Frontend -> API: POST /data
    API -> Auth: Validate Token
    Auth --> API: Valid
    API -> Database: Save Data
    Database --> API: Success
    API --> Frontend: 201 Created
    Frontend --> User: Show Success
```
````

**Result:**

```mermaid
zenuml
    User -> Frontend: Click Button
    Frontend -> API: POST /data
    API -> Auth: Validate Token
    Auth --> API: Valid
    API -> Database: Save Data
    Database --> API: Success
    API --> Frontend: 201 Created
    Frontend --> User: Show Success
```

### Example 2: Decorators, Groups, and Blocks

````markdown
```mermaid
zenuml
    title Order Service

    @Actor Client #FFEBE6
    @Boundary OrderController #0747A6
    @EC2 <<BFF>> OrderService #E3FCEF

    group BusinessService {
      @Lambda PurchaseService
      @AzureFunction InvoiceService
    }

    @Starter(Client)
    // POST /orders
    OrderController.post(payload) {
      OrderService.create(payload) {
        order = new Order(payload)
        if (order != null) {
          par {
            PurchaseService.createPO(order)
            InvoiceService.createInvoice(order)
          }
        }
      }
    }
```
````

**Result:**

```mermaid
zenuml
    title Order Service

    @Actor Client #FFEBE6
    @Boundary OrderController #0747A6
    @EC2 <<BFF>> OrderService #E3FCEF

    group BusinessService {
      @Lambda PurchaseService
      @AzureFunction InvoiceService
    }

    @Starter(Client)
    // POST /orders
    OrderController.post(payload) {
      OrderService.create(payload) {
        order = new Order(payload)
        if (order != null) {
          par {
            PurchaseService.createPO(order)
            InvoiceService.createInvoice(order)
          }
        }
      }
    }
```

## Notes

- `zenuml` - Start ZenUML diagram
- `->` - Synchronous call, `-->` - Return/response
- `title` - Diagram title
- Decorators like `@Actor`, `@Boundary`, `@EC2`, `@Lambda`, `@AzureFunction` declare participants with types and optional colors
- `group Name { ... }` - Logical group of related participants
- `@Starter(Participant)` - Entry point
- Blocks (`{ ... }`), `if (...) { ... }`, and `par { ... }` describe control flow, conditions, and parallel execution

## Gotchas/Warnings

- ⚠️ **Syntax**: ZenUML syntax differs significantly from standard Mermaid `sequenceDiagram`
- ⚠️ **Support**: ZenUML support in Mermaid may be limited or behind a feature flag in some versions
- ⚠️ **Fallback**: Use standard `sequenceDiagram` if `zenuml` does not render in your environment
- ⚠️ **Versioning**: Check current Mermaid / ZenUML docs for any syntax changes

