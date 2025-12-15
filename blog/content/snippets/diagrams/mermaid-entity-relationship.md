---
title: "Mermaid Entity Relationship Diagrams"
date: 2024-12-12T22:00:00Z
draft: false
description: "Create Entity Relationship Diagrams (ERD) with Mermaid"
tags: ["mermaid", "erd", "entity-relationship", "database", "diagram", "diagrams"]
category: "diagrams"
---

Entity Relationship Diagrams (ERD) visualize database schemas, showing entities, their attributes, and relationships. Perfect for database design and documentation.

## Use Case

Use ER diagrams when you need to:
- Design database schemas
- Document database structure
- Show relationships between entities
- Visualize data models
- Communicate database architecture

## Code

````markdown
```mermaid
erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE-ITEM : contains
    PRODUCT ||--o{ LINE-ITEM : "ordered in"
```
````

**Result:**

```mermaid
erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE-ITEM : contains
    PRODUCT ||--o{ LINE-ITEM : "ordered in"
```

## Explanation

- `erDiagram` - Start ER diagram
- Entity names in UPPERCASE
- Relationship syntax: `Entity1 ||--o{ Entity2 : label`
- Cardinality symbols:
  - `||--||` : One to one
  - `}o--||` : Many to one
  - `}o--o{` : Many to many
  - `||--o{` : One to many
  - `}o--|{` : One or more to many

## Examples

### Example 1: E-Commerce Database

````markdown
```mermaid
erDiagram
    CUSTOMER {
        int customer_id PK
        string name
        string email
        string address
    }
    
    ORDER {
        int order_id PK
        int customer_id FK
        date order_date
        decimal total
    }
    
    PRODUCT {
        int product_id PK
        string name
        decimal price
        int stock
    }
    
    ORDER_ITEM {
        int order_id FK
        int product_id FK
        int quantity
        decimal price
    }
    
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ ORDER_ITEM : contains
    PRODUCT ||--o{ ORDER_ITEM : "ordered in"
```
````

**Result:**

```mermaid
erDiagram
    CUSTOMER {
        int customer_id PK
        string name
        string email
        string address
    }
    
    ORDER {
        int order_id PK
        int customer_id FK
        date order_date
        decimal total
    }
    
    PRODUCT {
        int product_id PK
        string name
        decimal price
        int stock
    }
    
    ORDER_ITEM {
        int order_id FK
        int product_id FK
        int quantity
        decimal price
    }
    
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ ORDER_ITEM : contains
    PRODUCT ||--o{ ORDER_ITEM : "ordered in"
```

### Example 2: University Database

````markdown
```mermaid
erDiagram
    STUDENT {
        int student_id PK
        string name
        string email
    }
    
    COURSE {
        int course_id PK
        string title
        int credits
    }
    
    ENROLLMENT {
        int student_id FK
        int course_id FK
        string grade
    }
    
    PROFESSOR {
        int professor_id PK
        string name
        string department
    }
    
    STUDENT }o--o{ COURSE : enrolls
    COURSE }o--|| PROFESSOR : "taught by"
```
````

**Result:**

```mermaid
erDiagram
    STUDENT {
        int student_id PK
        string name
        string email
    }
    
    COURSE {
        int course_id PK
        string title
        int credits
    }
    
    ENROLLMENT {
        int student_id FK
        int course_id FK
        string grade
    }
    
    PROFESSOR {
        int professor_id PK
        string name
        string department
    }
    
    STUDENT }o--o{ COURSE : enrolls
    COURSE }o--|| PROFESSOR : "taught by"
```

## Relationship Cardinality

| Symbol | Meaning | Description |
|--------|---------|-------------|
| `||--||` | One to One | Each entity relates to exactly one other |
| `}o--||` | Many to One | Many entities relate to one |
| `||--o{` | One to Many | One entity relates to many |
| `}o--o{` | Many to Many | Many entities relate to many |
| `}o--|{` | One or More to Many | At least one relates to many |

## Notes

- Entity names should be in UPPERCASE
- Attributes are defined inside curly braces
- PK = Primary Key, FK = Foreign Key
- Relationship labels are optional but recommended
- Use descriptive relationship names

## Gotchas/Warnings

- ⚠️ **Entity Names**: Must be uppercase and use hyphens for multi-word names
- ⚠️ **Attributes**: Define inside entity blocks with type and constraints
- ⚠️ **Relationships**: Cardinality symbols must match the relationship direction
- ⚠️ **Complexity**: Large ER diagrams can become hard to read - break into modules

