---
title: "Architecture Diagrams (C4 Model)"
date: 2024-12-12T19:10:00Z
draft: false
description: "Create system architecture diagrams using C4 model with Mermaid"
tags: ["mermaid", "architecture", "diagram", "c4", "system-design", "software-architecture"]
category: "diagrams"
---

The C4 model provides a hierarchical way to visualize software architecture at different levels of abstraction: Context, Containers, Components, and Code. Perfect for documenting system architecture.

## Use Case

Use architecture diagrams when you need to:
- Document system architecture
- Show system boundaries and interactions
- Communicate design to stakeholders
- Plan system components and their relationships

## C4 Model Levels

1. **Context** - System in its environment (users, external systems)
2. **Container** - High-level technology choices (apps, databases, services)
3. **Component** - Components within a container
4. **Code** - Class diagrams (covered in UML snippet)

## Code

````markdown
```mermaid
C4Context
    title System Context Diagram
    
    Person(user, "User", "A user of the system")
    System(systemA, "System A", "Main system")
    System_Ext(systemB, "External System", "Third-party service")
    
    Rel(user, systemA, "Uses")
    Rel(systemA, systemB, "Calls API")
```
````

**Result:**

```mermaid
C4Context
    title System Context Diagram
    
    Person(user, "User", "A user of the system")
    System(systemA, "System A", "Main system")
    System_Ext(systemB, "External System", "Third-party service")
    
    Rel(user, systemA, "Uses")
    Rel(systemA, systemB, "Calls API")
```

## Explanation

- `Person()` - User or actor
- `System()` - Internal system
- `System_Ext()` - External system
- `Container()` - Application, database, etc.
- `Component()` - Internal component
- `Rel()` - Relationship with label

## Examples

### Example 1: System Context

````markdown
```mermaid
C4Context
    title Research Platform - System Context
    
    Person(researcher, "Researcher", "Conducts research and experiments")
    Person(admin, "Administrator", "Manages system")
    
    System(platform, "Research Platform", "Manages research projects, experiments, and data")
    
    System_Ext(storage, "Cloud Storage", "Stores research data")
    System_Ext(compute, "HPC Cluster", "Runs computational experiments")
    System_Ext(auth, "Auth Service", "Handles authentication")
    
    Rel(researcher, platform, "Uses", "HTTPS")
    Rel(admin, platform, "Administers", "HTTPS")
    Rel(platform, storage, "Stores/retrieves data", "S3 API")
    Rel(platform, compute, "Submits jobs", "SSH/API")
    Rel(platform, auth, "Authenticates users", "OAuth 2.0")
```
````

**Result:**

```mermaid
C4Context
    title Research Platform - System Context
    
    Person(researcher, "Researcher", "Conducts research and experiments")
    Person(admin, "Administrator", "Manages system")
    
    System(platform, "Research Platform", "Manages research projects, experiments, and data")
    
    System_Ext(storage, "Cloud Storage", "Stores research data")
    System_Ext(compute, "HPC Cluster", "Runs computational experiments")
    System_Ext(auth, "Auth Service", "Handles authentication")
    
    Rel(researcher, platform, "Uses", "HTTPS")
    Rel(admin, platform, "Administers", "HTTPS")
    Rel(platform, storage, "Stores/retrieves data", "S3 API")
    Rel(platform, compute, "Submits jobs", "SSH/API")
    Rel(platform, auth, "Authenticates users", "OAuth 2.0")
```

### Example 2: Container Diagram

````markdown
```mermaid
C4Container
    title Research Platform - Container Diagram
    
    Person(user, "Researcher")
    
    Container(web, "Web Application", "React", "Provides research UI")
    Container(api, "API Server", "Go", "Handles business logic")
    Container(worker, "Worker Service", "Python", "Processes experiments")
    ContainerDb(db, "Database", "PostgreSQL", "Stores research data")
    ContainerDb(cache, "Cache", "Redis", "Caches results")
    Container(queue, "Message Queue", "RabbitMQ", "Job queue")
    
    Rel(user, web, "Uses", "HTTPS")
    Rel(web, api, "Calls", "REST/JSON")
    Rel(api, db, "Reads/writes", "SQL")
    Rel(api, cache, "Reads/writes", "Redis protocol")
    Rel(api, queue, "Publishes jobs", "AMQP")
    Rel(worker, queue, "Consumes jobs", "AMQP")
    Rel(worker, db, "Updates results", "SQL")
```
````

**Result:**

```mermaid
C4Container
    title Research Platform - Container Diagram
    
    Person(user, "Researcher")
    
    Container(web, "Web Application", "React", "Provides research UI")
    Container(api, "API Server", "Go", "Handles business logic")
    Container(worker, "Worker Service", "Python", "Processes experiments")
    ContainerDb(db, "Database", "PostgreSQL", "Stores research data")
    ContainerDb(cache, "Cache", "Redis", "Caches results")
    Container(queue, "Message Queue", "RabbitMQ", "Job queue")
    
    Rel(user, web, "Uses", "HTTPS")
    Rel(web, api, "Calls", "REST/JSON")
    Rel(api, db, "Reads/writes", "SQL")
    Rel(api, cache, "Reads/writes", "Redis protocol")
    Rel(api, queue, "Publishes jobs", "AMQP")
    Rel(worker, queue, "Consumes jobs", "AMQP")
    Rel(worker, db, "Updates results", "SQL")
```

### Example 3: Component Diagram

````markdown
```mermaid
C4Component
    title API Server - Component Diagram
    
    Container(web, "Web App", "React")
    
    Component(controller, "API Controller", "Go", "Handles HTTP requests")
    Component(service, "Business Logic", "Go", "Core logic")
    Component(repo, "Repository", "Go", "Data access")
    ComponentDb(db, "Database", "PostgreSQL")
    
    Rel(web, controller, "Calls", "REST")
    Rel(controller, service, "Uses")
    Rel(service, repo, "Uses")
    Rel(repo, db, "Queries", "SQL")
```
````

**Result:**

```mermaid
C4Component
    title API Server - Component Diagram
    
    Container(web, "Web App", "React")
    
    Component(controller, "API Controller", "Go", "Handles HTTP requests")
    Component(service, "Business Logic", "Go", "Core logic")
    Component(repo, "Repository", "Go", "Data access")
    ComponentDb(db, "Database", "PostgreSQL")
    
    Rel(web, controller, "Calls", "REST")
    Rel(controller, service, "Uses")
    Rel(service, repo, "Uses")
    Rel(repo, db, "Queries", "SQL")
```

### Example 4: Simple Architecture with Flowchart

For simpler architectures, you can use flowcharts:

````markdown
```mermaid
graph TB
    subgraph "Client Layer"
        Web[Web Browser]
        Mobile[Mobile App]
    end
    
    subgraph "API Layer"
        Gateway[API Gateway]
        Auth[Auth Service]
    end
    
    subgraph "Business Layer"
        Service1[User Service]
        Service2[Data Service]
        Service3[Analytics Service]
    end
    
    subgraph "Data Layer"
        DB[(Database)]
        Cache[(Cache)]
        Storage[(Object Storage)]
    end
    
    Web --> Gateway
    Mobile --> Gateway
    Gateway --> Auth
    Gateway --> Service1
    Gateway --> Service2
    Gateway --> Service3
    Service1 --> DB
    Service2 --> DB
    Service2 --> Cache
    Service3 --> Storage
```
````

**Result:**

```mermaid
graph TB
    subgraph "Client Layer"
        Web[Web Browser]
        Mobile[Mobile App]
    end
    
    subgraph "API Layer"
        Gateway[API Gateway]
        Auth[Auth Service]
    end
    
    subgraph "Business Layer"
        Service1[User Service]
        Service2[Data Service]
        Service3[Analytics Service]
    end
    
    subgraph "Data Layer"
        DB[(Database)]
        Cache[(Cache)]
        Storage[(Object Storage)]
    end
    
    Web --> Gateway
    Mobile --> Gateway
    Gateway --> Auth
    Gateway --> Service1
    Gateway --> Service2
    Gateway --> Service3
    Service1 --> DB
    Service2 --> DB
    Service2 --> Cache
    Service3 --> Storage
```

## Notes

- Start with Context diagram for high-level view
- Use Container diagram for deployment architecture
- Component diagram for detailed internal structure
- Keep diagrams focused - one diagram per level

## Gotchas/Warnings

- ⚠️ **Level mixing**: Don't mix C4 levels in one diagram
- ⚠️ **Too detailed**: Keep appropriate level of abstraction
- ⚠️ **Technology**: Include technology choices in descriptions
- ⚠️ **Boundaries**: Clearly show system boundaries