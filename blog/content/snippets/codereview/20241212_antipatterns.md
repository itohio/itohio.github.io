---
title: "Common Antipatterns"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "antipatterns", "architecture"]
---


Common software antipatterns to avoid across all languages and architectures.

---

## God Object

**Problem**: One class/module does everything.

**Example**:
```python
class Application:
    def connect_database(self): pass
    def send_email(self): pass
    def process_payment(self): pass
    def generate_report(self): pass
    def authenticate_user(self): pass
    # ... 50 more methods
```

**Solution**: Split into focused, single-responsibility classes.

---

## Spaghetti Code

**Problem**: Unstructured, tangled control flow.

**Symptoms**:
- Deep nesting (5+ levels)
- Goto statements
- No clear separation of concerns
- Hard to follow logic

**Solution**: Refactor into functions, use early returns, apply design patterns.

---

## Golden Hammer

**Problem**: Using the same solution for every problem ("If all you have is a hammer...").

**Example**: Using microservices for a simple CRUD app, or using blockchain for everything.

**Solution**: Choose the right tool for the job.

---

## Premature Optimization

**Problem**: Optimizing before knowing where bottlenecks are.

**Quote**: "Premature optimization is the root of all evil" - Donald Knuth

**Solution**: Profile first, optimize later.

---

## Cargo Cult Programming

**Problem**: Copying code/patterns without understanding why.

**Example**: Adding `try-catch` everywhere "because best practices say so".

**Solution**: Understand the reasoning behind patterns before applying them.

---

## Big Ball of Mud

**Problem**: No discernible architecture, everything depends on everything.

**Solution**: Introduce layers, modules, and clear boundaries.

---

## Lava Flow

**Problem**: Dead code that nobody dares to remove.

**Solution**: Use version control, remove unused code confidently.

---

## Copy-Paste Programming

**Problem**: Duplicating code instead of abstracting.

**Solution**: DRY principle - extract common functionality.

---

## Magic Numbers

**Problem**: Hardcoded values without explanation.

```python
# ❌ Bad
if status == 3:
    process()

# ✅ Good
STATUS_APPROVED = 3
if status == STATUS_APPROVED:
    process()
```

---

## Shotgun Surgery

**Problem**: One change requires modifications in many places.

**Solution**: Better abstraction and encapsulation.

---