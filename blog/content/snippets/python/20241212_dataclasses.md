---
title: "Python Dataclasses"
date: 2024-12-12
draft: false
category: "python"
tags: ["python-knowhow", "dataclasses", "python3.7+"]
---


Python dataclasses for clean, boilerplate-free data structures.

---

## Basic Usage

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

# Automatically generates __init__, __repr__, __eq__
p1 = Point(1.0, 2.0)
p2 = Point(1.0, 2.0)

print(p1)           # Point(x=1.0, y=2.0)
print(p1 == p2)     # True
```

---

## Default Values

```python
from dataclasses import dataclass, field

@dataclass
class User:
    name: str
    age: int = 0
    email: str = "unknown@example.com"
    active: bool = True

user1 = User("Alice")
print(user1)
# User(name='Alice', age=0, email='unknown@example.com', active=True)

user2 = User("Bob", 30, "bob@example.com")
print(user2)
# User(name='Bob', age=30, email='bob@example.com', active=True)
```

---

## Mutable Default Values

```python
from dataclasses import dataclass, field
from typing import List

# ❌ WRONG - Don't use mutable defaults directly
# @dataclass
# class Team:
#     members: List[str] = []  # This is shared across instances!

# ✅ CORRECT - Use field with default_factory
@dataclass
class Team:
    name: str
    members: List[str] = field(default_factory=list)

team1 = Team("Alpha")
team2 = Team("Beta")

team1.members.append("Alice")
print(team1.members)  # ['Alice']
print(team2.members)  # []  (separate list!)
```

---

## Field Options

```python
from dataclasses import dataclass, field

@dataclass
class Product:
    name: str
    price: float
    
    # Exclude from __init__
    id: int = field(init=False)
    
    # Exclude from __repr__
    secret_key: str = field(repr=False, default="secret")
    
    # Exclude from comparison
    created_at: str = field(compare=False, default="2024-01-01")
    
    # Exclude from hash
    description: str = field(hash=False, default="")
    
    def __post_init__(self):
        # Set id after initialization
        self.id = hash(self.name)

product = Product("Widget", 9.99)
print(product)
# Product(name='Widget', price=9.99, id=-123456789, created_at='2024-01-01', description='')
```

---

## Post-Init Processing

```python
from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)
    
    def __post_init__(self):
        self.area = self.width * self.height

rect = Rectangle(10, 5)
print(rect.area)  # 50.0
```

---

## Frozen (Immutable) Dataclasses

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
# p.x = 3.0  # FrozenInstanceError!

# Frozen dataclasses are hashable
points = {Point(0, 0), Point(1, 1), Point(0, 0)}
print(len(points))  # 2 (duplicates removed)
```

---

## Inheritance

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

@dataclass
class Employee(Person):
    employee_id: int
    department: str

emp = Employee("Alice", 30, 12345, "Engineering")
print(emp)
# Employee(name='Alice', age=30, employee_id=12345, department='Engineering')
```

---

## Ordering

```python
from dataclasses import dataclass

@dataclass(order=True)
class Student:
    name: str = field(compare=False)
    grade: float

students = [
    Student("Charlie", 85.5),
    Student("Alice", 92.0),
    Student("Bob", 88.0),
]

students.sort()
for s in students:
    print(s)
# Student(name='Charlie', grade=85.5)
# Student(name='Bob', grade=88.0)
# Student(name='Alice', grade=92.0)
```

---

## Custom Sort Key

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class Task:
    priority: int = field(compare=True)
    name: str = field(compare=False)
    description: str = field(compare=False)

tasks = [
    Task(3, "Low priority", "Can wait"),
    Task(1, "Critical", "Do now!"),
    Task(2, "Medium", "Soon"),
]

tasks.sort()
for t in tasks:
    print(f"{t.priority}: {t.name}")
# 1: Critical
# 2: Medium
# 3: Low priority
```

---

## Conversion Methods

```python
from dataclasses import dataclass, asdict, astuple

@dataclass
class User:
    name: str
    age: int
    email: str

user = User("Alice", 30, "alice@example.com")

# Convert to dict
print(asdict(user))
# {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

# Convert to tuple
print(astuple(user))
# ('Alice', 30, 'alice@example.com')
```

---

## Nested Dataclasses

```python
from dataclasses import dataclass, asdict

@dataclass
class Address:
    street: str
    city: str
    zip_code: str

@dataclass
class Person:
    name: str
    age: int
    address: Address

person = Person(
    "Alice",
    30,
    Address("123 Main St", "Springfield", "12345")
)

print(person)
# Person(name='Alice', age=30, address=Address(street='123 Main St', city='Springfield', zip_code='12345'))

# Convert nested to dict
print(asdict(person))
# {'name': 'Alice', 'age': 30, 'address': {'street': '123 Main St', 'city': 'Springfield', 'zip_code': '12345'}}
```

---

## Slots for Memory Efficiency

```python
from dataclasses import dataclass

@dataclass(slots=True)  # Python 3.10+
class Point:
    x: float
    y: float

# Uses __slots__ internally for memory efficiency
# Faster attribute access, lower memory usage
# Cannot add new attributes dynamically
```

---

## KW-Only Fields

```python
from dataclasses import dataclass

@dataclass(kw_only=True)  # Python 3.10+
class User:
    name: str
    age: int
    email: str

# Must use keyword arguments
user = User(name="Alice", age=30, email="alice@example.com")
# user = User("Alice", 30, "alice@example.com")  # TypeError!
```

---

## Match-Case Support

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

def describe_point(p: Point):
    match p:
        case Point(0, 0):
            return "Origin"
        case Point(x, 0):
            return f"On X-axis at {x}"
        case Point(0, y):
            return f"On Y-axis at {y}"
        case Point(x, y):
            return f"At ({x}, {y})"

print(describe_point(Point(0, 0)))    # Origin
print(describe_point(Point(5, 0)))    # On X-axis at 5
print(describe_point(Point(3, 4)))    # At (3, 4)
```

---

## Validation

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int
    email: str
    
    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")
        if "@" not in self.email:
            raise ValueError("Invalid email")
        if not self.name:
            raise ValueError("Name cannot be empty")

try:
    user = User("", -5, "invalid")
except ValueError as e:
    print(e)  # Name cannot be empty
```

---

## Factory Pattern

```python
from dataclasses import dataclass, field
from typing import ClassVar
from datetime import datetime

@dataclass
class LogEntry:
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    level: str = "INFO"
    
    # Class variable (not an instance field)
    log_count: ClassVar[int] = 0
    
    def __post_init__(self):
        LogEntry.log_count += 1

log1 = LogEntry("First log")
log2 = LogEntry("Second log")

print(f"Total logs: {LogEntry.log_count}")  # 2
print(log1.timestamp < log2.timestamp)  # True
```

---

## JSON Serialization

```python
from dataclasses import dataclass, asdict
import json

@dataclass
class User:
    name: str
    age: int
    email: str

user = User("Alice", 30, "alice@example.com")

# To JSON
json_str = json.dumps(asdict(user))
print(json_str)
# {"name": "Alice", "age": 30, "email": "alice@example.com"}

# From JSON
data = json.loads(json_str)
user2 = User(**data)
print(user2)
# User(name='Alice', age=30, email='alice@example.com')
```

---

## Complex Example: API Response

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Author:
    id: int
    name: str
    email: str

@dataclass
class Comment:
    id: int
    text: str
    author: Author
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Post:
    id: int
    title: str
    content: str
    author: Author
    comments: List[Comment] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    published: bool = False
    views: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_comment(self, text: str, author: Author):
        comment = Comment(
            id=len(self.comments) + 1,
            text=text,
            author=author
        )
        self.comments.append(comment)
    
    def publish(self):
        self.published = True

# Usage
author = Author(1, "Alice", "alice@example.com")
post = Post(1, "Python Tips", "Here are some tips...", author)
post.tags = ["python", "programming"]
post.publish()

commenter = Author(2, "Bob", "bob@example.com")
post.add_comment("Great post!", commenter)

print(f"Post: {post.title}")
print(f"Published: {post.published}")
print(f"Comments: {len(post.comments)}")
```

---

## Comparison with Regular Classes

```python
# Without dataclass (verbose)
class PointOld:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"PointOld(x={self.x}, y={self.y})"
    
    def __eq__(self, other):
        if not isinstance(other, PointOld):
            return NotImplemented
        return self.x == other.x and self.y == other.y

# With dataclass (concise)
@dataclass
class Point:
    x: float
    y: float

# Both work the same, but dataclass is much shorter!
```

---