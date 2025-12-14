---
title: "Python Code Smells"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "python", "code-smells"]
---


Common code smells in Python and how to fix them.

---

## Mutable Default Arguments

```python
# ❌ Bad: Mutable default argument
def append_to_list(item, my_list=[]):
    my_list.append(item)
    return my_list

print(append_to_list(1))  # [1]
print(append_to_list(2))  # [1, 2] - Unexpected!

# ✅ Good: Use None as default
def append_to_list(item, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list
```

---

## Bare Except

```python
# ❌ Bad: Catches everything including KeyboardInterrupt
try:
    risky_operation()
except:
    pass

# ✅ Good: Catch specific exceptions
try:
    risky_operation()
except (ValueError, TypeError) as e:
    logger.error(f"Operation failed: {e}")
    raise
```

---

## Using `is` for Value Comparison

```python
# ❌ Bad: Using 'is' for value comparison
if x is True:
    pass

if name is "John":
    pass

# ✅ Good: Use == for values
if x:  # or if x == True:
    pass

if name == "John":
    pass

# ✅ Correct use of 'is'
if x is None:
    pass
```

---

## Not Using List Comprehensions

```python
# ❌ Bad: Verbose loop
squares = []
for i in range(10):
    squares.append(i ** 2)

# ✅ Good: List comprehension
squares = [i ** 2 for i in range(10)]
```

---

## String Concatenation in Loops

```python
# ❌ Bad: Inefficient
result = ""
for item in items:
    result += str(item) + ","

# ✅ Good: Use join
result = ",".join(str(item) for item in items)
```

---

## Not Using Context Managers

```python
# ❌ Bad: Manual resource management
file = open('file.txt')
try:
    data = file.read()
finally:
    file.close()

# ✅ Good: Context manager
with open('file.txt') as file:
    data = file.read()
```

---

## Not Using `get()` for Dictionaries

```python
# ❌ Bad: KeyError risk
value = my_dict['key']

# ✅ Good: Use get with default
value = my_dict.get('key', default_value)
```

---

## Using `list` as Variable Name

```python
# ❌ Bad: Shadows built-in
list = [1, 2, 3]

# ✅ Good: Use descriptive name
items = [1, 2, 3]
```

---