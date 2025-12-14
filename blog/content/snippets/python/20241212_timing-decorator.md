---
title: "Function Timing Decorator"
date: 2024-12-12T17:00:00Z
draft: false
description: "Decorator to measure function execution time"
type: "snippet"
tags: ["python", "decorator", "performance", "timing"]
category: "python"
---



A simple decorator to measure and print the execution time of any function. Useful for quick performance profiling during development.

## Use Case

Use this when you want to quickly measure how long a function takes to execute without setting up complex profiling tools. Perfect for identifying slow functions during development.

## Code

```python
import time
from functools import wraps

def timing_decorator(func):
    """Decorator that prints the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper
```

## Explanation

The decorator uses `time.perf_counter()` which provides the highest resolution timer available on the system. It wraps the original function, measures the time before and after execution, and prints the elapsed time.

The `@wraps(func)` decorator preserves the original function's metadata (name, docstring, etc.).

## Parameters/Options

- `func`: The function to be timed
- Returns: The decorated function that includes timing

## Examples

### Example 1: Basic usage

```python
@timing_decorator
def slow_function():
    time.sleep(2)
    return "Done"

result = slow_function()
```

**Output:**
```
slow_function took 2.0012 seconds
```

### Example 2: With function arguments

```python
@timing_decorator
def process_data(data, multiplier=2):
    result = [x * multiplier for x in data]
    return result

output = process_data([1, 2, 3, 4, 5], multiplier=3)
```

**Output:**
```
process_data took 0.0001 seconds
```

### Example 3: Enhanced version with return value

```python
import time
from functools import wraps

def timing_decorator_v2(func):
    """Enhanced decorator that returns timing info."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # Return both result and timing
        return {
            'result': result,
            'elapsed_time': elapsed_time,
            'function_name': func.__name__
        }
    return wrapper

@timing_decorator_v2
def calculate(n):
    return sum(range(n))

output = calculate(1000000)
print(f"Result: {output['result']}")
print(f"Time: {output['elapsed_time']:.4f}s")
```

## Notes

For more detailed profiling, consider using:
- `cProfile` module for comprehensive profiling
- `line_profiler` for line-by-line profiling
- `memory_profiler` for memory usage profiling

This decorator is best for quick checks during development, not for production use.

## Gotchas/Warnings

- ⚠️ **Overhead**: The decorator itself adds minimal overhead (~microseconds)
- ⚠️ **Nested decorators**: Order matters when stacking multiple decorators
- ⚠️ **Async functions**: This won't work with async functions - use `asyncio` timing instead
- ⚠️ **Production use**: Remove or disable timing decorators in production for performance