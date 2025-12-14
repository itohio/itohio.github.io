---
title: "C/C++ Code Smells"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "c", "cpp", "c++", "code-smells"]
---


Common code smells in C/C++ and how to fix them.

---

## Memory Leaks

```cpp
// ❌ Bad
void process() {
    int* data = new int[100];
    // ... use data ...
    // forgot to delete!
}

// ✅ Good: Use RAII
void process() {
    std::vector<int> data(100);
    // ... use data ...
    // automatically cleaned up
}

// ✅ Good: Smart pointers
void process() {
    auto data = std::make_unique<int[]>(100);
    // ... use data ...
    // automatically cleaned up
}
```

---

## Buffer Overflow

```c
// ❌ Bad
char buffer[10];
strcpy(buffer, user_input);  // Dangerous!

// ✅ Good
char buffer[10];
strncpy(buffer, user_input, sizeof(buffer) - 1);
buffer[sizeof(buffer) - 1] = '\0';

// ✅ Better (C++)
std::string buffer = user_input;
```

---

## Use After Free

```cpp
// ❌ Bad
int* ptr = new int(42);
delete ptr;
*ptr = 10;  // Use after free!

// ✅ Good
auto ptr = std::make_unique<int>(42);
// Can't use after it's freed
```

---

## Raw Pointers

```cpp
// ❌ Bad
Widget* createWidget() {
    return new Widget();
}

// ✅ Good
std::unique_ptr<Widget> createWidget() {
    return std::make_unique<Widget>();
}
```

---

## Not Using const

```cpp
// ❌ Bad
void process(std::vector<int>& data) {
    for (int i = 0; i < data.size(); i++) {
        std::cout << data[i];
    }
}

// ✅ Good
void process(const std::vector<int>& data) {
    for (const auto& item : data) {
        std::cout << item;
    }
}
```

---