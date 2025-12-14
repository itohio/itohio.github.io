---
title: "TypeScript Code Smells"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "typescript", "javascript", "code-smells"]
---


Common code smells in TypeScript and how to fix them.

---

## Using `any`

```typescript
// ❌ Bad
function process(data: any) {
    return data.value;
}

// ✅ Good
interface Data {
    value: string;
}
function process(data: Data) {
    return data.value;
}
```

---

## Not Using Optional Chaining

```typescript
// ❌ Bad
const city = user && user.address && user.address.city;

// ✅ Good
const city = user?.address?.city;
```

---

## Not Using Nullish Coalescing

```typescript
// ❌ Bad
const value = input || 'default';  // 0, false, '' are replaced

// ✅ Good
const value = input ?? 'default';  // Only null/undefined replaced
```

---

## Mutation of Function Parameters

```typescript
// ❌ Bad
function addItem(arr: Item[], item: Item) {
    arr.push(item);
    return arr;
}

// ✅ Good
function addItem(arr: readonly Item[], item: Item): Item[] {
    return [...arr, item];
}
```

---

## Not Using Union Types

```typescript
// ❌ Bad
function handleResponse(success: boolean, data?: any, error?: any) {
    if (success) {
        return data;
    }
    throw error;
}

// ✅ Good
type Success = { success: true; data: Data };
type Failure = { success: false; error: Error };
type Result = Success | Failure;

function handleResponse(result: Result) {
    if (result.success) {
        return result.data;
    }
    throw result.error;
}
```

---