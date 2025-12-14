---
title: "TypeScript Secure Coding"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "typescript", "javascript", "security"]
---


Secure coding practices for TypeScript applications.

---

## XSS Prevention

```typescript
// ❌ Vulnerable
function displayMessage(message: string) {
    document.getElementById('output')!.innerHTML = message;
}

// ✅ Secure
function displayMessage(message: string) {
    const element = document.getElementById('output')!;
    element.textContent = message;
}

// ✅ With sanitization
import DOMPurify from 'dompurify';
function displayMessage(message: string) {
    const clean = DOMPurify.sanitize(message);
    document.getElementById('output')!.innerHTML = clean;
}
```

---

## SQL Injection Prevention (Node.js)

```typescript
// ❌ Vulnerable
const username = req.body.username;
const query = `SELECT * FROM users WHERE username = '${username}'`;
db.query(query);

// ✅ Secure
const username = req.body.username;
const query = 'SELECT * FROM users WHERE username = ?';
db.query(query, [username]);
```

---

## Command Injection Prevention

```typescript
// ❌ Vulnerable
import { exec } from 'child_process';
const filename = req.query.file;
exec(`cat ${filename}`, (error, stdout) => {
    res.send(stdout);
});

// ✅ Secure
import { execFile } from 'child_process';
const filename = req.query.file as string;
if (!/^[a-zA-Z0-9_.-]+$/.test(filename)) {
    throw new Error('Invalid filename');
}
execFile('cat', [filename], (error, stdout) => {
    res.send(stdout);
});
```

---

## Secure Password Hashing

```typescript
// ❌ Insecure
import crypto from 'crypto';
const hash = crypto.createHash('md5').update(password).digest('hex');

// ✅ Secure
import bcrypt from 'bcrypt';
const saltRounds = 12;
const hash = await bcrypt.hash(password, saltRounds);
const match = await bcrypt.compare(password, hash);
```

---

## Secure Random Generation

```typescript
// ❌ Insecure
const token = Math.random().toString(36).substring(2);

// ✅ Secure
import crypto from 'crypto';
const token = crypto.randomBytes(32).toString('hex');
```

---