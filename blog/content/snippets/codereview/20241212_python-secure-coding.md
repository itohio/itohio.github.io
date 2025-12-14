---
title: "Python Secure Coding"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "python", "security"]
---


Secure coding practices for Python applications.

---

## SQL Injection Prevention

```python
# ❌ Vulnerable
user_input = request.GET['username']
query = f"SELECT * FROM users WHERE username = '{user_input}'"
cursor.execute(query)

# ✅ Secure: Parameterized queries
user_input = request.GET['username']
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, (user_input,))
```

---

## Command Injection Prevention

```python
# ❌ Vulnerable
filename = request.GET['file']
os.system(f'cat {filename}')

# ✅ Secure: Use subprocess with list
import subprocess
filename = request.GET['file']
if not re.match(r'^[a-zA-Z0-9_.-]+$', filename):
    raise ValueError("Invalid filename")
subprocess.run(['cat', filename], check=True)
```

---

## XSS Prevention

```python
# ❌ Vulnerable
from flask import Flask, request
@app.route('/search')
def search():
    query = request.args.get('q')
    return f'<h1>Results for: {query}</h1>'

# ✅ Secure: Escape output
from markupsafe import escape
@app.route('/search')
def search():
    query = request.args.get('q')
    return f'<h1>Results for: {escape(query)}</h1>'
```

---

## Secure Password Hashing

```python
# ❌ Insecure
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()

# ✅ Secure: Use bcrypt
import bcrypt
password = "user_password"
salt = bcrypt.gensalt(rounds=12)
hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
```

---

## Secure Random Generation

```python
# ❌ Insecure
import random
token = ''.join(random.choices(string.ascii_letters, k=32))

# ✅ Secure
import secrets
token = secrets.token_urlsafe(32)
```

---