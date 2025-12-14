---
title: "C/C++ Secure Coding"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "c", "cpp", "c++", "security"]
---


Secure coding practices for C/C++ applications.

---

## Buffer Overflow Prevention

```c
// ❌ Vulnerable
char buffer[100];
gets(buffer);  // Never use!
sprintf(buffer, "%s", user_input);

// ✅ Secure
char buffer[100];
fgets(buffer, sizeof(buffer), stdin);
snprintf(buffer, sizeof(buffer), "%s", user_input);
```

---

## Integer Overflow

```c
// ❌ Vulnerable
size_t size = user_size;
char* buffer = malloc(size);

// ✅ Secure
size_t size = user_size;
if (size > MAX_ALLOWED_SIZE || size == 0) {
    return ERROR;
}
char* buffer = malloc(size);
if (!buffer) {
    return ERROR;
}
```

---

## Format String Vulnerability

```c
// ❌ Vulnerable
printf(user_input);  // Dangerous!

// ✅ Secure
printf("%s", user_input);
```

---

## SQL Injection (C++)

```cpp
// ❌ Vulnerable
std::string query = "SELECT * FROM users WHERE username = '" + username + "'";
mysql_query(conn, query.c_str());

// ✅ Secure: Prepared statements
MYSQL_STMT* stmt = mysql_stmt_init(conn);
const char* query = "SELECT * FROM users WHERE username = ?";
mysql_stmt_prepare(stmt, query, strlen(query));
mysql_stmt_bind_param(stmt, bind);
```

---

## Command Injection

```c
// ❌ Vulnerable
char cmd[256];
sprintf(cmd, "cat %s", filename);
system(cmd);

// ✅ Secure
if (!is_valid_filename(filename)) {
    return ERROR;
}
char* args[] = {"cat", filename, NULL};
execvp(args[0], args);
```

---

## Use After Free Prevention

```cpp
// ❌ Vulnerable
delete ptr;
// ... later ...
ptr->method();  // Use after free

// ✅ Secure: Smart pointers
auto ptr = std::make_unique<Object>();
// Can't use after scope ends
```

---