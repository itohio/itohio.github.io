---
title: "Go Secure Coding"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "go", "golang", "security"]
---


Secure coding practices for Go applications.

---

## SQL Injection Prevention

```go
// ❌ Vulnerable
username := r.FormValue("username")
query := fmt.Sprintf("SELECT * FROM users WHERE username = '%s'", username)
db.Query(query)

// ✅ Secure
username := r.FormValue("username")
query := "SELECT * FROM users WHERE username = $1"
db.Query(query, username)
```

---

## Command Injection Prevention

```go
// ❌ Vulnerable
filename := r.FormValue("file")
cmd := exec.Command("sh", "-c", "cat "+filename)
output, _ := cmd.Output()

// ✅ Secure
filename := r.FormValue("file")
if !regexp.MustCompile(`^[a-zA-Z0-9_.-]+$`).MatchString(filename) {
    return errors.New("invalid filename")
}
cmd := exec.Command("cat", filename)
output, err := cmd.Output()
```

---

## XSS Prevention

```go
// ❌ Vulnerable
func handler(w http.ResponseWriter, r *http.Request) {
    query := r.URL.Query().Get("q")
    fmt.Fprintf(w, "<h1>Results: %s</h1>", query)
}

// ✅ Secure: Use html/template
import "html/template"

func handler(w http.ResponseWriter, r *http.Request) {
    query := r.URL.Query().Get("q")
    tmpl := template.Must(template.New("search").Parse("<h1>Results: {{.}}</h1>"))
    tmpl.Execute(w, query)
}
```

---

## Secure Password Hashing

```go
// ❌ Insecure
import "crypto/md5"
hash := md5.Sum([]byte(password))

// ✅ Secure
import "golang.org/x/crypto/bcrypt"

hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
err = bcrypt.CompareHashAndPassword(hashedPassword, []byte(password))
```

---

## Secure Random Generation

```go
// ❌ Insecure
import "math/rand"
token := make([]byte, 32)
rand.Read(token)

// ✅ Secure
import "crypto/rand"
token := make([]byte, 32)
_, err := rand.Read(token)
```

---