---
title: "Scheme Essentials"
date: 2024-12-12
draft: false
description: "Scheme programming fundamentals"
tags: ["scheme", "lisp", "functional-programming", "racket"]
---



## Basic Syntax

```scheme
; Comments start with semicolon

;; Variables
(define x 42)
(define pi 3.14159)

;; Functions
(define (square x)
  (* x x))

(define (factorial n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))

;; Lambda
(lambda (x) (* x x))
((lambda (x) (* x x)) 5) ; => 25

;; Let bindings
(let ((x 10)
      (y 20))
  (+ x y))  ; => 30

;; Let* (sequential bindings)
(let* ((x 10)
       (y (+ x 5)))
  y)  ; => 15

;; Letrec (recursive bindings)
(letrec ((even? (lambda (n)
                  (if (= n 0)
                      #t
                      (odd? (- n 1)))))
         (odd? (lambda (n)
                 (if (= n 0)
                     #f
                     (even? (- n 1))))))
  (even? 10))  ; => #t
```

## Data Structures

### Lists

```scheme
;; Create lists
'(1 2 3 4)
(list 1 2 3 4)
(cons 1 (cons 2 (cons 3 '())))

;; Access elements
(car '(1 2 3))    ; => 1
(cdr '(1 2 3))    ; => (2 3)
(cadr '(1 2 3))   ; => 2 (car of cdr)
(caddr '(1 2 3 4)) ; => 3

;; List operations
(append '(1 2) '(3 4))  ; => (1 2 3 4)
(reverse '(1 2 3))      ; => (3 2 1)
(length '(1 2 3))       ; => 3
(member 2 '(1 2 3))     ; => (2 3)

;; List predicates
(null? '())       ; => #t
(pair? '(1 2))    ; => #t
(list? '(1 2 3))  ; => #t
```

### Vectors

```scheme
;; Create vector
(vector 1 2 3 4)
#(1 2 3 4)

;; Access elements
(vector-ref #(1 2 3) 1)  ; => 2

;; Modify (if mutable)
(define v (vector 1 2 3))
(vector-set! v 1 99)
v  ; => #(1 99 3)

;; Length
(vector-length #(1 2 3))  ; => 3
```

## Control Flow

### Conditionals

```scheme
;; if
(if (> x 10)
    "Greater"
    "Not greater")

;; cond
(cond
  ((< x 0) "Negative")
  ((= x 0) "Zero")
  ((> x 0) "Positive")
  (else "Unknown"))

;; case
(case (day-of-week)
  ((monday tuesday wednesday thursday friday) "Weekday")
  ((saturday sunday) "Weekend")
  (else "Unknown"))

;; and, or
(and (> x 0) (< x 100))
(or (= x 0) (= x 1))
```

### Recursion

```scheme
;; Factorial
(define (factorial n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))

;; Tail-recursive factorial
(define (factorial-tail n)
  (define (iter n acc)
    (if (<= n 1)
        acc
        (iter (- n 1) (* n acc))))
  (iter n 1))

;; List sum
(define (sum lst)
  (if (null? lst)
      0
      (+ (car lst) (sum (cdr lst)))))

;; List length
(define (my-length lst)
  (if (null? lst)
      0
      (+ 1 (my-length (cdr lst)))))
```

## Higher-Order Functions

```scheme
;; map
(map square '(1 2 3 4))  ; => (1 4 9 16)
(map + '(1 2 3) '(10 20 30))  ; => (11 22 33)

;; filter
(filter even? '(1 2 3 4 5 6))  ; => (2 4 6)

;; fold (reduce)
(foldl + 0 '(1 2 3 4 5))  ; => 15
(foldr cons '() '(1 2 3))  ; => (1 2 3)

;; apply
(apply + '(1 2 3))  ; => 6
```

## Macros (Syntax Rules)

```scheme
;; Simple macro
(define-syntax when
  (syntax-rules ()
    ((when test body ...)
     (if test
         (begin body ...)))))

(when (> 5 3)
  (display "True")
  (newline))

;; Unless macro
(define-syntax unless
  (syntax-rules ()
    ((unless test body ...)
     (if (not test)
         (begin body ...)))))

;; While loop macro
(define-syntax while
  (syntax-rules ()
    ((while test body ...)
     (let loop ()
       (when test
         body ...
         (loop))))))
```

## Continuations

```scheme
;; call/cc (call-with-current-continuation)
(define (return-from-middle)
  (call/cc
    (lambda (return)
      (display "Before")
      (newline)
      (return 42)  ; Early return
      (display "After")  ; Never executed
      (newline))))

;; Non-local exit
(define (search-list lst target)
  (call/cc
    (lambda (return)
      (for-each
        (lambda (item)
          (when (equal? item target)
            (return item)))
        lst)
      #f)))
```

## Practical Examples

### List Processing

```scheme
;; Quicksort
(define (quicksort lst)
  (if (null? lst)
      '()
      (let ((pivot (car lst))
            (rest (cdr lst)))
        (append
          (quicksort (filter (lambda (x) (< x pivot)) rest))
          (list pivot)
          (quicksort (filter (lambda (x) (>= x pivot)) rest))))))

;; Flatten list
(define (flatten lst)
  (cond
    ((null? lst) '())
    ((pair? (car lst))
     (append (flatten (car lst))
             (flatten (cdr lst))))
    (else
     (cons (car lst) (flatten (cdr lst))))))

;; Remove duplicates
(define (remove-duplicates lst)
  (cond
    ((null? lst) '())
    ((member (car lst) (cdr lst))
     (remove-duplicates (cdr lst)))
    (else
     (cons (car lst) (remove-duplicates (cdr lst))))))
```

### File I/O

```scheme
;; Read file
(define (read-file filename)
  (call-with-input-file filename
    (lambda (port)
      (let loop ((line (read-line port))
                 (result '()))
        (if (eof-object? line)
            (reverse result)
            (loop (read-line port)
                  (cons line result)))))))

;; Write file
(define (write-file filename content)
  (call-with-output-file filename
    (lambda (port)
      (display content port))))
```

### String Operations

```scheme
;; String concatenation
(string-append "Hello" " " "World")

;; String to list
(string->list "Hello")  ; => (#\H #\e #\l #\l #\o)

;; List to string
(list->string '(#\H #\i))  ; => "Hi"

;; Substring
(substring "Hello World" 0 5)  ; => "Hello"

;; String length
(string-length "Hello")  ; => 5
```

## Racket-Specific Features

```scheme
#lang racket

;; Require modules
(require racket/list)
(require racket/string)

;; Struct (like record)
(struct point (x y) #:transparent)

(define p (point 3 4))
(point-x p)  ; => 3
(point-y p)  ; => 4

;; Pattern matching
(match '(1 2 3)
  [(list a b c) (+ a b c)])  ; => 6

(match 5
  [0 "zero"]
  [1 "one"]
  [n (format "number: ~a" n)])  ; => "number: 5"

;; For loops
(for ([i (in-range 5)])
  (displayln i))

(for/list ([i (in-range 5)])
  (* i i))  ; => (0 1 4 9 16)

(for/sum ([i (in-range 1 6)])
  i)  ; => 15
```

## Further Reading

- [The Little Schemer](https://mitpress.mit.edu/books/little-schemer-fourth-edition)
- [Structure and Interpretation of Computer Programs (SICP)](https://mitpress.mit.edu/sites/default/files/sicp/index.html)
- [Racket Documentation](https://docs.racket-lang.org/)
- [Scheme R7RS Standard](https://small.r7rs.org/)

