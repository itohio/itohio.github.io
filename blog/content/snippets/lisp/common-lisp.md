---
title: "Common Lisp Essentials"
date: 2024-12-12
draft: false
description: "Common Lisp basics and useful patterns"
tags: ["lisp", "common-lisp", "functional-programming"]
---



## Basic Syntax

```lisp
; Comments start with semicolon

;; Variables
(defvar *global-var* 42)  ; Global variable (convention: *name*)
(defparameter *param* 100) ; Always re-evaluated
(let ((x 10) (y 20))      ; Local variables
  (+ x y))

;; Constants
(defconstant +pi+ 3.14159) ; Convention: +name+

;; Functions
(defun square (x)
  "Return the square of x"
  (* x x))

(defun greet (name)
  (format t "Hello, ~a!~%" name))

;; Lambda
(lambda (x) (* x x))
((lambda (x) (* x x)) 5) ; => 25

;; Multiple values
(defun divide-with-remainder (a b)
  (values (floor a b) (mod a b)))

(multiple-value-bind (quotient remainder)
    (divide-with-remainder 17 5)
  (format t "~a remainder ~a~%" quotient remainder))
```

## Data Structures

### Lists

```lisp
;; Create lists
'(1 2 3 4)
(list 1 2 3 4)
(cons 1 (cons 2 (cons 3 nil)))

;; Access elements
(first '(1 2 3))  ; => 1
(car '(1 2 3))    ; => 1
(rest '(1 2 3))   ; => (2 3)
(cdr '(1 2 3))    ; => (2 3)
(nth 2 '(a b c d)) ; => C (0-indexed)

;; List operations
(append '(1 2) '(3 4))    ; => (1 2 3 4)
(reverse '(1 2 3))        ; => (3 2 1)
(length '(1 2 3))         ; => 3
(member 2 '(1 2 3))       ; => (2 3)
(remove 2 '(1 2 3 2 4))   ; => (1 3 4)

;; Association lists
(defvar *alist* '((a . 1) (b . 2) (c . 3)))
(assoc 'b *alist*)        ; => (B . 2)
(cdr (assoc 'b *alist*))  ; => 2
```

### Vectors (Arrays)

```lisp
;; Create vector
(vector 1 2 3 4)
#(1 2 3 4)

;; Access elements
(aref #(1 2 3) 1)  ; => 2

;; Modify
(let ((v (vector 1 2 3)))
  (setf (aref v 1) 99)
  v)  ; => #(1 99 3)

;; Length
(length #(1 2 3))  ; => 3
```

### Hash Tables

```lisp
;; Create hash table
(defvar *ht* (make-hash-table))

;; Set values
(setf (gethash 'key1 *ht*) 100)
(setf (gethash 'key2 *ht*) 200)

;; Get values
(gethash 'key1 *ht*)  ; => 100, T
(gethash 'key3 *ht* 'default)  ; => DEFAULT, NIL

;; Iterate
(maphash (lambda (k v)
           (format t "~a => ~a~%" k v))
         *ht*)

;; Remove
(remhash 'key1 *ht*)
```

## Control Flow

### Conditionals

```lisp
;; if
(if (> x 10)
    (print "Greater")
    (print "Not greater"))

;; when (no else clause)
(when (> x 10)
  (print "Greater")
  (do-something))

;; unless
(unless (< x 0)
  (print "Non-negative"))

;; cond (multiple conditions)
(cond
  ((< x 0) "Negative")
  ((= x 0) "Zero")
  ((> x 0) "Positive")
  (t "Unknown"))  ; t is default case

;; case (switch)
(case (day-of-week)
  ((monday tuesday wednesday thursday friday) "Weekday")
  ((saturday sunday) "Weekend")
  (otherwise "Unknown"))
```

### Loops

```lisp
;; dotimes (count loop)
(dotimes (i 5)
  (print i))

;; dolist (iterate list)
(dolist (item '(a b c))
  (print item))

;; loop macro (powerful)
(loop for i from 1 to 10
      collect i)  ; => (1 2 3 4 5 6 7 8 9 10)

(loop for i from 1 to 10
      when (evenp i)
      collect i)  ; => (2 4 6 8 10)

(loop for x in '(1 2 3 4 5)
      sum x)  ; => 15

(loop for i from 1 to 100
      while (< i 50)
      collect i)

;; do (general loop)
(do ((i 0 (1+ i)))
    ((>= i 5))
  (print i))
```

## Higher-Order Functions

```lisp
;; map
(mapcar #'square '(1 2 3 4))  ; => (1 4 9 16)
(mapcar #'+ '(1 2 3) '(10 20 30))  ; => (11 22 33)

;; filter
(remove-if-not #'evenp '(1 2 3 4 5 6))  ; => (2 4 6)
(remove-if #'oddp '(1 2 3 4 5 6))       ; => (2 4 6)

;; reduce
(reduce #'+ '(1 2 3 4 5))  ; => 15
(reduce #'max '(3 1 4 1 5 9 2 6))  ; => 9

;; apply
(apply #'+ '(1 2 3))  ; => 6

;; funcall
(funcall #'+ 1 2 3)  ; => 6
```

## Macros

```lisp
;; Simple macro
(defmacro when-positive (x &body body)
  `(when (> ,x 0)
     ,@body))

(when-positive 5
  (print "Positive")
  (print "Yes"))

;; Macro with gensym (avoid variable capture)
(defmacro with-gensyms (syms &body body)
  `(let ,(mapcar (lambda (s) `(,s (gensym))) syms)
     ,@body))

;; Anaphoric if
(defmacro aif (test then &optional else)
  `(let ((it ,test))
     (if it ,then ,else)))

(aif (find 'x '(a b x c))
     (format t "Found: ~a~%" it)
     (format t "Not found~%"))
```

## CLOS (Common Lisp Object System)

```lisp
;; Define class
(defclass person ()
  ((name :initarg :name
         :accessor person-name)
   (age :initarg :age
        :accessor person-age
        :initform 0)))

;; Create instance
(defvar *john* (make-instance 'person
                              :name "John"
                              :age 30))

;; Access slots
(person-name *john*)  ; => "John"
(setf (person-age *john*) 31)

;; Methods
(defmethod greet ((p person))
  (format t "Hello, I'm ~a~%" (person-name p)))

(greet *john*)

;; Inheritance
(defclass employee (person)
  ((company :initarg :company
            :accessor employee-company)))

;; Multiple dispatch
(defmethod combine ((x number) (y number))
  (+ x y))

(defmethod combine ((x string) (y string))
  (concatenate 'string x y))

(defmethod combine ((x list) (y list))
  (append x y))
```

## Practical Examples

### File I/O

```lisp
;; Read file
(with-open-file (stream "file.txt")
  (loop for line = (read-line stream nil)
        while line
        collect line))

;; Write file
(with-open-file (stream "output.txt"
                        :direction :output
                        :if-exists :supersede)
  (format stream "Hello, World!~%"))

;; Read S-expressions
(with-open-file (stream "data.lisp")
  (read stream))
```

### Error Handling

```lisp
;; handler-case (like try-catch)
(handler-case
    (/ 1 0)
  (division-by-zero ()
    (format t "Cannot divide by zero~%"))
  (error (e)
    (format t "Error: ~a~%" e)))

;; unwind-protect (like finally)
(unwind-protect
    (progn
      (open-resource)
      (use-resource))
  (close-resource))  ; Always executed
```

### String Operations

```lisp
;; Concatenate
(concatenate 'string "Hello" " " "World")

;; Format
(format nil "~a is ~d years old" "John" 30)

;; Subseq
(subseq "Hello World" 0 5)  ; => "Hello"

;; Search
(search "World" "Hello World")  ; => 6

;; Replace
(substitute #\- #\Space "Hello World")  ; => "Hello-World"
```

## Quicklisp (Package Manager)

```lisp
;; Load Quicklisp
(load "~/quicklisp/setup.lisp")

;; Install package
(ql:quickload "alexandria")
(ql:quickload "cl-ppcre")  ; Regex library

;; Use package
(use-package :alexandria)
```

## Further Reading

- [Practical Common Lisp](http://www.gigamonkeys.com/book/)
- [Common Lisp HyperSpec](http://www.lispworks.com/documentation/HyperSpec/Front/index.htm)
- [Quicklisp](https://www.quicklisp.org/)
- [SBCL](http://www.sbcl.org/) - Steel Bank Common Lisp (recommended implementation)

