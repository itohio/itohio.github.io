---
title: "Clojure Essentials"
date: 2024-12-12
draft: false
description: "Clojure for JVM development"
tags: ["clojure", "lisp", "jvm", "functional-programming"]
---



## Basic Syntax

```clojure
;; Comments start with semicolon

;; Variables (immutable by default)
(def x 42)
(def pi 3.14159)

;; Functions
(defn square [x]
  (* x x))

(defn greet [name]
  (str "Hello, " name "!"))

;; Multi-arity functions
(defn greet
  ([] (greet "World"))
  ([name] (str "Hello, " name "!")))

;; Anonymous functions
(fn [x] (* x x))
#(* % %)  ; Shorthand
((fn [x] (* x x)) 5)  ; => 25

;; Let bindings
(let [x 10
      y 20]
  (+ x y))  ; => 30
```

## Data Structures (Immutable)

### Lists

```clojure
;; Create lists
'(1 2 3 4)
(list 1 2 3 4)

;; Access
(first '(1 2 3))   ; => 1
(rest '(1 2 3))    ; => (2 3)
(nth '(1 2 3) 1)   ; => 2

;; Operations
(cons 0 '(1 2 3))  ; => (0 1 2 3)
(conj '(1 2 3) 0)  ; => (0 1 2 3) (adds to front)
```

### Vectors

```clojure
;; Create vectors
[1 2 3 4]
(vector 1 2 3 4)

;; Access
(get [1 2 3] 1)    ; => 2
([1 2 3] 1)        ; => 2 (vector as function)
(nth [1 2 3] 1)    ; => 2

;; Operations
(conj [1 2 3] 4)   ; => [1 2 3 4] (adds to end)
(assoc [1 2 3] 1 99)  ; => [1 99 3]
(subvec [1 2 3 4 5] 1 4)  ; => [2 3 4]
```

### Maps

```clojure
;; Create maps
{:name "John" :age 30}
(hash-map :name "John" :age 30)

;; Access
(get {:name "John"} :name)  ; => "John"
(:name {:name "John"})      ; => "John" (keyword as function)
({:name "John"} :name)      ; => "John" (map as function)

;; Operations
(assoc {:name "John"} :age 30)  ; => {:name "John" :age 30}
(dissoc {:name "John" :age 30} :age)  ; => {:name "John"}
(merge {:a 1} {:b 2})  ; => {:a 1 :b 2}
(update {:count 5} :count inc)  ; => {:count 6}
```

### Sets

```clojure
;; Create sets
#{1 2 3 4}
(hash-set 1 2 3 4)

;; Operations
(conj #{1 2 3} 4)  ; => #{1 2 3 4}
(disj #{1 2 3 4} 2)  ; => #{1 3 4}
(contains? #{1 2 3} 2)  ; => true

;; Set operations
(clojure.set/union #{1 2} #{2 3})  ; => #{1 2 3}
(clojure.set/intersection #{1 2 3} #{2 3 4})  ; => #{2 3}
(clojure.set/difference #{1 2 3} #{2 3 4})  ; => #{1}
```

## Control Flow

### Conditionals

```clojure
;; if
(if (> x 10)
  "Greater"
  "Not greater")

;; when (no else clause)
(when (> x 10)
  (println "Greater")
  (do-something))

;; cond
(cond
  (< x 0) "Negative"
  (= x 0) "Zero"
  (> x 0) "Positive"
  :else "Unknown")

;; case
(case day
  :monday "Start of week"
  :friday "End of week"
  (:saturday :sunday) "Weekend"
  "Midweek")

;; if-let (bind and test)
(if-let [result (find-something)]
  (println "Found:" result)
  (println "Not found"))

;; when-let
(when-let [result (find-something)]
  (println "Found:" result)
  (process result))
```

## Sequences and Collections

### Sequence Operations

```clojure
;; map
(map inc [1 2 3 4])  ; => (2 3 4 5)
(map + [1 2 3] [10 20 30])  ; => (11 22 33)

;; filter
(filter even? [1 2 3 4 5 6])  ; => (2 4 6)
(remove odd? [1 2 3 4 5 6])   ; => (2 4 6)

;; reduce
(reduce + [1 2 3 4 5])  ; => 15
(reduce + 100 [1 2 3])  ; => 106 (with initial value)

;; take/drop
(take 3 [1 2 3 4 5])  ; => (1 2 3)
(drop 2 [1 2 3 4 5])  ; => (3 4 5)

;; partition
(partition 2 [1 2 3 4 5 6])  ; => ((1 2) (3 4) (5 6))
(partition-all 2 [1 2 3 4 5])  ; => ((1 2) (3 4) (5))

;; group-by
(group-by even? [1 2 3 4 5 6])  ; => {false [1 3 5], true [2 4 6]}

;; sort
(sort [3 1 4 1 5 9])  ; => (1 1 3 4 5 9)
(sort-by :age [{:name "John" :age 30} {:name "Jane" :age 25}])
```

### Lazy Sequences

```clojure
;; range
(range 10)  ; => (0 1 2 3 4 5 6 7 8 9)
(range 5 10)  ; => (5 6 7 8 9)

;; repeat
(take 5 (repeat "x"))  ; => ("x" "x" "x" "x" "x")

;; cycle
(take 5 (cycle [1 2 3]))  ; => (1 2 3 1 2)

;; iterate
(take 5 (iterate inc 0))  ; => (0 1 2 3 4)

;; lazy-seq (custom lazy sequence)
(defn fibonacci
  ([] (fibonacci 0 1))
  ([a b] (lazy-seq (cons a (fibonacci b (+ a b))))))

(take 10 (fibonacci))  ; => (0 1 1 2 3 5 8 13 21 34)
```

## Destructuring

```clojure
;; Vector destructuring
(let [[a b c] [1 2 3]]
  (+ a b c))  ; => 6

;; With rest
(let [[first & rest] [1 2 3 4 5]]
  [first rest])  ; => [1 (2 3 4 5)]

;; Map destructuring
(let [{:keys [name age]} {:name "John" :age 30}]
  (str name " is " age))  ; => "John is 30"

;; With defaults
(let [{:keys [name age] :or {age 0}} {:name "John"}]
  age)  ; => 0

;; Function parameters
(defn greet [{:keys [name age]}]
  (str name " is " age " years old"))

(greet {:name "John" :age 30})
```

## Macros

```clojure
;; Simple macro
(defmacro unless [test & body]
  `(if (not ~test)
     (do ~@body)))

(unless false
  (println "This will print"))

;; Macro with gensym
(defmacro with-logging [expr]
  `(let [result# ~expr]
     (println "Result:" result#)
     result#))

;; Threading macros
(-> 5
    (+ 3)
    (* 2)
    (- 1))  ; => 15 (same as (- (* (+ 5 3) 2) 1))

(->> [1 2 3 4 5]
     (map inc)
     (filter even?)
     (reduce +))  ; => 12
```

## Namespaces

```clojure
;; Define namespace
(ns myapp.core
  (:require [clojure.string :as str]
            [clojure.set :as set]))

;; Use functions
(str/upper-case "hello")  ; => "HELLO"
(set/union #{1 2} #{2 3})  ; => #{1 2 3}

;; Refer specific functions
(ns myapp.core
  (:require [clojure.string :refer [upper-case lower-case]]))

(upper-case "hello")  ; => "HELLO"
```

## State Management

### Atoms (Synchronous, Independent)

```clojure
;; Create atom
(def counter (atom 0))

;; Read
@counter  ; => 0

;; Update
(swap! counter inc)  ; => 1
(swap! counter + 10)  ; => 11

;; Set
(reset! counter 0)  ; => 0
```

### Refs (Synchronous, Coordinated)

```clojure
;; Create refs
(def account1 (ref 1000))
(def account2 (ref 500))

;; Transaction
(dosync
  (alter account1 - 100)
  (alter account2 + 100))

@account1  ; => 900
@account2  ; => 600
```

### Agents (Asynchronous)

```clojure
;; Create agent
(def logger (agent []))

;; Send action
(send logger conj "Log entry 1")
(send logger conj "Log entry 2")

;; Wait for completion
(await logger)

@logger  ; => ["Log entry 1" "Log entry 2"]
```

## Java Interop

```clojure
;; Call static method
(Math/pow 2 3)  ; => 8.0

;; Create object
(def date (java.util.Date.))

;; Call method
(.getTime date)

;; Access field
(.-field object)

;; Import
(import '(java.util Date Calendar))
(import '[java.io File FileReader])

;; Chaining
(.. "hello"
    (toUpperCase)
    (substring 0 3))  ; => "HEL"
```

## Practical Examples

### File I/O

```clojure
;; Read file
(slurp "file.txt")

;; Write file
(spit "output.txt" "Hello, World!")

;; Read lines
(with-open [rdr (clojure.java.io/reader "file.txt")]
  (doall (line-seq rdr)))
```

### HTTP Request

```clojure
;; Using clj-http library
(require '[clj-http.client :as http])

;; GET request
(http/get "https://api.example.com/data")

;; POST request
(http/post "https://api.example.com/data"
           {:body (json/write-str {:key "value"})
            :headers {"Content-Type" "application/json"}})
```

### JSON Processing

```clojure
;; Using cheshire library
(require '[cheshire.core :as json])

;; Parse JSON
(json/parse-string "{\"name\":\"John\",\"age\":30}" true)
; => {:name "John" :age 30}

;; Generate JSON
(json/generate-string {:name "John" :age 30})
; => "{\"name\":\"John\",\"age\":30}"
```

## Testing

```clojure
(ns myapp.core-test
  (:require [clojure.test :refer :all]
            [myapp.core :refer :all]))

(deftest test-square
  (testing "Square function"
    (is (= 4 (square 2)))
    (is (= 9 (square 3)))))

(deftest test-greet
  (testing "Greet function"
    (is (= "Hello, John!" (greet "John")))))
```

## Further Reading

- [Clojure Documentation](https://clojure.org/)
- [Clojure for the Brave and True](https://www.braveclojure.com/)
- [ClojureDocs](https://clojuredocs.org/)
- [Leiningen](https://leiningen.org/) - Build tool
- [deps.edn](https://clojure.org/guides/deps_and_cli) - Dependency management

