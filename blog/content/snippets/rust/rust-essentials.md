---
title: "Rust Essentials"
date: 2024-12-12T21:20:00Z
draft: false
description: "Most common Rust patterns and idioms"
type: "snippet"
tags: ["rust", "ownership", "borrowing", "error-handling", "rust-knowhow"]
category: "rust"
---



Essential Rust patterns covering ownership, borrowing, error handling, iterators, and common idioms. Master these to write idiomatic Rust code.

## Use Case

Use these patterns when you need to:
- Understand Rust's ownership system
- Handle errors properly
- Work with iterators and collections
- Write safe, concurrent code

## Ownership & Borrowing

### Move Semantics

```rust
// Ownership transfer (move)
let s1 = String::from("hello");
let s2 = s1;  // s1 is moved, no longer valid
// println!("{}", s1);  // Error: value borrowed after move

// Clone for deep copy
let s3 = s2.clone();
println!("{} {}", s2, s3);  // Both valid

// Copy trait for stack types
let x = 5;
let y = x;  // Copy, both valid
println!("{} {}", x, y);
```

### Borrowing

```rust
// Immutable borrow
fn calculate_length(s: &String) -> usize {
    s.len()  // Can read but not modify
}

let s = String::from("hello");
let len = calculate_length(&s);
println!("Length of '{}' is {}", s, len);  // s still valid

// Mutable borrow
fn append_world(s: &mut String) {
    s.push_str(", world!");
}

let mut s = String::from("hello");
append_world(&mut s);
println!("{}", s);  // "hello, world!"

// Rules:
// - One mutable reference OR multiple immutable references
// - References must always be valid
```

## Error Handling

### Result Type

```rust
use std::fs::File;
use std::io::{self, Read};

// Return Result
fn read_file(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;  // ? operator propagates errors
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

// Pattern matching
match read_file("data.txt") {
    Ok(contents) => println!("File: {}", contents),
    Err(e) => eprintln!("Error: {}", e),
}

// unwrap_or_else for default
let contents = read_file("data.txt")
    .unwrap_or_else(|_| String::from("default content"));
```

### Custom Errors

```rust
use std::fmt;

#[derive(Debug)]
enum MyError {
    NotFound(String),
    InvalidInput(String),
    IoError(std::io::Error),
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MyError::NotFound(msg) => write!(f, "Not found: {}", msg),
            MyError::InvalidInput(msg) => write!(f, "Invalid: {}", msg),
            MyError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for MyError {}

// Using custom error
fn process_data(input: &str) -> Result<String, MyError> {
    if input.is_empty() {
        return Err(MyError::InvalidInput("empty input".to_string()));
    }
    Ok(input.to_uppercase())
}
```

## Iterators

### Common Patterns

```rust
let numbers = vec![1, 2, 3, 4, 5];

// map, filter, collect
let doubled: Vec<i32> = numbers.iter()
    .map(|x| x * 2)
    .collect();

let evens: Vec<i32> = numbers.iter()
    .filter(|x| *x % 2 == 0)
    .copied()
    .collect();

// fold (reduce)
let sum: i32 = numbers.iter().sum();
let product: i32 = numbers.iter().product();
let custom = numbers.iter().fold(0, |acc, x| acc + x);

// find, any, all
let first_even = numbers.iter().find(|x| *x % 2 == 0);
let has_even = numbers.iter().any(|x| x % 2 == 0);
let all_positive = numbers.iter().all(|x| *x > 0);

// chain iterators
let a = vec![1, 2, 3];
let b = vec![4, 5, 6];
let combined: Vec<i32> = a.iter().chain(b.iter()).copied().collect();

// enumerate
for (i, value) in numbers.iter().enumerate() {
    println!("{}: {}", i, value);
}
```

## Pattern Matching

```rust
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

fn process_message(msg: Message) {
    match msg {
        Message::Quit => println!("Quit"),
        Message::Move { x, y } => println!("Move to ({}, {})", x, y),
        Message::Write(text) => println!("Text: {}", text),
        Message::ChangeColor(r, g, b) => println!("Color: ({}, {}, {})", r, g, b),
    }
}

// if let for single pattern
let some_value = Some(3);
if let Some(x) = some_value {
    println!("Got: {}", x);
}

// while let
let mut stack = vec![1, 2, 3];
while let Some(top) = stack.pop() {
    println!("{}", top);
}
```

## Traits

```rust
// Define trait
trait Summary {
    fn summarize(&self) -> String;
    
    // Default implementation
    fn summarize_author(&self) -> String {
        String::from("Unknown")
    }
}

// Implement trait
struct Article {
    title: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}: {}", self.title, &self.content[..50])
    }
}

// Trait bounds
fn notify<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}

// Multiple traits
fn process<T: Summary + Clone>(item: &T) {
    // Can use both Summary and Clone methods
}

// where clause for complex bounds
fn complex<T, U>(t: &T, u: &U) -> String
where
    T: Summary + Clone,
    U: Summary,
{
    format!("{} and {}", t.summarize(), u.summarize())
}
```

## Lifetimes

```rust
// Lifetime annotations
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

// Struct with lifetime
struct ImportantExcerpt<'a> {
    part: &'a str,
}

impl<'a> ImportantExcerpt<'a> {
    fn level(&self) -> i32 {
        3
    }
    
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("Attention: {}", announcement);
        self.part
    }
}
```

## Smart Pointers

```rust
use std::rc::Rc;
use std::cell::RefCell;

// Box for heap allocation
let b = Box::new(5);
println!("b = {}", b);

// Rc for multiple ownership
let a = Rc::new(vec![1, 2, 3]);
let b = Rc::clone(&a);
let c = Rc::clone(&a);
println!("Reference count: {}", Rc::strong_count(&a));  // 3

// RefCell for interior mutability
let data = RefCell::new(5);
*data.borrow_mut() += 1;
println!("{}", data.borrow());  // 6

// Combining Rc and RefCell
let shared_data = Rc::new(RefCell::new(vec![1, 2, 3]));
let clone1 = Rc::clone(&shared_data);
clone1.borrow_mut().push(4);
println!("{:?}", shared_data.borrow());  // [1, 2, 3, 4]
```

## Concurrency

```rust
use std::thread;
use std::sync::{Arc, Mutex};

// Spawn threads
let handle = thread::spawn(|| {
    println!("Hello from thread!");
});
handle.join().unwrap();

// Shared state with Arc and Mutex
let counter = Arc::new(Mutex::new(0));
let mut handles = vec![];

for _ in 0..10 {
    let counter = Arc::clone(&counter);
    let handle = thread::spawn(move || {
        let mut num = counter.lock().unwrap();
        *num += 1;
    });
    handles.push(handle);
}

for handle in handles {
    handle.join().unwrap();
}

println!("Result: {}", *counter.lock().unwrap());  // 10
```

## Common Macros

```rust
// vec! macro
let v = vec![1, 2, 3];

// println! and format!
println!("Hello, {}!", "world");
let s = format!("x = {}, y = {}", 10, 20);

// assert! and debug_assert!
assert!(2 + 2 == 4);
debug_assert!(expensive_check());  // Only in debug builds

// matches! macro
let foo = 'f';
assert!(matches!(foo, 'A'..='Z' | 'a'..='z'));

// dbg! macro
let a = 2;
let b = dbg!(a * 2) + 1;  // Prints: [src/main.rs:2] a * 2 = 4
```

## Notes

- Ownership prevents data races at compile time
- Use `&` for borrowing, `&mut` for mutable borrowing
- `?` operator simplifies error propagation
- Iterators are zero-cost abstractions
- Traits enable polymorphism without inheritance

## Gotchas/Warnings

- ⚠️ **Borrowing rules**: Can't have mutable and immutable borrows simultaneously
- ⚠️ **Lifetimes**: Compiler infers most, explicit only when ambiguous
- ⚠️ **Clone vs Copy**: Clone is explicit, Copy is implicit for simple types
- ⚠️ **unwrap()**: Panics on error - use in prototypes only