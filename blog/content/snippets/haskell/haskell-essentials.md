---
title: "Haskell Essentials"
date: 2024-12-12T21:30:00Z
draft: false
description: "Most common Haskell patterns and functional programming idioms"
type: "snippet"
tags: ["haskell", "functional-programming", "monads", "type-classes", "haskell-knowhow"]
category: "haskell"
---



Essential Haskell patterns covering pure functions, type classes, monads, functors, and common idioms for functional programming.

## Use Case

Use these patterns when you need to:
- Write pure functional code
- Understand Haskell's type system
- Work with monads and functors
- Handle side effects functionally

## Basic Syntax

```haskell
-- Function definition
square :: Int -> Int
square x = x * x

-- Pattern matching
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Guards
abs' :: Int -> Int
abs' n
    | n < 0     = -n
    | otherwise = n

-- Let and where
cylinder :: Double -> Double -> Double
cylinder r h =
    let sideArea = 2 * pi * r * h
        topArea = pi * r^2
    in sideArea + 2 * topArea

-- Where clause
bmiTell :: Double -> Double -> String
bmiTell weight height
    | bmi <= 18.5 = "Underweight"
    | bmi <= 25.0 = "Normal"
    | bmi <= 30.0 = "Overweight"
    | otherwise   = "Obese"
  where bmi = weight / height^2
```

## Lists and List Comprehensions

```haskell
-- List operations
numbers = [1,2,3,4,5]
head numbers        -- 1
tail numbers        -- [2,3,4,5]
init numbers        -- [1,2,3,4]
last numbers        -- 5
take 3 numbers      -- [1,2,3]
drop 2 numbers      -- [3,4,5]

-- List comprehension
squares = [x^2 | x <- [1..10]]
evens = [x | x <- [1..20], x `mod` 2 == 0]
cartesian = [(x,y) | x <- [1,2,3], y <- [4,5,6]]

-- Infinite lists (lazy evaluation)
naturals = [1..]
evens' = [2,4..]
fibonacci = 0 : 1 : zipWith (+) fibonacci (tail fibonacci)
```

## Higher-Order Functions

```haskell
-- map, filter, fold
doubled = map (*2) [1,2,3,4,5]
evens = filter even [1..10]
sum' = foldl (+) 0 [1,2,3,4,5]
product' = foldr (*) 1 [1,2,3,4,5]

-- Function composition
(.) :: (b -> c) -> (a -> b) -> (a -> c)
f . g = \x -> f (g x)

-- Example
negateSum = negate . sum
result = negateSum [1,2,3]  -- -6

-- $ operator (function application)
sqrt $ 3 + 4 + 9  -- sqrt (3 + 4 + 9)
sum $ map (*2) $ filter (>3) [1..10]
```

## Type Classes

```haskell
-- Eq type class
class Eq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
    x /= y = not (x == y)

-- Ord type class
data TrafficLight = Red | Yellow | Green

instance Eq TrafficLight where
    Red == Red = True
    Yellow == Yellow = True
    Green == Green = True
    _ == _ = False

instance Ord TrafficLight where
    Red `compare` _ = LT
    _ `compare` Red = GT
    Yellow `compare` Yellow = EQ
    Yellow `compare` Green = LT
    Green `compare` Yellow = GT
    Green `compare` Green = EQ

-- Show and Read
instance Show TrafficLight where
    show Red = "Red light"
    show Yellow = "Yellow light"
    show Green = "Green light"
```

## Functors

```haskell
-- Functor type class
class Functor f where
    fmap :: (a -> b) -> f a -> f b

-- List is a Functor
fmap (*2) [1,2,3]  -- [2,4,6]

-- Maybe is a Functor
fmap (*2) (Just 3)  -- Just 6
fmap (*2) Nothing   -- Nothing

-- Custom Functor
data Box a = Box a deriving (Show)

instance Functor Box where
    fmap f (Box x) = Box (f x)

-- Usage
fmap (*2) (Box 3)  -- Box 6
```

## Applicative Functors

```haskell
-- Applicative type class
class Functor f => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b

-- Maybe Applicative
Just (*2) <*> Just 3  -- Just 6
Just (*) <*> Just 3 <*> Just 5  -- Just 15

-- List Applicative
[(+1), (*2)] <*> [1,2,3]  -- [2,3,4,2,4,6]

-- Applicative style
import Control.Applicative
(+) <$> Just 3 <*> Just 5  -- Just 8
```

## Monads

```haskell
-- Monad type class
class Applicative m => Monad m where
    return :: a -> m a
    (>>=) :: m a -> (a -> m b) -> m b

-- Maybe Monad
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide x y = Just (x / y)

calculation = do
    a <- safeDivide 10 2    -- Just 5
    b <- safeDivide 20 4    -- Just 5
    c <- safeDivide a b     -- Just 1
    return c

-- Equivalent to:
calculation' = 
    safeDivide 10 2 >>= \a ->
    safeDivide 20 4 >>= \b ->
    safeDivide a b

-- List Monad
pairs = do
    x <- [1,2,3]
    y <- [4,5,6]
    return (x,y)
-- [(1,4),(1,5),(1,6),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6)]
```

## IO Monad

```haskell
-- Basic IO
main :: IO ()
main = do
    putStrLn "What's your name?"
    name <- getLine
    putStrLn $ "Hello, " ++ name ++ "!"

-- Reading files
readFileExample :: IO ()
readFileExample = do
    contents <- readFile "input.txt"
    putStrLn contents

-- Writing files
writeFileExample :: IO ()
writeFileExample = do
    writeFile "output.txt" "Hello, World!"

-- Multiple actions
processFile :: FilePath -> IO ()
processFile path = do
    contents <- readFile path
    let processed = map toUpper contents
    writeFile (path ++ ".processed") processed
    putStrLn "File processed!"
```

## Common Type Classes

```haskell
-- Semigroup
class Semigroup a where
    (<>) :: a -> a -> a

-- Monoid
class Semigroup a => Monoid a where
    mempty :: a
    mappend :: a -> a -> a
    mconcat :: [a] -> a

-- Examples
[1,2,3] <> [4,5,6]  -- [1,2,3,4,5,6]
"Hello" <> " " <> "World"  -- "Hello World"
Sum 3 <> Sum 5  -- Sum 8
Product 3 <> Product 5  -- Product 15

-- Foldable
sum' = foldr (+) 0
product' = foldr (*) 1
length' = foldr (\_ acc -> acc + 1) 0

-- Traversable
sequence' :: Monad m => [m a] -> m [a]
traverse' :: Applicative f => (a -> f b) -> [a] -> f [b]
```

## Algebraic Data Types

```haskell
-- Sum types (OR)
data Bool' = True' | False'
data Maybe' a = Nothing' | Just' a

-- Product types (AND)
data Point = Point Double Double
data Person = Person { name :: String, age :: Int }

-- Recursive types
data List a = Empty | Cons a (List a)
data Tree a = Leaf a | Node (Tree a) a (Tree a)

-- Example tree operations
treeMap :: (a -> b) -> Tree a -> Tree b
treeMap f (Leaf x) = Leaf (f x)
treeMap f (Node left x right) = 
    Node (treeMap f left) (f x) (treeMap f right)

treeSum :: Num a => Tree a -> a
treeSum (Leaf x) = x
treeSum (Node left x right) = treeSum left + x + treeSum right
```

## Common Patterns

### Error Handling with Either

```haskell
data Either' a b = Left' a | Right' b

divide :: Double -> Double -> Either String Double
divide _ 0 = Left "Division by zero"
divide x y = Right (x / y)

-- Chaining with bind
calculation = do
    a <- divide 10 2
    b <- divide 20 4
    divide a b
```

### State Monad

```haskell
import Control.Monad.State

type Stack = [Int]

pop :: State Stack Int
pop = state $ \(x:xs) -> (x, xs)

push :: Int -> State Stack ()
push x = state $ \xs -> ((), x:xs)

stackOps :: State Stack Int
stackOps = do
    push 3
    push 5
    a <- pop
    b <- pop
    return (a + b)

-- Run: runState stackOps []  -- (8, [])
```

### Reader Monad

```haskell
import Control.Monad.Reader

type Config = String

computation :: Reader Config String
computation = do
    config <- ask
    return $ "Using config: " ++ config

-- Run: runReader computation "my-config"
```

## Notes

- Haskell is lazy - expressions evaluated only when needed
- Pure functions have no side effects
- IO monad isolates side effects
- Type inference is powerful - often don't need type signatures
- Pattern matching is exhaustive - compiler warns on missing cases

## Gotchas/Warnings

- ⚠️ **Lazy evaluation**: Can cause space leaks if not careful
- ⚠️ **Infinite lists**: Work due to laziness, but be careful with strict operations
- ⚠️ **Monad transformers**: Stacking monads requires understanding transformers
- ⚠️ **String performance**: Use `Text` or `ByteString` for performance