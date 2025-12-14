---
title: "Haskell Code Smells"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "haskell", "code-smells"]
---


Common code smells in Haskell and how to fix them.

---

## Partial Functions

```haskell
-- ❌ Bad: head can fail
getFirst :: [a] -> a
getFirst xs = head xs

-- ✅ Good: Use Maybe
getFirst :: [a] -> Maybe a
getFirst [] = Nothing
getFirst (x:_) = Just x
```

---

## Not Using Pattern Matching

```haskell
-- ❌ Bad
processResult :: Either String Int -> String
processResult r = 
    if isLeft r 
    then fromLeft "" r 
    else show (fromRight 0 r)

-- ✅ Good
processResult :: Either String Int -> String
processResult (Left err) = err
processResult (Right val) = show val
```

---

## Lazy IO

```haskell
-- ❌ Bad: Lazy IO can leak resources
readConfig :: FilePath -> IO Config
readConfig path = do
    contents <- readFile path
    return (parse contents)

-- ✅ Good: Strict IO
import qualified Data.Text.IO as TIO
readConfig :: FilePath -> IO Config
readConfig path = do
    contents <- TIO.readFile path
    return (parse contents)
```

---

## String Instead of Text

```haskell
-- ❌ Bad: String is [Char], inefficient
processName :: String -> String
processName name = map toUpper name

-- ✅ Good: Use Text
import qualified Data.Text as T
processName :: T.Text -> T.Text
processName = T.toUpper
```

---

## Not Using Applicative

```haskell
-- ❌ Bad
validateUser :: Maybe String -> Maybe Int -> Maybe User
validateUser maybeName maybeAge =
    case maybeName of
        Nothing -> Nothing
        Just name -> case maybeAge of
            Nothing -> Nothing
            Just age -> Just (User name age)

-- ✅ Good
validateUser :: Maybe String -> Maybe Int -> Maybe User
validateUser maybeName maybeAge = User <$> maybeName <*> maybeAge
```

---