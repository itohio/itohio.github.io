---
title: "Haskell Secure Coding"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "haskell", "security"]
---


Secure coding practices for Haskell applications.

---

## SQL Injection Prevention

```haskell
-- ❌ Vulnerable
import Database.PostgreSQL.Simple
getUserBad :: Connection -> String -> IO [User]
getUserBad conn username = 
    query_ conn $ fromString $ "SELECT * FROM users WHERE username = '" ++ username ++ "'"

-- ✅ Secure: Parameterized queries
getUserGood :: Connection -> String -> IO [User]
getUserGood conn username = 
    query conn "SELECT * FROM users WHERE username = ?" (Only username)
```

---

## Command Injection Prevention

```haskell
-- ❌ Vulnerable
import System.Process
runCommandBad :: String -> IO String
runCommandBad filename = 
    readProcess "sh" ["-c", "cat " ++ filename] ""

-- ✅ Secure
import System.Process
import Data.Char (isAlphaNum)
runCommandGood :: String -> IO (Either String String)
runCommandGood filename 
    | all (\c -> isAlphaNum c || c `elem` "._-") filename = 
        Right <$> readProcess "cat" [filename] ""
    | otherwise = return $ Left "Invalid filename"
```

---

## XSS Prevention

```haskell
-- ❌ Vulnerable
import Text.Blaze.Html5 as H
displayMessage :: String -> Html
displayMessage msg = H.div $ toHtml msg

-- ✅ Secure: Text.Blaze escapes by default
import Text.Blaze.Html5 as H
import qualified Data.Text as T
displayMessage :: T.Text -> Html
displayMessage msg = H.div $ toHtml msg

-- For raw HTML (use carefully)
displayTrustedHtml :: T.Text -> Html
displayTrustedHtml = preEscapedToHtml
```

---

## Secure Random Generation

```haskell
-- ❌ Insecure
import System.Random
generateToken :: IO Int
generateToken = randomRIO (0, maxBound)

-- ✅ Secure
import Crypto.Random
import qualified Data.ByteString as BS

generateToken :: IO BS.ByteString
generateToken = getRandomBytes 32
```

---

## Timing Attack Prevention

```haskell
-- ❌ Vulnerable to timing attacks
comparePasswords :: String -> String -> Bool
comparePasswords = (==)

-- ✅ Secure: Constant-time comparison
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import Crypto.MAC.HMAC

constantTimeCompare :: ByteString -> ByteString -> Bool
constantTimeCompare a b = 
    BS.length a == BS.length b && 
    BS.foldl' (\acc (x, y) -> acc && x == y) True (BS.zip a b)
```

---