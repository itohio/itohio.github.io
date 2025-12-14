---
title: "SQL Language Basics"
date: 2024-12-12
draft: false
category: "database"
tags: ["database-knowhow", "sql", "basics"]
---


SQL language fundamentals - essential commands and patterns for relational databases.

---

## Data Definition Language (DDL)

### CREATE TABLE

```sql
-- Basic table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- With constraints
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    total DECIMAL(10, 2) NOT NULL CHECK (total >= 0),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT valid_status CHECK (status IN ('pending', 'processing', 'completed', 'cancelled'))
);

-- With indexes
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_category (category),
    INDEX idx_price (price),
    FULLTEXT INDEX idx_search (name, description)
);
```

### ALTER TABLE

```sql
-- Add column
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- Modify column
ALTER TABLE users ALTER COLUMN email TYPE VARCHAR(150);

-- Drop column
ALTER TABLE users DROP COLUMN phone;

-- Add constraint
ALTER TABLE users ADD CONSTRAINT email_format CHECK (email LIKE '%@%');

-- Add foreign key
ALTER TABLE orders ADD FOREIGN KEY (user_id) REFERENCES users(id);

-- Add index
CREATE INDEX idx_username ON users(username);

-- Rename table
ALTER TABLE users RENAME TO customers;
```

### DROP TABLE

```sql
-- Drop table
DROP TABLE users;

-- Drop if exists
DROP TABLE IF EXISTS users;

-- Drop with cascade (removes dependent objects)
DROP TABLE users CASCADE;
```

---

## Data Manipulation Language (DML)

### INSERT

```sql
-- Single row
INSERT INTO users (username, email) 
VALUES ('john_doe', 'john@example.com');

-- Multiple rows
INSERT INTO users (username, email) VALUES
    ('alice', 'alice@example.com'),
    ('bob', 'bob@example.com'),
    ('charlie', 'charlie@example.com');

-- Insert from SELECT
INSERT INTO archived_users (username, email)
SELECT username, email FROM users WHERE created_at < '2020-01-01';

-- Insert with RETURNING (PostgreSQL)
INSERT INTO users (username, email) 
VALUES ('jane', 'jane@example.com')
RETURNING id, created_at;

-- Upsert (INSERT ... ON CONFLICT)
INSERT INTO users (id, username, email)
VALUES (1, 'john', 'john@example.com')
ON CONFLICT (id) DO UPDATE SET
    username = EXCLUDED.username,
    email = EXCLUDED.email;
```

### SELECT

```sql
-- Basic SELECT
SELECT * FROM users;

-- Specific columns
SELECT id, username, email FROM users;

-- With WHERE
SELECT * FROM users WHERE created_at > '2023-01-01';

-- With multiple conditions
SELECT * FROM users 
WHERE created_at > '2023-01-01' 
  AND (username LIKE 'john%' OR email LIKE '%@gmail.com');

-- DISTINCT
SELECT DISTINCT category FROM products;

-- ORDER BY
SELECT * FROM users ORDER BY created_at DESC;
SELECT * FROM products ORDER BY price ASC, name;

-- LIMIT and OFFSET (pagination)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 20;

-- Aggregate functions
SELECT COUNT(*) FROM users;
SELECT AVG(price) FROM products;
SELECT SUM(total) FROM orders;
SELECT MIN(price), MAX(price) FROM products;

-- GROUP BY
SELECT category, COUNT(*), AVG(price)
FROM products
GROUP BY category;

-- HAVING (filter after GROUP BY)
SELECT category, COUNT(*) as count
FROM products
GROUP BY category
HAVING COUNT(*) > 5;
```

### UPDATE

```sql
-- Update single column
UPDATE users SET email = 'newemail@example.com' WHERE id = 1;

-- Update multiple columns
UPDATE users 
SET username = 'new_username', 
    email = 'new@example.com'
WHERE id = 1;

-- Update with calculation
UPDATE products SET price = price * 1.1 WHERE category = 'electronics';

-- Update from another table
UPDATE orders o
SET status = 'completed'
FROM users u
WHERE o.user_id = u.id AND u.username = 'john';

-- Update with RETURNING
UPDATE users 
SET email = 'updated@example.com' 
WHERE id = 1
RETURNING *;
```

### DELETE

```sql
-- Delete specific rows
DELETE FROM users WHERE id = 1;

-- Delete with condition
DELETE FROM users WHERE created_at < '2020-01-01';

-- Delete all rows (keep table structure)
DELETE FROM users;

-- Truncate (faster, resets auto-increment)
TRUNCATE TABLE users;

-- Delete with RETURNING
DELETE FROM users WHERE id = 1 RETURNING *;
```

---

## Joins

```sql
-- INNER JOIN (only matching rows)
SELECT u.username, o.total
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN (all from left table)
SELECT u.username, o.total
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- RIGHT JOIN (all from right table)
SELECT u.username, o.total
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;

-- FULL OUTER JOIN (all from both tables)
SELECT u.username, o.total
FROM users u
FULL OUTER JOIN orders o ON u.id = o.user_id;

-- Multiple joins
SELECT u.username, o.total, p.name
FROM users u
INNER JOIN orders o ON u.id = o.user_id
INNER JOIN order_items oi ON o.id = oi.order_id
INNER JOIN products p ON oi.product_id = p.id;

-- Self join
SELECT e1.name AS employee, e2.name AS manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.id;
```

---

## Subqueries

```sql
-- Subquery in WHERE
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE total > 100);

-- Subquery in SELECT
SELECT username,
       (SELECT COUNT(*) FROM orders WHERE user_id = users.id) AS order_count
FROM users;

-- Subquery in FROM
SELECT avg_price.category, avg_price.avg
FROM (
    SELECT category, AVG(price) as avg
    FROM products
    GROUP BY category
) AS avg_price
WHERE avg_price.avg > 50;

-- EXISTS
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id
);

-- NOT EXISTS
SELECT * FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id
);
```

---

## Common Table Expressions (CTE)

```sql
-- Basic CTE
WITH high_value_orders AS (
    SELECT user_id, SUM(total) as total_spent
    FROM orders
    WHERE total > 100
    GROUP BY user_id
)
SELECT u.username, hvo.total_spent
FROM users u
INNER JOIN high_value_orders hvo ON u.id = hvo.user_id;

-- Multiple CTEs
WITH 
    user_orders AS (
        SELECT user_id, COUNT(*) as order_count
        FROM orders
        GROUP BY user_id
    ),
    user_spending AS (
        SELECT user_id, SUM(total) as total_spent
        FROM orders
        GROUP BY user_id
    )
SELECT u.username, uo.order_count, us.total_spent
FROM users u
LEFT JOIN user_orders uo ON u.id = uo.user_id
LEFT JOIN user_spending us ON u.id = us.user_id;

-- Recursive CTE (hierarchy)
WITH RECURSIVE employee_hierarchy AS (
    -- Base case
    SELECT id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case
    SELECT e.id, e.name, e.manager_id, eh.level + 1
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.id
)
SELECT * FROM employee_hierarchy ORDER BY level, name;
```

---

## Window Functions

```sql
-- ROW_NUMBER
SELECT username, total,
       ROW_NUMBER() OVER (ORDER BY total DESC) as rank
FROM orders;

-- RANK (with gaps)
SELECT username, total,
       RANK() OVER (ORDER BY total DESC) as rank
FROM orders;

-- DENSE_RANK (no gaps)
SELECT username, total,
       DENSE_RANK() OVER (ORDER BY total DESC) as rank
FROM orders;

-- PARTITION BY
SELECT category, name, price,
       ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) as rank_in_category
FROM products;

-- Running total
SELECT date, amount,
       SUM(amount) OVER (ORDER BY date) as running_total
FROM transactions;

-- Moving average
SELECT date, price,
       AVG(price) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg_7days
FROM stock_prices;

-- LAG and LEAD
SELECT date, price,
       LAG(price, 1) OVER (ORDER BY date) as prev_price,
       LEAD(price, 1) OVER (ORDER BY date) as next_price
FROM stock_prices;
```

---

## Transactions

```sql
-- Basic transaction
BEGIN;
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- Rollback on error
BEGIN;
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    -- If error occurs
    ROLLBACK;

-- Savepoints
BEGIN;
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    SAVEPOINT my_savepoint;
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;
    -- Rollback to savepoint if needed
    ROLLBACK TO my_savepoint;
    -- Or commit
COMMIT;

-- Isolation levels
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

---

## String Functions

```sql
-- Concatenation
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM users;
SELECT first_name || ' ' || last_name AS full_name FROM users;  -- PostgreSQL

-- UPPER/LOWER
SELECT UPPER(username), LOWER(email) FROM users;

-- SUBSTRING
SELECT SUBSTRING(email, 1, 10) FROM users;

-- LENGTH
SELECT username, LENGTH(username) FROM users;

-- TRIM
SELECT TRIM('  hello  ');
SELECT LTRIM('  hello');
SELECT RTRIM('hello  ');

-- REPLACE
SELECT REPLACE(email, '@gmail.com', '@example.com') FROM users;

-- LIKE pattern matching
SELECT * FROM users WHERE email LIKE '%@gmail.com';
SELECT * FROM users WHERE username LIKE 'john%';
SELECT * FROM users WHERE phone LIKE '555-____';  -- _ matches single char

-- REGEXP (MySQL/PostgreSQL)
SELECT * FROM users WHERE email ~ '^[a-z]+@gmail\.com$';  -- PostgreSQL
SELECT * FROM users WHERE email REGEXP '^[a-z]+@gmail\\.com$';  -- MySQL
```

---

## Date Functions

```sql
-- Current date/time
SELECT CURRENT_DATE;
SELECT CURRENT_TIME;
SELECT CURRENT_TIMESTAMP;
SELECT NOW();

-- Date arithmetic
SELECT CURRENT_DATE + INTERVAL '7 days';  -- PostgreSQL
SELECT DATE_ADD(CURRENT_DATE, INTERVAL 7 DAY);  -- MySQL

-- Extract parts
SELECT EXTRACT(YEAR FROM created_at) FROM users;
SELECT EXTRACT(MONTH FROM created_at) FROM users;
SELECT DATE_PART('year', created_at) FROM users;  -- PostgreSQL

-- Format date
SELECT TO_CHAR(created_at, 'YYYY-MM-DD') FROM users;  -- PostgreSQL
SELECT DATE_FORMAT(created_at, '%Y-%m-%d') FROM users;  -- MySQL

-- Date difference
SELECT AGE(CURRENT_DATE, created_at) FROM users;  -- PostgreSQL
SELECT DATEDIFF(CURRENT_DATE, created_at) FROM users;  -- MySQL

-- Truncate to period
SELECT DATE_TRUNC('month', created_at) FROM users;  -- PostgreSQL
```

---

## Conditional Logic

```sql
-- CASE
SELECT username,
       CASE 
           WHEN total > 1000 THEN 'VIP'
           WHEN total > 500 THEN 'Premium'
           ELSE 'Regular'
       END AS customer_tier
FROM users;

-- COALESCE (return first non-null)
SELECT username, COALESCE(phone, email, 'No contact') AS contact FROM users;

-- NULLIF (return NULL if equal)
SELECT NULLIF(column_name, '') FROM table_name;

-- IF (MySQL)
SELECT IF(total > 100, 'High', 'Low') AS value_category FROM orders;
```

---

## Best Practices

```sql
-- ✅ Use parameterized queries (prevent SQL injection)
-- Application code:
-- cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

-- ✅ Use indexes for frequently queried columns
CREATE INDEX idx_email ON users(email);

-- ✅ Use EXPLAIN to analyze queries
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- ✅ Avoid SELECT *
SELECT id, username, email FROM users;  -- Better

-- ✅ Use transactions for multiple related operations
BEGIN;
    -- Multiple operations
COMMIT;

-- ✅ Use appropriate data types
-- INT for integers, VARCHAR for variable strings, TEXT for long text
-- DECIMAL for money, TIMESTAMP for dates

-- ❌ Avoid N+1 queries
-- Bad: Query in loop
-- Good: Use JOIN or IN clause
```

---

## Quick Reference

```sql
-- Create
CREATE TABLE table_name (column type constraints);

-- Read
SELECT columns FROM table WHERE condition;

-- Update
UPDATE table SET column = value WHERE condition;

-- Delete
DELETE FROM table WHERE condition;

-- Join
SELECT * FROM t1 JOIN t2 ON t1.id = t2.id;

-- Aggregate
SELECT COUNT(*), AVG(column) FROM table GROUP BY category;

-- Subquery
SELECT * FROM table WHERE id IN (SELECT id FROM other);

-- Transaction
BEGIN; ... COMMIT; or ROLLBACK;
```

---