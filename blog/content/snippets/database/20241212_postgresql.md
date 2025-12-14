---
title: "PostgreSQL Essentials & pgvector"
date: 2024-12-12
draft: false
category: "database"
tags: ["database-knowhow", "postgresql", "pgvector", "sql"]
---


PostgreSQL advanced features including procedures, views, pgvector for vector similarity search, and Docker setup.

---

## Docker Setup

### Docker Run

```bash
# Run PostgreSQL
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_DB=mydb \
  -v postgres-data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:16-alpine

# Connect to PostgreSQL
docker exec -it postgres psql -U myuser -d mydb

# With pgvector
docker run -d \
  --name postgres-vector \
  -e POSTGRES_PASSWORD=password \
  -v postgres-data:/var/lib/postgresql/data \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: postgres
    environment:
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U myuser"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # PostgreSQL with pgvector
  postgres-vector:
    image: pgvector/pgvector:pg16
    container_name: postgres-vector
    environment:
      POSTGRES_USER: vectoruser
      POSTGRES_PASSWORD: password
      POSTGRES_DB: vectordb
    volumes:
      - postgres-vector-data:/var/lib/postgresql/data
    ports:
      - "5433:5432"
    restart: unless-stopped

  # pgAdmin (optional)
  pgadmin:
    image: dpage/pgadmin4
    container_name: pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  postgres-data:
  postgres-vector-data:
```

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f postgres

# Connect to database
docker-compose exec postgres psql -U myuser -d mydb

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Basic Commands

```bash
# Connect to database
psql -h localhost -U myuser -d mydb

# List databases
\l

# Connect to database
\c mydb

# List tables
\dt

# Describe table
\d table_name

# List views
\dv

# List functions
\df

# List schemas
\dn

# Execute SQL file
\i /path/to/file.sql

# Export query results to CSV
\copy (SELECT * FROM users) TO '/path/to/users.csv' CSV HEADER

# Quit
\q
```

---

## Views

```sql
-- Create view
CREATE VIEW user_orders AS
SELECT 
    u.id,
    u.username,
    u.email,
    COUNT(o.id) AS order_count,
    SUM(o.total) AS total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username, u.email;

-- Query view
SELECT * FROM user_orders WHERE order_count > 5;

-- Materialized view (cached results)
CREATE MATERIALIZED VIEW user_stats AS
SELECT 
    u.id,
    u.username,
    COUNT(o.id) AS order_count,
    AVG(o.total) AS avg_order_value
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW user_stats;

-- Refresh concurrently (doesn't lock)
REFRESH MATERIALIZED VIEW CONCURRENTLY user_stats;

-- Drop view
DROP VIEW user_orders;
DROP MATERIALIZED VIEW user_stats;

-- Replace view
CREATE OR REPLACE VIEW user_orders AS
SELECT u.id, u.username, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username;
```

---

## Stored Procedures & Functions

### Functions

```sql
-- Simple function
CREATE OR REPLACE FUNCTION get_user_order_count(user_id_param INTEGER)
RETURNS INTEGER AS $$
DECLARE
    order_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO order_count
    FROM orders
    WHERE user_id = user_id_param;
    
    RETURN order_count;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT get_user_order_count(1);

-- Function returning table
CREATE OR REPLACE FUNCTION get_high_value_customers(min_spent DECIMAL)
RETURNS TABLE(
    user_id INTEGER,
    username VARCHAR,
    total_spent DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        u.id,
        u.username,
        SUM(o.total) AS total
    FROM users u
    INNER JOIN orders o ON u.id = o.user_id
    GROUP BY u.id, u.username
    HAVING SUM(o.total) >= min_spent;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT * FROM get_high_value_customers(1000.00);

-- Function with exception handling
CREATE OR REPLACE FUNCTION safe_divide(a DECIMAL, b DECIMAL)
RETURNS DECIMAL AS $$
BEGIN
    IF b = 0 THEN
        RAISE EXCEPTION 'Division by zero';
    END IF;
    RETURN a / b;
EXCEPTION
    WHEN division_by_zero THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

### Procedures

```sql
-- Stored procedure
CREATE OR REPLACE PROCEDURE process_order(
    p_user_id INTEGER,
    p_total DECIMAL
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_order_id INTEGER;
BEGIN
    -- Insert order
    INSERT INTO orders (user_id, total, status)
    VALUES (p_user_id, p_total, 'pending')
    RETURNING id INTO v_order_id;
    
    -- Update user stats
    UPDATE user_stats
    SET order_count = order_count + 1,
        total_spent = total_spent + p_total
    WHERE user_id = p_user_id;
    
    -- Log
    INSERT INTO audit_log (action, details)
    VALUES ('order_created', 'Order ' || v_order_id || ' created');
    
    COMMIT;
END;
$$;

-- Call procedure
CALL process_order(1, 99.99);

-- Procedure with transaction control
CREATE OR REPLACE PROCEDURE transfer_funds(
    from_account INTEGER,
    to_account INTEGER,
    amount DECIMAL
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Start transaction
    BEGIN
        -- Deduct from source
        UPDATE accounts
        SET balance = balance - amount
        WHERE id = from_account;
        
        -- Check if sufficient funds
        IF NOT FOUND OR (SELECT balance FROM accounts WHERE id = from_account) < 0 THEN
            RAISE EXCEPTION 'Insufficient funds';
        END IF;
        
        -- Add to destination
        UPDATE accounts
        SET balance = balance + amount
        WHERE id = to_account;
        
        COMMIT;
    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            RAISE;
    END;
END;
$$;
```

---

## Triggers

```sql
-- Create trigger function
CREATE OR REPLACE FUNCTION update_modified_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
CREATE TRIGGER users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_modified_timestamp();

-- Audit trigger
CREATE OR REPLACE FUNCTION audit_user_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, action, new_data)
        VALUES ('users', 'INSERT', row_to_json(NEW));
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, action, old_data, new_data)
        VALUES ('users', 'UPDATE', row_to_json(OLD), row_to_json(NEW));
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, action, old_data)
        VALUES ('users', 'DELETE', row_to_json(OLD));
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_users
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW
EXECUTE FUNCTION audit_user_changes();
```

---

## pgvector - Vector Similarity Search

### Installation

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;
```

### Create Table with Vector Column

```sql
-- Create table for embeddings
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI ada-002 dimension
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Or use HNSW index (better for high-dimensional vectors)
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
```

### Insert Vectors

```sql
-- Insert document with embedding
INSERT INTO documents (title, content, embedding)
VALUES (
    'Machine Learning Basics',
    'Introduction to ML concepts...',
    '[0.1, 0.2, 0.3, ...]'::vector  -- 1536 dimensions
);

-- Insert from Python
-- import psycopg2
-- from openai import OpenAI
-- 
-- client = OpenAI()
-- response = client.embeddings.create(
--     model="text-embedding-ada-002",
--     input="Your text here"
-- )
-- embedding = response.data[0].embedding
-- 
-- cursor.execute(
--     "INSERT INTO documents (title, content, embedding) VALUES (%s, %s, %s)",
--     (title, content, embedding)
-- )
```

### Similarity Search

```sql
-- Cosine similarity (most common)
SELECT 
    id,
    title,
    1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- L2 distance (Euclidean)
SELECT 
    id,
    title,
    embedding <-> '[0.1, 0.2, ...]'::vector AS distance
FROM documents
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- Inner product
SELECT 
    id,
    title,
    (embedding <#> '[0.1, 0.2, ...]'::vector) * -1 AS similarity
FROM documents
ORDER BY embedding <#> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- With filters
SELECT 
    id,
    title,
    1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM documents
WHERE created_at > '2024-01-01'
  AND similarity > 0.8
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

### Python Integration

```python
import psycopg2
from openai import OpenAI
import numpy as np

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="vectordb",
    user="vectoruser",
    password="password"
)
cursor = conn.cursor()

# Initialize OpenAI
client = OpenAI(api_key="your-api-key")

def get_embedding(text):
    """Get embedding from OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def insert_document(title, content):
    """Insert document with embedding"""
    embedding = get_embedding(content)
    cursor.execute(
        """
        INSERT INTO documents (title, content, embedding)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (title, content, embedding)
    )
    doc_id = cursor.fetchone()[0]
    conn.commit()
    return doc_id

def search_similar(query, limit=10):
    """Search for similar documents"""
    query_embedding = get_embedding(query)
    cursor.execute(
        """
        SELECT 
            id,
            title,
            content,
            1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, limit)
    )
    return cursor.fetchall()

# Usage
doc_id = insert_document(
    "Introduction to Neural Networks",
    "Neural networks are computing systems inspired by biological neural networks..."
)

results = search_similar("What are neural networks?")
for id, title, content, similarity in results:
    print(f"{title} (similarity: {similarity:.4f})")
```

---

## Advanced Features

### JSON/JSONB

```sql
-- Create table with JSONB
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200),
    attributes JSONB
);

-- Insert JSON data
INSERT INTO products (name, attributes) VALUES
    ('Laptop', '{"brand": "Dell", "ram": "16GB", "storage": "512GB SSD"}'),
    ('Phone', '{"brand": "Apple", "model": "iPhone 14", "color": "black"}');

-- Query JSON
SELECT * FROM products WHERE attributes->>'brand' = 'Dell';
SELECT * FROM products WHERE attributes->'ram' = '"16GB"';

-- JSON operators
SELECT attributes->'brand' FROM products;  -- Get JSON object
SELECT attributes->>'brand' FROM products;  -- Get text
SELECT attributes#>'{specs,ram}' FROM products;  -- Get nested

-- JSON functions
SELECT jsonb_array_elements(attributes->'tags') FROM products;
SELECT jsonb_object_keys(attributes) FROM products;

-- Index on JSON
CREATE INDEX idx_brand ON products ((attributes->>'brand'));
```

### Full-Text Search

```sql
-- Create table
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    search_vector tsvector
);

-- Generate search vector
UPDATE articles
SET search_vector = 
    setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
    setweight(to_tsvector('english', COALESCE(content, '')), 'B');

-- Create GIN index
CREATE INDEX idx_search ON articles USING GIN(search_vector);

-- Search
SELECT title, ts_rank(search_vector, query) AS rank
FROM articles, to_tsquery('english', 'postgresql & database') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- Auto-update trigger
CREATE TRIGGER articles_search_update
BEFORE INSERT OR UPDATE ON articles
FOR EACH ROW EXECUTE FUNCTION
tsvector_update_trigger(search_vector, 'pg_catalog.english', title, content);
```

### Partitioning

```sql
-- Create partitioned table
CREATE TABLE orders (
    id SERIAL,
    user_id INTEGER,
    total DECIMAL(10, 2),
    created_at TIMESTAMP
) PARTITION BY RANGE (created_at);

-- Create partitions
CREATE TABLE orders_2023 PARTITION OF orders
FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE orders_2024 PARTITION OF orders
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Insert automatically routes to correct partition
INSERT INTO orders (user_id, total, created_at)
VALUES (1, 99.99, '2024-06-15');
```

---

## Performance Tips

```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- Create indexes
CREATE INDEX idx_email ON users(email);
CREATE INDEX idx_created_at ON users(created_at);
CREATE INDEX idx_composite ON users(email, created_at);

-- Partial index
CREATE INDEX idx_active_users ON users(email) WHERE active = true;

-- Vacuum and analyze
VACUUM ANALYZE users;

-- Show slow queries
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

---