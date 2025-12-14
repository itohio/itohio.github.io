---
title: "Docker Compose Commands & Setup"
date: 2024-12-12
draft: false
category: "docker"
tags: ["docker-knowhow", "docker-compose", "containers", "devops"]
---


Docker Compose commands and configuration patterns for multi-container applications.

---

## Basic Commands

```bash
# Start services
docker-compose up
docker-compose up -d  # Detached mode

# Start specific services
docker-compose up app db

# Build and start
docker-compose up --build

# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers, volumes, images
docker-compose down -v --rmi all

# View logs
docker-compose logs
docker-compose logs -f  # Follow
docker-compose logs -f app  # Specific service

# List containers
docker-compose ps

# Execute command in service
docker-compose exec app bash
docker-compose exec db psql -U postgres

# Run one-off command
docker-compose run app npm test
docker-compose run --rm app npm test  # Remove after

# View config
docker-compose config

# Validate config
docker-compose config --quiet

# Build services
docker-compose build
docker-compose build --no-cache

# Pull images
docker-compose pull

# Restart services
docker-compose restart
docker-compose restart app

# Pause services
docker-compose pause

# Unpause services
docker-compose unpause

# View service logs
docker-compose logs --tail=100 app
```

---

## Complete Examples

### Full-Stack Web App

```yaml
version: '3.8'

services:
  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: production
    image: myapp-frontend:latest
    container_name: frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8080
    volumes:
      - ./frontend:/app
      - /app/node_modules
    networks:
      - app-network
    depends_on:
      - backend
    restart: unless-stopped

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    image: myapp-backend:latest
    container_name: backend
    ports:
      - "8080:8080"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgres://postgres:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=${JWT_SECRET}
    env_file:
      - .env
    volumes:
      - ./backend:/app
      - /app/node_modules
      - uploads:/app/uploads
    networks:
      - app-network
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # PostgreSQL Database
  db:
    image: postgres:15-alpine
    container_name: postgres
    environment:
      POSTGRES_DB: mydb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - app-network
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: redis
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    networks:
      - app-network
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - app-network
    depends_on:
      - frontend
      - backend
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  uploads:

networks:
  app-network:
    driver: bridge
```

### Microservices Architecture

```yaml
version: '3.8'

services:
  # API Gateway
  gateway:
    build: ./gateway
    ports:
      - "8080:8080"
    environment:
      - AUTH_SERVICE_URL=http://auth:8081
      - USER_SERVICE_URL=http://user:8082
      - ORDER_SERVICE_URL=http://order:8083
    networks:
      - microservices
    depends_on:
      - auth
      - user
      - order

  # Auth Service
  auth:
    build: ./services/auth
    environment:
      - DATABASE_URL=postgres://postgres:password@auth-db:5432/auth
      - JWT_SECRET=${JWT_SECRET}
    networks:
      - microservices
      - auth-network
    depends_on:
      - auth-db

  auth-db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: auth
      POSTGRES_PASSWORD: password
    volumes:
      - auth-db-data:/var/lib/postgresql/data
    networks:
      - auth-network

  # User Service
  user:
    build: ./services/user
    environment:
      - DATABASE_URL=postgres://postgres:password@user-db:5432/users
    networks:
      - microservices
      - user-network
    depends_on:
      - user-db

  user-db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: users
      POSTGRES_PASSWORD: password
    volumes:
      - user-db-data:/var/lib/postgresql/data
    networks:
      - user-network

  # Order Service
  order:
    build: ./services/order
    environment:
      - DATABASE_URL=postgres://postgres:password@order-db:5432/orders
      - RABBITMQ_URL=amqp://rabbitmq:5672
    networks:
      - microservices
      - order-network
    depends_on:
      - order-db
      - rabbitmq

  order-db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: orders
      POSTGRES_PASSWORD: password
    volumes:
      - order-db-data:/var/lib/postgresql/data
    networks:
      - order-network

  # Message Queue
  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: password
    volumes:
      - rabbitmq-data:/var/lib/rabbitmq
    networks:
      - microservices

volumes:
  auth-db-data:
  user-db-data:
  order-db-data:
  rabbitmq-data:

networks:
  microservices:
    driver: bridge
  auth-network:
    driver: bridge
  user-network:
    driver: bridge
  order-network:
    driver: bridge
```

### Development Environment

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgres://postgres:password@db:5432/dev
    command: npm run dev
    networks:
      - dev-network
    depends_on:
      - db
      - redis

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: dev
      POSTGRES_PASSWORD: password
    volumes:
      - postgres-dev-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - dev-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - dev-network

  # Database admin tool
  adminer:
    image: adminer
    ports:
      - "8080:8080"
    networks:
      - dev-network
    depends_on:
      - db

  # Redis admin tool
  redis-commander:
    image: rediscommander/redis-commander:latest
    environment:
      - REDIS_HOSTS=local:redis:6379
    ports:
      - "8081:8081"
    networks:
      - dev-network
    depends_on:
      - redis

volumes:
  postgres-dev-data:

networks:
  dev-network:
    driver: bridge
```

---

## Advanced Patterns

### Override Files

**docker-compose.yml** (base):
```yaml
version: '3.8'
services:
  app:
    image: myapp:latest
    environment:
      - NODE_ENV=production
```

**docker-compose.override.yml** (development):
```yaml
version: '3.8'
services:
  app:
    build: .
    volumes:
      - .:/app
    environment:
      - NODE_ENV=development
    command: npm run dev
```

**docker-compose.prod.yml** (production):
```yaml
version: '3.8'
services:
  app:
    image: myapp:${VERSION}
    restart: always
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

```bash
# Use override (default)
docker-compose up

# Use specific file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up

# Multiple override files
docker-compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.local.yml up
```

### Environment Variables

**.env file:**
```bash
# Database
POSTGRES_DB=mydb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secretpassword

# Application
NODE_ENV=production
JWT_SECRET=your-secret-key
API_PORT=8080

# Redis
REDIS_PASSWORD=redispassword

# Version
VERSION=1.0.0
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  app:
    image: myapp:${VERSION:-latest}
    environment:
      - NODE_ENV=${NODE_ENV}
      - JWT_SECRET=${JWT_SECRET}
    ports:
      - "${API_PORT}:8080"
```

### Health Checks

```yaml
services:
  app:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
  
  db:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  redis:
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
```

### Resource Limits

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '0.50'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

### Logging

```yaml
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Makefile Integration

```makefile
.PHONY: up down build logs shell test clean

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f

shell:
	docker-compose exec app bash

test:
	docker-compose run --rm app npm test

clean:
	docker-compose down -v --rmi all
	docker system prune -f
```

---

## Troubleshooting

```bash
# View service logs
docker-compose logs -f app

# Rebuild service
docker-compose up -d --build app

# Recreate containers
docker-compose up -d --force-recreate

# Remove volumes
docker-compose down -v

# Check service health
docker-compose ps

# Validate compose file
docker-compose config

# View service IPs
docker-compose exec app cat /etc/hosts
```

---

## Notes

**Best Practices:**
- ✅ Use version control for docker-compose.yml
- ✅ Use .env files for secrets (don't commit)
- ✅ Use health checks for critical services
- ✅ Set restart policies for production
- ✅ Use named volumes for data persistence
- ✅ Use networks to isolate services
- ✅ Set resource limits in production
- ✅ Use depends_on with health checks
- ✅ Use override files for different environments

---