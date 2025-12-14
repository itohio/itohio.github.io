---
title: "Docker Commands Cheatsheet"
date: 2024-12-12
draft: false
category: "docker"
tags: ["docker-knowhow", "docker", "containers", "devops"]
---


Essential Docker commands for daily container management. Quick reference for building, running, and debugging containers.

---

## Images

### Build

```bash
# Build image from Dockerfile
docker build -t myapp:latest .
docker build -t myapp:v1.0.0 .

# Build with build args
docker build --build-arg NODE_ENV=production -t myapp:latest .

# Build with specific Dockerfile
docker build -f Dockerfile.prod -t myapp:prod .

# Build without cache
docker build --no-cache -t myapp:latest .

# Build with target stage (multi-stage)
docker build --target production -t myapp:prod .

# Build for specific platform
docker build --platform linux/amd64 -t myapp:latest .
```

### List & Inspect

```bash
# List images
docker images
docker image ls

# List all images (including intermediate)
docker images -a

# Filter images
docker images --filter "dangling=true"
docker images --filter "reference=myapp:*"

# Inspect image
docker inspect myapp:latest

# View image history
docker history myapp:latest

# View image layers
docker image inspect myapp:latest | jq '.[0].RootFS.Layers'
```

### Pull & Push

```bash
# Pull image
docker pull nginx:latest
docker pull nginx:1.25-alpine

# Pull from specific registry
docker pull ghcr.io/user/repo:tag

# Push image
docker push myapp:latest

# Tag image
docker tag myapp:latest myregistry.com/myapp:latest
docker tag myapp:latest myapp:v1.0.0

# Login to registry
docker login
docker login ghcr.io -u username
docker login registry.example.com
```

### Remove

```bash
# Remove image
docker rmi myapp:latest
docker image rm myapp:latest

# Remove multiple images
docker rmi image1 image2 image3

# Remove dangling images
docker image prune

# Remove all unused images
docker image prune -a

# Force remove
docker rmi -f myapp:latest
```

---

## Containers

### Run

```bash
# Run container
docker run nginx

# Run with name
docker run --name mynginx nginx

# Run in background (detached)
docker run -d nginx

# Run with port mapping
docker run -p 8080:80 nginx
docker run -p 127.0.0.1:8080:80 nginx  # Bind to localhost only

# Run with environment variables
docker run -e NODE_ENV=production myapp
docker run --env-file .env myapp

# Run with volume mount
docker run -v /host/path:/container/path myapp
docker run -v myvolume:/data myapp  # Named volume

# Run with bind mount (current directory)
docker run -v $(pwd):/app myapp  # Linux/Mac
docker run -v ${PWD}:/app myapp  # PowerShell

# Run with network
docker run --network mynetwork myapp

# Run with restart policy
docker run --restart unless-stopped myapp
docker run --restart always myapp

# Run with resource limits
docker run --memory="512m" --cpus="1.5" myapp

# Run with user
docker run --user 1000:1000 myapp

# Run with working directory
docker run -w /app myapp

# Run interactive with TTY
docker run -it ubuntu bash

# Run and remove after exit
docker run --rm myapp

# Run with all options combined
docker run -d \
  --name myapp \
  -p 8080:80 \
  -e NODE_ENV=production \
  -v $(pwd):/app \
  --network mynetwork \
  --restart unless-stopped \
  myapp:latest
```

### List & Inspect

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# List with specific format
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"

# Filter containers
docker ps --filter "status=running"
docker ps --filter "name=myapp"

# Inspect container
docker inspect mycontainer

# View container logs
docker logs mycontainer
docker logs -f mycontainer  # Follow
docker logs --tail 100 mycontainer  # Last 100 lines
docker logs --since 10m mycontainer  # Last 10 minutes

# View container stats
docker stats
docker stats mycontainer

# View container processes
docker top mycontainer

# View container port mappings
docker port mycontainer
```

### Execute & Interact

```bash
# Execute command in running container
docker exec mycontainer ls -la

# Execute interactive shell
docker exec -it mycontainer bash
docker exec -it mycontainer sh  # Alpine

# Execute as specific user
docker exec -u root -it mycontainer bash

# Execute with environment variable
docker exec -e VAR=value mycontainer env

# Attach to running container
docker attach mycontainer

# Copy files to/from container
docker cp mycontainer:/path/to/file ./local/path
docker cp ./local/file mycontainer:/path/to/
```

### Control

```bash
# Start container
docker start mycontainer

# Stop container
docker stop mycontainer

# Stop with timeout
docker stop -t 30 mycontainer

# Restart container
docker restart mycontainer

# Pause container
docker pause mycontainer

# Unpause container
docker unpause mycontainer

# Kill container (force stop)
docker kill mycontainer

# Rename container
docker rename oldname newname

# Update container config
docker update --restart=always mycontainer
docker update --memory="1g" mycontainer
```

### Remove

```bash
# Remove container
docker rm mycontainer

# Force remove running container
docker rm -f mycontainer

# Remove multiple containers
docker rm container1 container2 container3

# Remove all stopped containers
docker container prune

# Remove all containers (including running)
docker rm -f $(docker ps -aq)
```

---

## Volumes

```bash
# Create volume
docker volume create myvolume

# List volumes
docker volume ls

# Inspect volume
docker volume inspect myvolume

# Remove volume
docker volume rm myvolume

# Remove all unused volumes
docker volume prune

# Remove all volumes
docker volume rm $(docker volume ls -q)

# Backup volume
docker run --rm -v myvolume:/source -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz -C /source .

# Restore volume
docker run --rm -v myvolume:/target -v $(pwd):/backup alpine tar xzf /backup/backup.tar.gz -C /target
```

---

## Networks

```bash
# Create network
docker network create mynetwork

# Create network with subnet
docker network create --subnet=172.18.0.0/16 mynetwork

# List networks
docker network ls

# Inspect network
docker network inspect mynetwork

# Connect container to network
docker network connect mynetwork mycontainer

# Disconnect container from network
docker network disconnect mynetwork mycontainer

# Remove network
docker network rm mynetwork

# Remove all unused networks
docker network prune
```

---

## System

```bash
# View Docker info
docker info

# View Docker version
docker version

# View disk usage
docker system df

# Clean up everything
docker system prune

# Clean up everything including volumes
docker system prune -a --volumes

# View events
docker events

# View events with filter
docker events --filter "type=container"
```

---

## Dockerfile Best Practices

```dockerfile
# Use specific base image version
FROM node:18-alpine AS base

# Set working directory
WORKDIR /app

# Copy package files first (better caching)
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY . .

# Use non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001
USER nodejs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node healthcheck.js

# Start application
CMD ["node", "server.js"]
```

### Multi-Stage Build

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine AS production
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY --from=builder /app/dist ./dist
USER node
EXPOSE 3000
CMD ["node", "dist/server.js"]
```

### Go Multi-Stage Build

```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o main .

# Production stage
FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/main .
EXPOSE 8080
CMD ["./main"]
```

---

## Docker Compose Quick Reference

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    image: myapp:latest
    container_name: myapp
    ports:
      - "8080:80"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgres://db:5432/mydb
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - app-logs:/var/log
    networks:
      - mynetwork
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15-alpine
    container_name: postgres
    environment:
      POSTGRES_DB: mydb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - mynetwork
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: redis
    networks:
      - mynetwork
    restart: unless-stopped

volumes:
  postgres-data:
  app-logs:

networks:
  mynetwork:
    driver: bridge
```

---

## Common Patterns

### Development with Hot Reload

```bash
# Node.js with nodemon
docker run -d \
  -v $(pwd):/app \
  -v /app/node_modules \
  -p 3000:3000 \
  -e NODE_ENV=development \
  myapp npm run dev
```

### Database Container

```bash
# PostgreSQL
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=mydb \
  -v postgres-data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15-alpine

# MySQL
docker run -d \
  --name mysql \
  -e MYSQL_ROOT_PASSWORD=password \
  -e MYSQL_DATABASE=mydb \
  -v mysql-data:/var/lib/mysql \
  -p 3306:3306 \
  mysql:8

# MongoDB
docker run -d \
  --name mongo \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  -v mongo-data:/data/db \
  -p 27017:27017 \
  mongo:7

# Redis
docker run -d \
  --name redis \
  -v redis-data:/data \
  -p 6379:6379 \
  redis:7-alpine redis-server --appendonly yes
```

### Debugging

```bash
# View container logs
docker logs -f --tail 100 mycontainer

# Execute shell in container
docker exec -it mycontainer sh

# View container processes
docker top mycontainer

# View container resource usage
docker stats mycontainer

# Inspect container configuration
docker inspect mycontainer | jq '.[0].Config'

# View container network settings
docker inspect mycontainer | jq '.[0].NetworkSettings'

# Copy logs from container
docker cp mycontainer:/var/log/app.log ./app.log
```

---

## Troubleshooting

```bash
# Container won't start
docker logs mycontainer
docker inspect mycontainer

# Port already in use
docker ps -a | grep 8080
lsof -i :8080  # Find process using port

# Permission denied
# Run as root or add user to docker group
sudo usermod -aG docker $USER

# Out of disk space
docker system df
docker system prune -a --volumes

# Container keeps restarting
docker logs --tail 50 mycontainer
docker inspect mycontainer | jq '.[0].State'

# Network issues
docker network inspect bridge
docker exec mycontainer ping google.com

# DNS issues
docker run --dns 8.8.8.8 myapp

# Can't connect to Docker daemon
# Check if Docker is running
sudo systemctl status docker
sudo systemctl start docker
```

---

## Notes

**Best Practices:**
- ✅ Use specific image tags, not `latest`
- ✅ Use multi-stage builds to reduce image size
- ✅ Run containers as non-root user
- ✅ Use `.dockerignore` to exclude unnecessary files
- ✅ Minimize layers by combining RUN commands
- ✅ Use health checks for production containers
- ✅ Set resource limits (memory, CPU)
- ✅ Use named volumes for persistent data
- ✅ Use Docker Compose for multi-container apps

**Security:**
- ✅ Scan images for vulnerabilities: `docker scan myapp:latest`
- ✅ Don't store secrets in images
- ✅ Use secrets management (Docker secrets, env files)
- ✅ Keep base images updated
- ✅ Use official images when possible

---