---
title: "React Essentials - From Setup to Todo App"
date: 2024-12-12
draft: false
category: "frontend"
tags: ["frontend-knowhow", "react", "javascript", "typescript"]
---


Complete React guide from project setup to building a functional Todo application. Includes modern hooks, TypeScript, and best practices.

---

## Docker Setup

### Dockerfile

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  react-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: react-app
    ports:
      - "3000:80"
    restart: unless-stopped

  # Development with hot reload
  react-dev:
    image: node:18-alpine
    container_name: react-dev
    working_dir: /app
    volumes:
      - ./:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    command: npm run dev
    environment:
      - VITE_API_URL=http://localhost:8080
    restart: unless-stopped
```

---

## Project Setup

### Create React App (CRA) - Legacy

```bash
# Create new React app
npx create-react-app my-app
cd my-app
npm start

# With TypeScript
npx create-react-app my-app --template typescript
```

**⚠️ Gotcha:** CRA is no longer recommended by React team. Use Vite instead.

---

### Vite (Recommended)

```bash
# Create new React app with Vite
npm create vite@latest my-app -- --template react
cd my-app
npm install
npm run dev

# With TypeScript
npm create vite@latest my-app -- --template react-ts
```

**Node.js Version Requirements:**
- Vite 5.x: Node.js 18+ or 20+
- Vite 4.x: Node.js 14.18+ or 16+

**Check your Node version:**
```bash
node --version
npm --version
```

**⚠️ Gotcha:** If you get "ERR_OSSL_EVP_UNSUPPORTED" on Node 17+, use Node 18+ or set:
```bash
# Windows PowerShell
$env:NODE_OPTIONS="--openssl-legacy-provider"

# Linux/Mac
export NODE_OPTIONS=--openssl-legacy-provider
```

---

### Next.js (Full-Stack Framework)

```bash
npx create-next-app@latest my-app
cd my-app
npm run dev
```

**Features:**
- Server-side rendering (SSR)
- Static site generation (SSG)
- API routes
- File-based routing

---

## Basic React Concepts

### Functional Components

```tsx
// Basic component
function Welcome() {
  return <h1>Hello, World!</h1>;
}

// Component with props
interface GreetingProps {
  name: string;
  age?: number; // Optional
}

function Greeting({ name, age }: GreetingProps) {
  return (
    <div>
      <h1>Hello, {name}!</h1>
      {age && <p>Age: {age}</p>}
    </div>
  );
}

// Usage
<Greeting name="John" age={30} />
```

---

### Hooks

#### useState

```tsx
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <button onClick={() => setCount(count - 1)}>Decrement</button>
      <button onClick={() => setCount(0)}>Reset</button>
    </div>
  );
}
```

**⚠️ Gotcha:** State updates are asynchronous!

```tsx
// ❌ Wrong - won't work as expected
setCount(count + 1);
setCount(count + 1); // Still adds only 1

// ✅ Correct - use functional update
setCount(prev => prev + 1);
setCount(prev => prev + 1); // Adds 2
```

---

#### useEffect

```tsx
import { useState, useEffect } from 'react';

function DataFetcher() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Fetch data when component mounts
    fetch('https://api.example.com/data')
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []); // Empty dependency array = run once on mount
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  
  return <div>{JSON.stringify(data)}</div>;
}
```

**Dependency Array:**
```tsx
useEffect(() => {
  // Runs on every render
});

useEffect(() => {
  // Runs once on mount
}, []);

useEffect(() => {
  // Runs when count changes
}, [count]);

useEffect(() => {
  // Cleanup function
  const timer = setInterval(() => console.log('tick'), 1000);
  
  return () => clearInterval(timer); // Cleanup on unmount
}, []);
```

**⚠️ Gotcha:** Missing dependencies cause stale closures!

```tsx
// ❌ Wrong - count is stale
useEffect(() => {
  setInterval(() => {
    setCount(count + 1); // Always uses initial count
  }, 1000);
}, []);

// ✅ Correct
useEffect(() => {
  const timer = setInterval(() => {
    setCount(prev => prev + 1); // Functional update
  }, 1000);
  
  return () => clearInterval(timer);
}, []);
```

---

#### useContext

```tsx
import { createContext, useContext, useState } from 'react';

// Create context
const ThemeContext = createContext<'light' | 'dark'>('light');

// Provider component
function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  
  return (
    <ThemeContext.Provider value={theme}>
      <Toolbar />
      <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
        Toggle Theme
      </button>
    </ThemeContext.Provider>
  );
}

// Consumer component
function Toolbar() {
  const theme = useContext(ThemeContext);
  
  return (
    <div style={{ background: theme === 'light' ? '#fff' : '#333' }}>
      Current theme: {theme}
    </div>
  );
}
```

---

#### useRef

```tsx
import { useRef, useEffect } from 'react';

function TextInput() {
  const inputRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    // Focus input on mount
    inputRef.current?.focus();
  }, []);
  
  return <input ref={inputRef} type="text" />;
}

// Store mutable value without re-rendering
function Timer() {
  const countRef = useRef(0);
  
  const handleClick = () => {
    countRef.current += 1;
    console.log(countRef.current); // Updates without re-render
  };
  
  return <button onClick={handleClick}>Click me</button>;
}
```

---

#### useMemo & useCallback

```tsx
import { useMemo, useCallback, useState } from 'react';

function ExpensiveComponent() {
  const [count, setCount] = useState(0);
  const [input, setInput] = useState('');
  
  // Memoize expensive calculation
  const expensiveValue = useMemo(() => {
    console.log('Computing expensive value...');
    return count * 2;
  }, [count]); // Only recompute when count changes
  
  // Memoize callback function
  const handleClick = useCallback(() => {
    console.log('Button clicked');
    setCount(prev => prev + 1);
  }, []); // Function reference stays the same
  
  return (
    <div>
      <p>Expensive value: {expensiveValue}</p>
      <button onClick={handleClick}>Increment</button>
      <input value={input} onChange={e => setInput(e.target.value)} />
    </div>
  );
}
```

**⚠️ Gotcha:** Don't overuse memoization! Only use when you have actual performance issues.

---

## Complete Todo App Example

```tsx
// src/App.tsx
import { useState } from 'react';
import './App.css';

interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

function App() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [input, setInput] = useState('');
  const [filter, setFilter] = useState<'all' | 'active' | 'completed'>('all');
  
  // Add todo
  const addTodo = () => {
    if (input.trim() === '') return;
    
    const newTodo: Todo = {
      id: Date.now(),
      text: input,
      completed: false,
    };
    
    setTodos([...todos, newTodo]);
    setInput('');
  };
  
  // Toggle todo completion
  const toggleTodo = (id: number) => {
    setTodos(todos.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };
  
  // Delete todo
  const deleteTodo = (id: number) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };
  
  // Edit todo
  const editTodo = (id: number, newText: string) => {
    setTodos(todos.map(todo =>
      todo.id === id ? { ...todo, text: newText } : todo
    ));
  };
  
  // Filter todos
  const filteredTodos = todos.filter(todo => {
    if (filter === 'active') return !todo.completed;
    if (filter === 'completed') return todo.completed;
    return true;
  });
  
  // Clear completed
  const clearCompleted = () => {
    setTodos(todos.filter(todo => !todo.completed));
  };
  
  const activeCount = todos.filter(todo => !todo.completed).length;
  
  return (
    <div className="app">
      <h1>Todo App</h1>
      
      {/* Input */}
      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyPress={e => e.key === 'Enter' && addTodo()}
          placeholder="What needs to be done?"
        />
        <button onClick={addTodo}>Add</button>
      </div>
      
      {/* Filter buttons */}
      <div className="filters">
        <button
          className={filter === 'all' ? 'active' : ''}
          onClick={() => setFilter('all')}
        >
          All ({todos.length})
        </button>
        <button
          className={filter === 'active' ? 'active' : ''}
          onClick={() => setFilter('active')}
        >
          Active ({activeCount})
        </button>
        <button
          className={filter === 'completed' ? 'active' : ''}
          onClick={() => setFilter('completed')}
        >
          Completed ({todos.length - activeCount})
        </button>
      </div>
      
      {/* Todo list */}
      <ul className="todo-list">
        {filteredTodos.map(todo => (
          <TodoItem
            key={todo.id}
            todo={todo}
            onToggle={toggleTodo}
            onDelete={deleteTodo}
            onEdit={editTodo}
          />
        ))}
      </ul>
      
      {/* Footer */}
      {todos.length > 0 && (
        <div className="footer">
          <span>{activeCount} item{activeCount !== 1 ? 's' : ''} left</span>
          <button onClick={clearCompleted}>Clear completed</button>
        </div>
      )}
    </div>
  );
}

// TodoItem component
interface TodoItemProps {
  todo: Todo;
  onToggle: (id: number) => void;
  onDelete: (id: number) => void;
  onEdit: (id: number, text: string) => void;
}

function TodoItem({ todo, onToggle, onDelete, onEdit }: TodoItemProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(todo.text);
  
  const handleEdit = () => {
    if (editText.trim() === '') {
      onDelete(todo.id);
    } else {
      onEdit(todo.id, editText);
    }
    setIsEditing(false);
  };
  
  if (isEditing) {
    return (
      <li className="todo-item editing">
        <input
          type="text"
          value={editText}
          onChange={e => setEditText(e.target.value)}
          onBlur={handleEdit}
          onKeyPress={e => e.key === 'Enter' && handleEdit()}
          autoFocus
        />
      </li>
    );
  }
  
  return (
    <li className={`todo-item ${todo.completed ? 'completed' : ''}`}>
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={() => onToggle(todo.id)}
      />
      <span onDoubleClick={() => setIsEditing(true)}>{todo.text}</span>
      <button onClick={() => onDelete(todo.id)}>×</button>
    </li>
  );
}

export default App;
```

### CSS (App.css)

```css
.app {
  max-width: 600px;
  margin: 50px auto;
  padding: 20px;
  font-family: Arial, sans-serif;
}

h1 {
  text-align: center;
  color: #333;
}

.input-container {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.input-container input {
  flex: 1;
  padding: 10px;
  font-size: 16px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.input-container button {
  padding: 10px 20px;
  font-size: 16px;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.input-container button:hover {
  background: #45a049;
}

.filters {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.filters button {
  flex: 1;
  padding: 8px;
  background: #f0f0f0;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
}

.filters button.active {
  background: #2196F3;
  color: white;
}

.todo-list {
  list-style: none;
  padding: 0;
}

.todo-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-bottom: 10px;
}

.todo-item.completed span {
  text-decoration: line-through;
  color: #999;
}

.todo-item input[type="checkbox"] {
  width: 20px;
  height: 20px;
  cursor: pointer;
}

.todo-item span {
  flex: 1;
  cursor: pointer;
}

.todo-item button {
  width: 30px;
  height: 30px;
  background: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 20px;
}

.todo-item button:hover {
  background: #da190b;
}

.todo-item.editing input {
  flex: 1;
  padding: 8px;
  font-size: 16px;
  border: 1px solid #2196F3;
  border-radius: 4px;
}

.footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #ddd;
}

.footer button {
  padding: 8px 16px;
  background: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.footer button:hover {
  background: #da190b;
}
```

---

## Common Gotchas

### 1. Key Prop in Lists

```tsx
// ❌ Wrong - using index as key
{todos.map((todo, index) => (
  <TodoItem key={index} todo={todo} />
))}

// ✅ Correct - using unique ID
{todos.map(todo => (
  <TodoItem key={todo.id} todo={todo} />
))}
```

### 2. Event Handlers

```tsx
// ❌ Wrong - calls function immediately
<button onClick={handleClick()}>Click</button>

// ✅ Correct - passes function reference
<button onClick={handleClick}>Click</button>

// ✅ Correct - with arguments
<button onClick={() => handleClick(id)}>Click</button>
```

### 3. Conditional Rendering

```tsx
// ✅ Ternary operator
{isLoading ? <Spinner /> : <Content />}

// ✅ Logical AND
{error && <ErrorMessage error={error} />}

// ✅ Nullish coalescing
{data?.items?.length ?? 0}

// ❌ Wrong - renders "0" or "false"
{items.length && <List items={items} />}

// ✅ Correct
{items.length > 0 && <List items={items} />}
```

### 4. Forms and Controlled Inputs

```tsx
// ❌ Uncontrolled (avoid)
<input type="text" />

// ✅ Controlled
const [value, setValue] = useState('');
<input type="text" value={value} onChange={e => setValue(e.target.value)} />
```

---