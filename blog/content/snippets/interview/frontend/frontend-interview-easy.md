---
title: "Frontend Interview Questions - Easy"
date: 2025-12-14
tags: ["frontend", "interview", "easy", "react", "vue", "javascript"]
---

Easy-level frontend interview questions covering HTML, CSS, JavaScript, React, and Vue fundamentals.

## Q1: What is the DOM and how does it work?

**Answer**:

```mermaid
graph TB
    A[HTML Document] --> B[Browser Parses]
    B --> C[DOM Tree]
    
    C --> D[Document]
    D --> E[html]
    E --> F[head]
    E --> G[body]
    G --> H[div]
    H --> I[p]
    H --> J[span]
    
    style C fill:#FFD700
    style D fill:#87CEEB
```

**DOM (Document Object Model)**: Tree-like representation of HTML document that JavaScript can manipulate.

### DOM Manipulation

```javascript
// Select elements
const element = document.getElementById('myId');
const elements = document.querySelectorAll('.myClass');

// Modify content
element.textContent = 'New text';
element.innerHTML = '<strong>Bold text</strong>';

// Modify attributes
element.setAttribute('class', 'new-class');
element.style.color = 'red';

// Create and append
const newDiv = document.createElement('div');
newDiv.textContent = 'Hello';
document.body.appendChild(newDiv);
```

---

## Q2: Explain the CSS Box Model.

**Answer**:

```mermaid
graph TB
    A[Box Model] --> B[Content<br/>Actual content]
    A --> C[Padding<br/>Space around content]
    A --> D[Border<br/>Edge of box]
    A --> E[Margin<br/>Space outside border]
    
    style A fill:#FFD700
```

### Visual Representation

```mermaid
graph TB
    subgraph Margin
        subgraph Border
            subgraph Padding
                A[Content<br/>Width × Height]
            end
        end
    end
    
    style A fill:#87CEEB
```

```css
.box {
  width: 200px;           /* Content width */
  height: 100px;          /* Content height */
  padding: 20px;          /* Space inside border */
  border: 5px solid black; /* Border */
  margin: 10px;           /* Space outside border */
}

/* Total width = 200 + 20*2 + 5*2 + 10*2 = 270px */
```

**Box-sizing**:
```css
/* Default: content-box */
.box { box-sizing: content-box; }
/* Width = content only */

/* Better: border-box */
.box { box-sizing: border-box; }
/* Width = content + padding + border */
```

---

## Q3: What is event bubbling and capturing?

**Answer**:

```mermaid
graph TB
    A[Event Propagation] --> B[Capturing Phase<br/>Top to target]
    A --> C[Target Phase<br/>At target]
    A --> D[Bubbling Phase<br/>Target to top]
    
    style A fill:#FFD700
```

### Event Flow

```mermaid
graph TB
    A[Window] --> B[Document]
    B --> C[html]
    C --> D[body]
    D --> E[div]
    E --> F[button CLICK]
    
    F -.Bubbling.-> E
    E -.Bubbling.-> D
    D -.Bubbling.-> C
    
    A --Capturing--> B
    B --Capturing--> C
    C --Capturing--> D
    
    style F fill:#FF6B6B
```

```javascript
// Bubbling (default)
element.addEventListener('click', handler);

// Capturing
element.addEventListener('click', handler, true);

// Stop propagation
function handler(event) {
  event.stopPropagation(); // Stop bubbling/capturing
  event.preventDefault();   // Prevent default action
}
```

**Example**:
```html
<div id="parent">
  <button id="child">Click me</button>
</div>

<script>
document.getElementById('parent').addEventListener('click', () => {
  console.log('Parent clicked');
});

document.getElementById('child').addEventListener('click', (e) => {
  console.log('Child clicked');
  // e.stopPropagation(); // Uncomment to stop bubbling
});

// Output: "Child clicked", "Parent clicked"
</script>
```

---

## Q4: What are React components and props?

**Answer**:

```mermaid
graph TB
    A[React Component] --> B[Function Component<br/>Modern approach]
    A --> C[Class Component<br/>Legacy]
    
    B --> D[Receives Props]
    C --> D
    
    D --> E[Returns JSX]
    
    style A fill:#FFD700
    style B fill:#90EE90
```

### Function Component

```javascript
// Function component
function Welcome(props) {
  return <h1>Hello, {props.name}!</h1>;
}

// Arrow function
const Welcome = ({ name }) => {
  return <h1>Hello, {name}!</h1>;
};

// Usage
<Welcome name="Alice" />
```

### Props Flow

```mermaid
graph LR
    A[Parent<br/>Component] -->|Props| B[Child<br/>Component]
    
    B -->|Cannot modify| A
    
    style A fill:#87CEEB
    style B fill:#90EE90
```

```javascript
function App() {
  return (
    <div>
      <Welcome name="Alice" age={25} />
      <Welcome name="Bob" age={30} />
    </div>
  );
}

function Welcome({ name, age }) {
  return (
    <div>
      <h1>Hello, {name}!</h1>
      <p>Age: {age}</p>
    </div>
  );
}
```

**Props are read-only** - components cannot modify their props.

---

## Q5: What is React state and how do you use useState?

**Answer**:

```mermaid
graph TB
    A[State] --> B[Component's<br/>Memory]
    A --> C[Triggers Re-render<br/>When changed]
    A --> D[Managed by<br/>useState Hook]
    
    style A fill:#FFD700
```

### useState Hook

```javascript
import { useState } from 'react';

function Counter() {
  // Declare state variable
  const [count, setCount] = useState(0);
  //     ^state  ^setter    ^initial value
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>
        Increment
      </button>
      <button onClick={() => setCount(count - 1)}>
        Decrement
      </button>
      <button onClick={() => setCount(0)}>
        Reset
      </button>
    </div>
  );
}
```

### State Updates

```mermaid
sequenceDiagram
    participant U as User
    participant C as Component
    participant R as React
    
    U->>C: Click button
    C->>C: Call setCount(count + 1)
    C->>R: State change
    R->>R: Schedule re-render
    R->>C: Re-render with new state
    C->>U: Updated UI
```

```javascript
// Multiple state variables
function Form() {
  const [name, setName] = useState('');
  const [age, setAge] = useState(0);
  const [email, setEmail] = useState('');
  
  return (
    <form>
      <input 
        value={name} 
        onChange={(e) => setName(e.target.value)} 
      />
      <input 
        type="number"
        value={age} 
        onChange={(e) => setAge(Number(e.target.value))} 
      />
      <input 
        type="email"
        value={email} 
        onChange={(e) => setEmail(e.target.value)} 
      />
    </form>
  );
}
```

---

## Q6: What is Vue.js and how does it differ from React?

**Answer**:

```mermaid
graph TB
    A[Vue.js] --> B[Progressive<br/>Framework]
    A --> C[Template Syntax<br/>HTML-like]
    A --> D[Reactive Data<br/>Automatic tracking]
    A --> E[Single File<br/>Components]
    
    style A fill:#FFD700
```

### Vue vs React

```mermaid
graph TB
    subgraph Vue["Vue.js"]
        V1[Template-based]
        V2[Two-way binding]
        V3[Directives v-if, v-for]
        V4[Options API / Composition API]
    end
    
    subgraph React
        R1[JSX]
        R2[One-way data flow]
        R3[JavaScript expressions]
        R4[Hooks]
    end
```

### Vue Component

```vue
<template>
  <div>
    <h1>{{ message }}</h1>
    <button @click="increment">Count: {{ count }}</button>
  </div>
</template>

<script>
export default {
  data() {
    return {
      message: 'Hello Vue!',
      count: 0
    }
  },
  methods: {
    increment() {
      this.count++;
    }
  }
}
</script>

<style scoped>
h1 {
  color: blue;
}
</style>
```

### React Component (Equivalent)

```javascript
import { useState } from 'react';

function MyComponent() {
  const [message] = useState('Hello React!');
  const [count, setCount] = useState(0);
  
  const increment = () => setCount(count + 1);
  
  return (
    <div>
      <h1 style={{ color: 'blue' }}>{message}</h1>
      <button onClick={increment}>Count: {count}</button>
    </div>
  );
}
```

---

## Q7: What are Vue directives?

**Answer**:

```mermaid
graph TB
    A[Vue Directives] --> B[v-if<br/>Conditional rendering]
    A --> C[v-for<br/>List rendering]
    A --> D[v-bind<br/>Bind attributes]
    A --> E[v-on<br/>Event listeners]
    A --> F[v-model<br/>Two-way binding]
    
    style A fill:#FFD700
```

### Common Directives

```vue
<template>
  <!-- v-if: Conditional rendering -->
  <div v-if="isLoggedIn">
    Welcome back!
  </div>
  <div v-else>
    Please log in
  </div>
  
  <!-- v-show: Toggle visibility (CSS display) -->
  <div v-show="isVisible">
    This toggles visibility
  </div>
  
  <!-- v-for: List rendering -->
  <ul>
    <li v-for="item in items" :key="item.id">
      {{ item.name }}
    </li>
  </ul>
  
  <!-- v-bind (shorthand :) -->
  <img :src="imageUrl" :alt="imageAlt" />
  <div :class="{ active: isActive }"></div>
  
  <!-- v-on (shorthand @) -->
  <button @click="handleClick">Click me</button>
  <input @input="handleInput" />
  
  <!-- v-model: Two-way binding -->
  <input v-model="username" />
  <p>Username: {{ username }}</p>
</template>

<script>
export default {
  data() {
    return {
      isLoggedIn: false,
      isVisible: true,
      items: [
        { id: 1, name: 'Item 1' },
        { id: 2, name: 'Item 2' }
      ],
      imageUrl: '/path/to/image.jpg',
      imageAlt: 'Description',
      isActive: true,
      username: ''
    }
  },
  methods: {
    handleClick() {
      console.log('Clicked!');
    },
    handleInput(event) {
      console.log(event.target.value);
    }
  }
}
</script>
```

---

## Q8: What is the Virtual DOM?

**Answer**:

```mermaid
graph TB
    A[Virtual DOM] --> B[Lightweight Copy<br/>of Real DOM]
    A --> C[JavaScript Object<br/>Representation]
    A --> D[Efficient Updates<br/>Diffing algorithm]
    
    style A fill:#FFD700
```

### How it Works

```mermaid
sequenceDiagram
    participant S as State Change
    participant V1 as Old Virtual DOM
    participant V2 as New Virtual DOM
    participant D as Diffing
    participant R as Real DOM
    
    S->>V2: Create new Virtual DOM
    V1->>D: Compare
    V2->>D: Compare
    D->>D: Find differences
    D->>R: Update only changed parts
```

### Example

```javascript
// State changes
setState({ count: count + 1 });

// React creates new Virtual DOM
const newVDOM = {
  type: 'div',
  props: {
    children: [
      { type: 'p', props: { children: 'Count: 1' } }
    ]
  }
};

// Compare with old Virtual DOM
const oldVDOM = {
  type: 'div',
  props: {
    children: [
      { type: 'p', props: { children: 'Count: 0' } }
    ]
  }
};

// Only update the text node in real DOM
// Instead of re-rendering entire component
```

**Benefits**:
- Faster than direct DOM manipulation
- Batches multiple updates
- Cross-platform (React Native)

---

## Q9: What is component lifecycle in React?

**Answer**:

```mermaid
graph TB
    A[Component Lifecycle] --> B[Mounting<br/>Component created]
    A --> C[Updating<br/>State/props change]
    A --> D[Unmounting<br/>Component removed]
    
    B --> E[useEffect<br/>with empty deps]
    C --> F[useEffect<br/>with deps]
    D --> G[useEffect<br/>cleanup]
    
    style A fill:#FFD700
```

### useEffect Hook

```javascript
import { useState, useEffect } from 'react';

function Component() {
  const [count, setCount] = useState(0);
  
  // Runs after every render
  useEffect(() => {
    console.log('Component rendered');
  });
  
  // Runs once on mount (like componentDidMount)
  useEffect(() => {
    console.log('Component mounted');
    
    // Cleanup on unmount
    return () => {
      console.log('Component unmounted');
    };
  }, []); // Empty dependency array
  
  // Runs when count changes
  useEffect(() => {
    console.log('Count changed:', count);
  }, [count]); // Dependency array
  
  return <button onClick={() => setCount(count + 1)}>Count: {count}</button>;
}
```

### Lifecycle Flow

```mermaid
sequenceDiagram
    participant M as Mount
    participant R as Render
    participant E as Effect
    participant U as Update
    participant C as Cleanup
    
    M->>R: Initial render
    R->>E: Run effects
    
    Note over U: State/Props change
    U->>R: Re-render
    R->>C: Cleanup previous effects
    C->>E: Run new effects
    
    Note over C: Component unmounts
    C->>C: Final cleanup
```

---

## Q10: What is Vue Composition API?

**Answer**:

```mermaid
graph TB
    A[Composition API] --> B[setup Function<br/>Entry point]
    A --> C[Reactive References<br/>ref, reactive]
    A --> D[Lifecycle Hooks<br/>onMounted, etc.]
    A --> E[Better Code<br/>Organization]
    
    style A fill:#FFD700
```

### Options API vs Composition API

```vue
<!-- Options API (Traditional) -->
<script>
export default {
  data() {
    return {
      count: 0,
      message: 'Hello'
    }
  },
  methods: {
    increment() {
      this.count++;
    }
  },
  mounted() {
    console.log('Mounted');
  }
}
</script>
```

```vue
<!-- Composition API (Modern) -->
<script setup>
import { ref, onMounted } from 'vue';

const count = ref(0);
const message = ref('Hello');

function increment() {
  count.value++;
}

onMounted(() => {
  console.log('Mounted');
});
</script>

<template>
  <div>
    <p>{{ message }}</p>
    <button @click="increment">Count: {{ count }}</button>
  </div>
</template>
```

### Reactive References

```javascript
import { ref, reactive, computed } from 'vue';

// ref: For primitives
const count = ref(0);
console.log(count.value); // Access with .value
count.value++;

// reactive: For objects
const state = reactive({
  name: 'Alice',
  age: 25
});
console.log(state.name); // Direct access
state.age++;

// computed: Derived state
const doubled = computed(() => count.value * 2);
console.log(doubled.value);
```

**Benefits**:
- Better TypeScript support
- More flexible code organization
- Easier to reuse logic
- Better tree-shaking

---

## Summary

Key frontend concepts:
- **DOM**: Tree structure, manipulation
- **CSS Box Model**: Content, padding, border, margin
- **Event Propagation**: Bubbling and capturing
- **React Components**: Function components, props
- **React State**: useState hook, re-rendering
- **Vue.js**: Template syntax, directives
- **Vue Directives**: v-if, v-for, v-model, v-bind, v-on
- **Virtual DOM**: Efficient updates, diffing
- **React Lifecycle**: useEffect hook, cleanup
- **Vue Composition API**: setup, ref, reactive

These fundamentals are essential for frontend development.

