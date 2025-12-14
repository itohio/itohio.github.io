---
title: "Vue.js Essentials - From Setup to Todo App"
date: 2024-12-12
draft: false
category: "frontend"
tags: ["frontend-knowhow", "vue", "javascript", "typescript"]
---


Complete Vue.js 3 guide from project setup to building a functional Todo application. Includes Composition API, TypeScript, and best practices.

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
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  vue-app:
    build: .
    container_name: vue-app
    ports:
      - "3000:80"
    restart: unless-stopped

  # Development
  vue-dev:
    image: node:18-alpine
    container_name: vue-dev
    working_dir: /app
    volumes:
      - ./:/app
      - /app/node_modules
    ports:
      - "5173:5173"
    command: npm run dev -- --host
    restart: unless-stopped
```

---

## Project Setup

### Vite (Recommended)

```bash
# Create new Vue app
npm create vue@latest my-app

# Follow prompts:
# ✔ Add TypeScript? › Yes
# ✔ Add JSX Support? › No
# ✔ Add Vue Router? › Yes
# ✔ Add Pinia (state management)? › Yes
# ✔ Add Vitest (unit testing)? › No
# ✔ Add ESLint? › Yes

cd my-app
npm install
npm run dev
```

**Node.js Version Requirements:**
- Vue 3.3+: Node.js 18+ or 20+
- Vue 3.2: Node.js 14.18+ or 16+

**Check versions:**
```bash
node --version
npm --version
vue --version
```

---

### Vue CLI (Legacy)

```bash
npm install -g @vue/cli
vue create my-app
cd my-app
npm run serve
```

**⚠️ Gotcha:** Vue CLI is in maintenance mode. Use Vite instead.

---

## Basic Vue Concepts

### Single File Components (SFC)

```vue
<!-- HelloWorld.vue -->
<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <button @click="count++">Count: {{ count }}</button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

// Props
defineProps<{
  msg: string;
}>();

// Reactive state
const count = ref(0);
</script>

<style scoped>
.hello {
  color: #42b983;
}
</style>
```

---

### Composition API vs Options API

#### Options API (Vue 2 style)

```vue
<script lang="ts">
export default {
  data() {
    return {
      count: 0,
      message: 'Hello'
    };
  },
  computed: {
    doubleCount() {
      return this.count * 2;
    }
  },
  methods: {
    increment() {
      this.count++;
    }
  },
  mounted() {
    console.log('Component mounted');
  }
};
</script>
```

#### Composition API (Vue 3 - Recommended)

```vue
<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';

const count = ref(0);
const message = ref('Hello');

const doubleCount = computed(() => count.value * 2);

const increment = () => {
  count.value++;
};

onMounted(() => {
  console.log('Component mounted');
});
</script>
```

**⚠️ Gotcha:** In Composition API, always use `.value` to access/modify refs!

```ts
// ❌ Wrong
count = 5;

// ✅ Correct
count.value = 5;
```

---

### Reactivity

#### ref vs reactive

```vue
<script setup lang="ts">
import { ref, reactive } from 'vue';

// ref - for primitives (unwraps in template)
const count = ref(0);
const message = ref('Hello');

// reactive - for objects
const state = reactive({
  count: 0,
  message: 'Hello'
});

// In script: use .value for ref
count.value++;

// In script: direct access for reactive
state.count++;
</script>

<template>
  <!-- In template: no .value needed for ref -->
  <p>{{ count }}</p>
  <p>{{ state.count }}</p>
</template>
```

**⚠️ Gotcha:** Don't destructure reactive objects!

```ts
// ❌ Wrong - loses reactivity
const { count, message } = reactive({ count: 0, message: 'Hello' });

// ✅ Correct - use toRefs
import { toRefs } from 'vue';
const state = reactive({ count: 0, message: 'Hello' });
const { count, message } = toRefs(state);
```

---

### Computed Properties

```vue
<script setup lang="ts">
import { ref, computed } from 'vue';

const firstName = ref('John');
const lastName = ref('Doe');

// Read-only computed
const fullName = computed(() => {
  return `${firstName.value} ${lastName.value}`;
});

// Writable computed
const fullNameWritable = computed({
  get() {
    return `${firstName.value} ${lastName.value}`;
  },
  set(value: string) {
    const parts = value.split(' ');
    firstName.value = parts[0];
    lastName.value = parts[1];
  }
});
</script>
```

---

### Watchers

```vue
<script setup lang="ts">
import { ref, watch, watchEffect } from 'vue';

const count = ref(0);
const message = ref('');

// Watch single source
watch(count, (newValue, oldValue) => {
  console.log(`Count changed from ${oldValue} to ${newValue}`);
});

// Watch multiple sources
watch([count, message], ([newCount, newMessage], [oldCount, oldMessage]) => {
  console.log('Something changed');
});

// Watch with options
watch(count, (newValue) => {
  console.log('Count:', newValue);
}, {
  immediate: true, // Run immediately
  deep: true       // Deep watch for objects
});

// watchEffect - automatically tracks dependencies
watchEffect(() => {
  console.log(`Count is ${count.value}`);
  // Automatically re-runs when count changes
});
</script>
```

---

### Lifecycle Hooks

```vue
<script setup lang="ts">
import {
  onBeforeMount,
  onMounted,
  onBeforeUpdate,
  onUpdated,
  onBeforeUnmount,
  onUnmounted
} from 'vue';

onBeforeMount(() => {
  console.log('Before mount');
});

onMounted(() => {
  console.log('Mounted - DOM is ready');
  // Fetch data, setup timers, etc.
});

onBeforeUpdate(() => {
  console.log('Before update');
});

onUpdated(() => {
  console.log('Updated');
});

onBeforeUnmount(() => {
  console.log('Before unmount');
  // Cleanup: remove event listeners, cancel timers
});

onUnmounted(() => {
  console.log('Unmounted');
});
</script>
```

---

## Complete Todo App Example

```vue
<!-- App.vue -->
<template>
  <div class="app">
    <h1>Vue Todo App</h1>
    
    <!-- Input -->
    <div class="input-container">
      <input
        v-model="newTodo"
        @keyup.enter="addTodo"
        placeholder="What needs to be done?"
      />
      <button @click="addTodo">Add</button>
    </div>
    
    <!-- Filter buttons -->
    <div class="filters">
      <button
        :class="{ active: filter === 'all' }"
        @click="filter = 'all'"
      >
        All ({{ todos.length }})
      </button>
      <button
        :class="{ active: filter === 'active' }"
        @click="filter = 'active'"
      >
        Active ({{ activeCount }})
      </button>
      <button
        :class="{ active: filter === 'completed' }"
        @click="filter = 'completed'"
      >
        Completed ({{ completedCount }})
      </button>
    </div>
    
    <!-- Todo list -->
    <ul class="todo-list">
      <TodoItem
        v-for="todo in filteredTodos"
        :key="todo.id"
        :todo="todo"
        @toggle="toggleTodo"
        @delete="deleteTodo"
        @edit="editTodo"
      />
    </ul>
    
    <!-- Footer -->
    <div v-if="todos.length > 0" class="footer">
      <span>{{ activeCount }} item{{ activeCount !== 1 ? 's' : '' }} left</span>
      <button @click="clearCompleted">Clear completed</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import TodoItem from './components/TodoItem.vue';

interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

type Filter = 'all' | 'active' | 'completed';

// State
const todos = ref<Todo[]>([]);
const newTodo = ref('');
const filter = ref<Filter>('all');

// Computed
const filteredTodos = computed(() => {
  switch (filter.value) {
    case 'active':
      return todos.value.filter(todo => !todo.completed);
    case 'completed':
      return todos.value.filter(todo => todo.completed);
    default:
      return todos.value;
  }
});

const activeCount = computed(() => {
  return todos.value.filter(todo => !todo.completed).length;
});

const completedCount = computed(() => {
  return todos.value.filter(todo => todo.completed).length;
});

// Methods
const addTodo = () => {
  if (newTodo.value.trim() === '') return;
  
  todos.value.push({
    id: Date.now(),
    text: newTodo.value,
    completed: false
  });
  
  newTodo.value = '';
};

const toggleTodo = (id: number) => {
  const todo = todos.value.find(t => t.id === id);
  if (todo) {
    todo.completed = !todo.completed;
  }
};

const deleteTodo = (id: number) => {
  todos.value = todos.value.filter(t => t.id !== id);
};

const editTodo = (id: number, newText: string) => {
  const todo = todos.value.find(t => t.id === id);
  if (todo) {
    todo.text = newText;
  }
};

const clearCompleted = () => {
  todos.value = todos.value.filter(t => !t.completed);
};
</script>

<style scoped>
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
  background: #42b983;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.input-container button:hover {
  background: #3aa876;
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
  background: #42b983;
  color: white;
}

.todo-list {
  list-style: none;
  padding: 0;
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
</style>
```

### TodoItem Component

```vue
<!-- components/TodoItem.vue -->
<template>
  <li :class="['todo-item', { completed: todo.completed, editing: isEditing }]">
    <template v-if="!isEditing">
      <input
        type="checkbox"
        :checked="todo.completed"
        @change="$emit('toggle', todo.id)"
      />
      <span @dblclick="startEdit">{{ todo.text }}</span>
      <button @click="$emit('delete', todo.id)">×</button>
    </template>
    
    <template v-else>
      <input
        ref="editInput"
        v-model="editText"
        type="text"
        @blur="finishEdit"
        @keyup.enter="finishEdit"
        @keyup.esc="cancelEdit"
      />
    </template>
  </li>
</template>

<script setup lang="ts">
import { ref, nextTick } from 'vue';

interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

const props = defineProps<{
  todo: Todo;
}>();

const emit = defineEmits<{
  toggle: [id: number];
  delete: [id: number];
  edit: [id: number, text: string];
}>();

const isEditing = ref(false);
const editText = ref('');
const editInput = ref<HTMLInputElement | null>(null);

const startEdit = () => {
  isEditing.value = true;
  editText.value = props.todo.text;
  
  // Focus input after DOM update
  nextTick(() => {
    editInput.value?.focus();
  });
};

const finishEdit = () => {
  if (editText.value.trim() === '') {
    emit('delete', props.todo.id);
  } else {
    emit('edit', props.todo.id, editText.value);
  }
  isEditing.value = false;
};

const cancelEdit = () => {
  isEditing.value = false;
  editText.value = props.todo.text;
};
</script>

<style scoped>
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

.todo-item.editing input[type="text"] {
  flex: 1;
  padding: 8px;
  font-size: 16px;
  border: 1px solid #42b983;
  border-radius: 4px;
}
</style>
```

---

## Common Gotchas

### 1. Template Syntax

```vue
<template>
  <!-- ✅ Correct: v-bind shorthand -->
  <img :src="imageUrl" :alt="altText" />
  
  <!-- ✅ Correct: v-on shorthand -->
  <button @click="handleClick">Click</button>
  
  <!-- ✅ Correct: v-model -->
  <input v-model="message" />
  
  <!-- ❌ Wrong: using .value in template -->
  <p>{{ count.value }}</p>
  
  <!-- ✅ Correct: no .value in template -->
  <p>{{ count }}</p>
</template>
```

### 2. Event Modifiers

```vue
<template>
  <!-- Prevent default -->
  <form @submit.prevent="handleSubmit">
  
  <!-- Stop propagation -->
  <button @click.stop="handleClick">
  
  <!-- Key modifiers -->
  <input @keyup.enter="submit" />
  <input @keyup.esc="cancel" />
  
  <!-- Mouse modifiers -->
  <button @click.right="showContextMenu">
  
  <!-- System modifiers -->
  <input @keyup.ctrl.enter="send" />
  
  <!-- Once modifier -->
  <button @click.once="doOnce">
</template>
```

### 3. Conditional Rendering

```vue
<template>
  <!-- v-if: conditionally render element -->
  <div v-if="isVisible">Visible</div>
  <div v-else-if="isHidden">Hidden</div>
  <div v-else>Default</div>
  
  <!-- v-show: toggle CSS display -->
  <div v-show="isVisible">Toggle visibility</div>
  
  <!-- ⚠️ v-if removes from DOM, v-show just hides -->
  <!-- Use v-if for rare toggles, v-show for frequent toggles -->
</template>
```

### 4. List Rendering

```vue
<template>
  <!-- ✅ Correct: unique key -->
  <div v-for="item in items" :key="item.id">
    {{ item.name }}
  </div>
  
  <!-- ❌ Wrong: index as key (avoid if list changes) -->
  <div v-for="(item, index) in items" :key="index">
    {{ item.name }}
  </div>
  
  <!-- ✅ Correct: with index -->
  <div v-for="(item, index) in items" :key="item.id">
    {{ index }}: {{ item.name }}
  </div>
</template>
```

### 5. Props and Emits

```vue
<script setup lang="ts">
// ✅ Correct: typed props
const props = defineProps<{
  title: string;
  count?: number; // Optional
}>();

// ✅ Correct: with defaults
const props = withDefaults(defineProps<{
  title: string;
  count?: number;
}>(), {
  count: 0
});

// ✅ Correct: typed emits
const emit = defineEmits<{
  update: [value: string];
  delete: [id: number];
}>();

// Emit event
emit('update', 'new value');
</script>
```

---