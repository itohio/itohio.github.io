---
title: "Common Code Smells"
date: 2024-12-12
draft: false
category: "codereview"
tags: ["codereview-knowhow", "code-smells", "refactoring", "best-practices"]
---

Common code smells to watch for during code reviews with examples and fixes.

---

## Long Methods

**Problem**: Methods that do too many things are hard to understand, test, and maintain.

```typescript
// ❌ Bad: 200-line method doing everything
function processOrder(order) {
  // validate order
  if (!order.id) throw new Error("Invalid order");
  if (!order.items || order.items.length === 0) throw new Error("No items");
  
  // calculate totals
  let subtotal = 0;
  for (const item of order.items) {
    subtotal += item.price * item.quantity;
  }
  const tax = subtotal * 0.08;
  const shipping = subtotal > 100 ? 0 : 10;
  const total = subtotal + tax + shipping;
  
  // update database
  db.orders.update(order.id, { total, status: 'processing' });
  
  // send email
  emailService.send(order.email, 'Order Confirmation', `Total: $${total}`);
  
  // log
  logger.info(`Order ${order.id} processed`);
  
  // ... 180 more lines
}

// ✅ Good: Broken into smaller, focused functions
function processOrder(order) {
  validateOrder(order);
  const total = calculateTotal(order);
  updateDatabase(order, total);
  sendConfirmationEmail(order, total);
  logOrderProcessed(order);
}

function validateOrder(order) {
  if (!order.id) throw new Error("Invalid order");
  if (!order.items || order.items.length === 0) throw new Error("No items");
}

function calculateTotal(order) {
  const subtotal = order.items.reduce((sum, item) => 
    sum + (item.price * item.quantity), 0);
  const tax = subtotal * TAX_RATE;
  const shipping = subtotal > FREE_SHIPPING_THRESHOLD ? 0 : SHIPPING_COST;
  return subtotal + tax + shipping;
}
```

**Rule of Thumb**: If a function is longer than your screen, it's probably too long.

---

## Too Many Parameters

**Problem**: Functions with many parameters are hard to call, test, and remember.

```typescript
// ❌ Bad: 7 parameters
function createUser(
  name: string,
  email: string,
  age: number,
  address: string,
  phone: string,
  role: string,
  department: string
) {
  // ...
}

// Calling it is error-prone
createUser("John", "john@example.com", 30, "123 Main St", "555-1234", "admin", "IT");

// ✅ Good: Use object parameter
interface UserData {
  name: string;
  email: string;
  age: number;
  address: string;
  phone: string;
  role: string;
  department: string;
}

function createUser(userData: UserData) {
  // ...
}

// Much clearer
createUser({
  name: "John",
  email: "john@example.com",
  age: 30,
  address: "123 Main St",
  phone: "555-1234",
  role: "admin",
  department: "IT"
});
```

**Rule of Thumb**: More than 3 parameters? Consider using an object.

---

## God Objects

**Problem**: Classes that know too much or do too much violate Single Responsibility Principle.

```typescript
// ❌ Bad: One class doing everything
class OrderManager {
  validateOrder(order) {}
  calculateTotal(order) {}
  calculateTax(amount) {}
  calculateShipping(order) {}
  processPayment(order) {}
  chargeCard(card, amount) {}
  refundPayment(orderId) {}
  sendEmail(to, subject, body) {}
  sendSMS(phone, message) {}
  generateInvoice(order) {}
  generatePDF(invoice) {}
  updateInventory(items) {}
  checkStock(itemId) {}
  reserveStock(items) {}
  logTransaction(order) {}
  logError(error) {}
  // ... 50 more methods
}

// ✅ Good: Separate responsibilities
class OrderValidator {
  validate(order) {}
}

class OrderCalculator {
  calculateTotal(order) {}
  calculateTax(amount) {}
  calculateShipping(order) {}
}

class PaymentProcessor {
  processPayment(order) {}
  chargeCard(card, amount) {}
  refundPayment(orderId) {}
}

class NotificationService {
  sendEmail(to, subject, body) {}
  sendSMS(phone, message) {}
}

class InvoiceGenerator {
  generate(order) {}
  generatePDF(invoice) {}
}

class InventoryManager {
  updateInventory(items) {}
  checkStock(itemId) {}
  reserveStock(items) {}
}

class TransactionLogger {
  logTransaction(order) {}
  logError(error) {}
}
```

**Rule of Thumb**: If your class has more than 10 methods, it's probably doing too much.

---

## Magic Numbers

**Problem**: Unexplained numbers in code make it hard to understand and maintain.

```typescript
// ❌ Bad
if (user.age > 18 && user.score > 100 && user.accountAge > 365) {
  grantPremiumAccess(user);
}

// What do these numbers mean?
const discount = price * 0.15;
const shipping = weight > 5 ? 25 : 10;

// ✅ Good: Named constants
const LEGAL_AGE = 18;
const MINIMUM_SCORE = 100;
const DAYS_IN_YEAR = 365;
const PREMIUM_DISCOUNT_RATE = 0.15;
const HEAVY_PACKAGE_THRESHOLD_KG = 5;
const STANDARD_SHIPPING_COST = 10;
const HEAVY_SHIPPING_COST = 25;

if (user.age > LEGAL_AGE && 
    user.score > MINIMUM_SCORE && 
    user.accountAge > DAYS_IN_YEAR) {
  grantPremiumAccess(user);
}

const discount = price * PREMIUM_DISCOUNT_RATE;
const shipping = weight > HEAVY_PACKAGE_THRESHOLD_KG 
  ? HEAVY_SHIPPING_COST 
  : STANDARD_SHIPPING_COST;
```

**Rule of Thumb**: Any number other than 0, 1, or -1 should probably be a named constant.

---

## Nested Conditionals

**Problem**: Deep nesting makes code hard to read and reason about.

```typescript
// ❌ Bad: Arrow of doom
if (user) {
  if (user.isActive) {
    if (user.hasPermission('write')) {
      if (user.balance > 0) {
        if (!user.isSuspended) {
          // do something
          return result;
        }
      }
    }
  }
}

// ✅ Good: Early returns (guard clauses)
if (!user) return;
if (!user.isActive) return;
if (!user.hasPermission('write')) return;
if (user.balance <= 0) return;
if (user.isSuspended) return;

// do something
return result;

// ✅ Also good: Extract to function
function canUserWrite(user) {
  return user &&
         user.isActive &&
         user.hasPermission('write') &&
         user.balance > 0 &&
         !user.isSuspended;
}

if (canUserWrite(user)) {
  // do something
  return result;
}
```

**Rule of Thumb**: More than 2 levels of nesting? Refactor with early returns or extract functions.

---

## Commented-Out Code

**Problem**: Dead code clutters the codebase and creates confusion.

```typescript
// ❌ Bad
function processData(data) {
  // const oldWay = data.map(x => x * 2);
  // return oldWay.filter(x => x > 10);
  
  // const anotherWay = data.filter(x => x > 5).map(x => x * 2);
  
  return data.map(x => x * 2).filter(x => x > 10);
}

// ✅ Good: Remove it (it's in git history if you need it)
function processData(data) {
  return data.map(x => x * 2).filter(x => x > 10);
}
```

**Rule of Thumb**: Delete commented-out code. If you need it later, use git history.

---

## Inconsistent Naming

**Problem**: Inconsistent names make code harder to understand and search.

```typescript
// ❌ Bad: Inconsistent naming
const usr = getUser();
const userInfo = getUserData();
const u = fetchUserDetails();
const user_profile = loadUserProfile();

function get_user_by_id(id) {}
function fetchUserByEmail(email) {}
function UserByName(name) {}

// ✅ Good: Consistent naming
const user = getUser();
const userDetails = getUserDetails();
const userProfile = getUserProfile();
const userSettings = getUserSettings();

function getUserById(id) {}
function getUserByEmail(email) {}
function getUserByName(name) {}
```

**Rule of Thumb**: Pick a naming convention and stick to it throughout the project.

---

## Boolean Parameters

**Problem**: Boolean parameters make function calls unclear.

```typescript
// ❌ Bad: What does true mean here?
sendEmail(user, true);
processOrder(order, false, true);
createUser(data, true, false, true);

// ✅ Good: Use options object or separate functions
interface EmailOptions {
  urgent: boolean;
}

sendEmail(user, { urgent: true });

// Or use separate functions
sendUrgentEmail(user);
sendRegularEmail(user);

// Or use enums
enum Priority {
  URGENT,
  NORMAL
}

sendEmail(user, Priority.URGENT);
```

**Rule of Thumb**: Avoid boolean parameters. Use options objects or enums instead.

---

## Primitive Obsession

**Problem**: Using primitives instead of small objects loses type safety and meaning.

```typescript
// ❌ Bad: Primitives everywhere
function createUser(
  email: string,
  phone: string,
  zipCode: string
) {
  // Easy to mix up parameters
  // No validation
}

createUser("555-1234", "john@example.com", "12345"); // Oops, wrong order!

// ✅ Good: Use value objects
class Email {
  constructor(private value: string) {
    if (!this.isValid(value)) {
      throw new Error("Invalid email");
    }
  }
  
  private isValid(email: string): boolean {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }
  
  toString(): string {
    return this.value;
  }
}

class PhoneNumber {
  constructor(private value: string) {
    if (!this.isValid(value)) {
      throw new Error("Invalid phone number");
    }
  }
  
  private isValid(phone: string): boolean {
    return /^\d{3}-\d{3}-\d{4}$/.test(phone);
  }
  
  toString(): string {
    return this.value;
  }
}

function createUser(
  email: Email,
  phone: PhoneNumber,
  zipCode: ZipCode
) {
  // Type safe, validated, clear intent
}

// Can't mix up the order - compiler error!
createUser(
  new Email("john@example.com"),
  new PhoneNumber("555-1234"),
  new ZipCode("12345")
);
```

**Rule of Thumb**: If a primitive has validation rules or behavior, make it a class.

---

## Long Parameter Lists in Constructors

**Problem**: Classes with many constructor parameters are hard to instantiate.

```typescript
// ❌ Bad
class User {
  constructor(
    name: string,
    email: string,
    age: number,
    address: string,
    phone: string,
    role: string,
    department: string,
    manager: string
  ) {
    // ...
  }
}

// ✅ Good: Use builder pattern
class UserBuilder {
  private name: string;
  private email: string;
  private age: number;
  private address: string;
  private phone: string;
  private role: string;
  private department: string;
  private manager: string;
  
  setName(name: string): this {
    this.name = name;
    return this;
  }
  
  setEmail(email: string): this {
    this.email = email;
    return this;
  }
  
  // ... other setters
  
  build(): User {
    return new User(this);
  }
}

const user = new UserBuilder()
  .setName("John")
  .setEmail("john@example.com")
  .setAge(30)
  .build();
```

**Rule of Thumb**: More than 3 constructor parameters? Consider builder pattern or options object.

---

## Feature Envy

**Problem**: A method that uses more features of another class than its own.

```typescript
// ❌ Bad: OrderProcessor is envious of Order's data
class OrderProcessor {
  calculateTotal(order: Order) {
    let total = 0;
    for (const item of order.items) {
      total += item.price * item.quantity;
    }
    const tax = total * order.taxRate;
    const shipping = order.shippingCost;
    return total + tax + shipping;
  }
}

// ✅ Good: Move the method to where the data is
class Order {
  items: Item[];
  taxRate: number;
  shippingCost: number;
  
  calculateTotal(): number {
    const subtotal = this.items.reduce(
      (sum, item) => sum + (item.price * item.quantity), 
      0
    );
    const tax = subtotal * this.taxRate;
    return subtotal + tax + this.shippingCost;
  }
}
```

**Rule of Thumb**: Methods should primarily use data from their own class.

---

## Quick Reference

### Immediate Red Flags

- 🚩 Functions > 50 lines
- 🚩 Classes > 500 lines
- 🚩 More than 3 levels of nesting
- 🚩 More than 5 parameters
- 🚩 Commented-out code
- 🚩 Magic numbers
- 🚩 Inconsistent naming
- 🚩 No error handling

### Refactoring Priorities

1. **Security issues** - Fix immediately
2. **Bugs** - Fix before merge
3. **Code smells** - Fix if time permits or create tech debt ticket
4. **Style issues** - Use automated tools (linters)

