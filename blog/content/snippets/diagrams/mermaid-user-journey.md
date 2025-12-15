---
title: "Mermaid User Journey Diagrams"
date: 2024-12-12T23:10:00Z
draft: false
description: "Create user journey maps with Mermaid"
tags: ["mermaid", "user-journey", "ux", "customer-journey", "diagram", "diagrams"]
category: "diagrams"
---

User journey diagrams map the user's experience through a process or system. Perfect for UX design, customer experience mapping, and process optimization.

## Use Case

Use user journey diagrams when you need to:
- Map user experiences
- Design UX flows
- Document customer journeys
- Identify pain points
- Optimize user flows

## Code

````markdown
```mermaid
journey
    title User Journey
    section Sign Up
      Visit Site: 5: User
      Create Account: 4: User
    section Onboarding
      Complete Profile: 3: User
      First Action: 5: User
```
````

**Result:**

```mermaid
journey
    title User Journey
    section Sign Up
      Visit Site: 5: User
      Create Account: 4: User
    section Onboarding
      Complete Profile: 3: User
      First Action: 5: User
```

## Explanation

- `journey` - Start user journey diagram
- `title` - Journey title
- `section` - Major phase of journey
- `Task: Score: Actor` - Task format
- Score: 1-5 (satisfaction/importance)

## Examples

### Example 1: E-Commerce Purchase

````markdown
```mermaid
journey
    title Shopping Experience
    section Discovery
      Browse Products: 5: Customer
      Search Items: 4: Customer
      View Details: 5: Customer
    section Purchase
      Add to Cart: 4: Customer
      Checkout: 3: Customer
      Payment: 2: Customer
    section Delivery
      Receive Order: 5: Customer
      Unbox Product: 5: Customer
```
````

**Result:**

```mermaid
journey
    title Shopping Experience
    section Discovery
      Browse Products: 5: Customer
      Search Items: 4: Customer
      View Details: 5: Customer
    section Purchase
      Add to Cart: 4: Customer
      Checkout: 3: Customer
      Payment: 2: Customer
    section Delivery
      Receive Order: 5: Customer
      Unbox Product: 5: Customer
```

### Example 2: Application Onboarding

````markdown
```mermaid
journey
    title App Onboarding
    section Sign Up
      Download App: 5: User
      Create Account: 4: User
      Verify Email: 3: User
    section Setup
      Complete Profile: 4: User
      Set Preferences: 3: User
      Tutorial: 5: User
    section First Use
      First Feature: 5: User
      Explore: 4: User
      Get Help: 3: User
```
````

**Result:**

```mermaid
journey
    title App Onboarding
    section Sign Up
      Download App: 5: User
      Create Account: 4: User
      Verify Email: 3: User
    section Setup
      Complete Profile: 4: User
      Set Preferences: 3: User
      Tutorial: 5: User
    section First Use
      First Feature: 5: User
      Explore: 4: User
      Get Help: 3: User
```

## Notes

- Score range: 1-5 (typically satisfaction or importance)
- Actor identifies who performs the task
- Sections group related tasks
- Use descriptive task names

## Gotchas/Warnings

- ⚠️ **Format**: Must use `Task: Score: Actor` format
- ⚠️ **Score**: Must be 1-5 (integer)
- ⚠️ **Sections**: Group related tasks logically
- ⚠️ **Actor**: Keep actor names consistent

