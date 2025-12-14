---
title: "Cardano Interview Questions - Medium"
date: 2025-12-14
tags: ["cardano", "blockchain", "interview", "medium", "plutus", "optimization"]
---

Medium-level Cardano interview questions covering advanced Plutus development and optimization.

## Q1: How do you optimize Plutus contract execution costs?

**Answer**:

**Optimization**:
- Minimize on-chain code
- Use off-chain code when possible
- Optimize datum/redeemer size
- Batch operations

---

## Q2: How do you implement complex state machines in Plutus?

**Answer**:

**State Machine**:
```haskell
data State = State1 | State2 | State3
data Transition = Transition1 | Transition2

transition :: State -> Transition -> Maybe State
transition State1 Transition1 = Just State2
transition State2 Transition2 = Just State3
transition _ _ = Nothing
```

---

## Q3: How do you handle time-locked transactions?

**Answer**:

**Time Locks**:
```haskell
{-# INLINABLE timeLocked #-}
timeLocked :: POSIXTime -> ScriptContext -> Bool
timeLocked deadline ctx =
    txInfoValidRange ctx `contains` (from deadline)
```

---

