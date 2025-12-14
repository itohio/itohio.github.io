---
title: "Solana Interview Questions - Medium"
date: 2025-12-14
tags: ["solana", "blockchain", "interview", "medium", "rust", "programs"]
---

Medium-level Solana interview questions covering advanced program development, optimization, and architecture.

## Q1: How do you implement cross-program invocations (CPIs)?

**Answer**:

**CPI Basics**:
```rust
use solana_program::{
    program::invoke,
    account_info::AccountInfo,
};

pub fn call_other_program(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let instruction = Instruction {
        program_id: *program_id,
        accounts: accounts.to_vec(),
        data: instruction_data.to_vec(),
    };
    
    invoke(&instruction, accounts)?;
    Ok(())
}
```

**CPI with Anchor**:
```rust
use anchor_lang::prelude::*;

#[program]
pub mod my_program {
    use super::*;
    
    pub fn call_token_program(ctx: Context<CallToken>, amount: u64) -> Result<()> {
        let cpi_accounts = token::Transfer {
            from: ctx.accounts.from.to_account_info(),
            to: ctx.accounts.to.to_account_info(),
            authority: ctx.accounts.authority.to_account_info(),
        };
        
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
        
        token::transfer(cpi_ctx, amount)?;
        Ok(())
    }
}
```

---

## Q2: How do you optimize Solana program compute units?

**Answer**:

**Compute Optimization**:
```rust
// Bad: Inefficient
for i in 0..1000 {
    // Expensive operation
}

// Good: Batch operations
let batch_size = 100;
for i in (0..1000).step_by(batch_size) {
    // Process batch
}

// Use compute budget
use solana_program::compute_budget::ComputeBudgetInstruction;

let modify_compute_units = ComputeBudgetInstruction::set_compute_unit_limit(200_000);
let modify_priority_fee = ComputeBudgetInstruction::set_compute_unit_price(1);
```

---

## Q3: How do you implement account ownership and access control?

**Answer**:

**Ownership Checks**:
```rust
pub fn verify_owner(account: &AccountInfo, expected_owner: &Pubkey) -> ProgramResult {
    if account.owner != expected_owner {
        return Err(ProgramError::IncorrectProgramId);
    }
    Ok(())
}

// Signer check
pub fn verify_signer(account: &AccountInfo) -> ProgramResult {
    if !account.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    Ok(())
}
```

---

## Q4: How do you handle errors and program exceptions?

**Answer**:

**Error Handling**:
```rust
#[derive(Debug)]
pub enum MyError {
    #[error("Insufficient funds")]
    InsufficientFunds,
    #[error("Invalid account")]
    InvalidAccount,
}

impl From<MyError> for ProgramError {
    fn from(e: MyError) -> Self {
        ProgramError::Custom(e as u32)
    }
}
```

---

## Q5: How do you implement token swaps and AMM logic?

**Answer**:

**Simple AMM**:
```rust
pub fn swap(
    ctx: Context<Swap>,
    amount_in: u64,
    minimum_amount_out: u64,
) -> Result<()> {
    let pool = &mut ctx.accounts.pool;
    
    // Calculate output (constant product)
    let amount_out = (pool.token_b_amount * amount_in) / (pool.token_a_amount + amount_in);
    
    require!(amount_out >= minimum_amount_out, SwapError::SlippageExceeded);
    
    // Transfer tokens
    // ...
    
    Ok(())
}
```

---

