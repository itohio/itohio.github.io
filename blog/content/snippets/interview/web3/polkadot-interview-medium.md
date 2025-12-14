---
title: "Polkadot Interview Questions - Medium"
date: 2025-12-14
tags: ["polkadot", "blockchain", "interview", "medium", "substrate", "pallets"]
---

Medium-level Polkadot interview questions covering advanced Substrate development and parachain architecture.

## Q1: How do you implement custom consensus in Substrate?

**Answer**:

**Custom Consensus**:
```rust
use sc_consensus::{BlockImport, BlockImportParams};

pub struct CustomBlockImport;

impl BlockImport<Block> for CustomBlockImport {
    fn import_block(
        &mut self,
        block: BlockImportParams<Block>,
    ) -> Result<ImportResult, ConsensusError> {
        // Custom import logic
        Ok(ImportResult::Imported(ImportedAux {
            header_only: false,
            clear_justification_requests: false,
            needs_justification: false,
            bad_justification: false,
            is_new_best: true,
        }))
    }
}
```

---

## Q2: How do you optimize Substrate runtime?

**Answer**:

**Optimization Techniques**:
- Use `#[pallet::compact]` for storage
- Batch operations
- Limit storage reads/writes
- Use `on_initialize` for pre-processing

---

## Q3: How do you implement cross-chain messaging?

**Answer**:

**XCM Implementation**:
```rust
use xcm::prelude::*;

pub fn send_cross_chain(
    dest: MultiLocation,
    message: Xcm<()>,
) -> Result<(), XcmError> {
    XcmPallet::<T>::send(
        RawOrigin::Root.into(),
        Box::new(dest.into()),
        Box::new(message),
    )?;
    Ok(())
}
```

---

