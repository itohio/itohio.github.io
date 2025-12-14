---
title: "Cosmos SDK Interview Questions - Medium"
date: 2025-12-14
tags: ["cosmos", "cosmos-sdk", "interview", "medium", "blockchain", "golang"]
---

Medium-level Cosmos SDK interview questions covering advanced module development, SDK internals, and Ignite customization.

## Q1: How do you implement custom state transitions and state machines in a module?

**Answer**:

**State Machine Pattern**:
```go
package keeper

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
)

// OrderState represents order state
type OrderState uint8

const (
    OrderStatePending OrderState = iota
    OrderStateConfirmed
    OrderStateShipped
    OrderStateDelivered
    OrderStateCancelled
)

// Order represents an order
type Order struct {
    Id      uint64
    State   OrderState
    Buyer   sdk.AccAddress
    Seller  sdk.AccAddress
    Amount  sdk.Coins
}

// StateMachine manages order state transitions
type StateMachine struct {
    keeper Keeper
}

// Transition validates and executes state transition
func (sm StateMachine) Transition(ctx sdk.Context, orderId uint64, newState OrderState) error {
    order, found := sm.keeper.GetOrder(ctx, orderId)
    if !found {
        return sdkerrors.Wrap(types.ErrOrderNotFound, fmt.Sprintf("order %d not found", orderId))
    }
    
    // Validate transition
    if !sm.isValidTransition(order.State, newState) {
        return sdkerrors.Wrap(
            types.ErrInvalidStateTransition,
            fmt.Sprintf("cannot transition from %d to %d", order.State, newState),
        )
    }
    
    // Execute transition
    order.State = newState
    sm.keeper.SetOrder(ctx, order)
    
    // Emit event
    ctx.EventManager().EmitEvent(
        sdk.NewEvent(
            types.EventTypeOrderStateChanged,
            sdk.NewAttribute(types.AttributeKeyOrderId, fmt.Sprintf("%d", orderId)),
            sdk.NewAttribute(types.AttributeKeyOldState, fmt.Sprintf("%d", order.State)),
            sdk.NewAttribute(types.AttributeKeyNewState, fmt.Sprintf("%d", newState)),
        ),
    )
    
    return nil
}

// isValidTransition checks if transition is allowed
func (sm StateMachine) isValidTransition(current, next OrderState) bool {
    transitions := map[OrderState][]OrderState{
        OrderStatePending:   {OrderStateConfirmed, OrderStateCancelled},
        OrderStateConfirmed: {OrderStateShipped, OrderStateCancelled},
        OrderStateShipped:     {OrderStateDelivered},
        OrderStateDelivered:  {},
        OrderStateCancelled: {},
    }
    
    allowed, exists := transitions[current]
    if !exists {
        return false
    }
    
    for _, state := range allowed {
        if state == next {
            return true
        }
    }
    
    return false
}
```

**Usage**:
```go
// In handler
func (k Keeper) ConfirmOrder(ctx sdk.Context, orderId uint64) error {
    sm := StateMachine{keeper: k}
    return sm.Transition(ctx, orderId, OrderStateConfirmed)
}
```

---

## Q2: How do you implement custom authentication and authorization in Cosmos SDK?

**Answer**:

**Custom AnteHandler**:
```go
package ante

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
    sdkerrors "github.com/cosmos/cosmos-sdk/types/errors"
    "github.com/cosmos/cosmos-sdk/x/auth/ante"
)

// CustomAnteHandler implements custom authentication
type CustomAnteHandler struct {
    accountKeeper ante.AccountKeeper
    signer        SignerVerifier
}

// NewAnteHandler creates a new ante handler
func NewAnteHandler(
    ak ante.AccountKeeper,
    signer SignerVerifier,
) sdk.AnteHandler {
    return sdk.ChainAnteDecorators(
        ante.NewSetUpContextDecorator(),
        ante.NewRejectExtensionOptionsDecorator(),
        ante.NewMempoolFeeDecorator(),
        ante.NewValidateBasicDecorator(),
        ante.NewTxTimeoutHeightDecorator(),
        ante.NewValidateSigCountDecorator(ak),
        ante.NewDeductFeeDecorator(ak),
        NewCustomAuthDecorator(ak, signer),
        ante.NewSetPubKeyDecorator(ak),
        ante.NewValidateSigDecorator(ak),
        ante.NewIncrementSequenceDecorator(ak),
    )
}

// CustomAuthDecorator performs custom authentication
type CustomAuthDecorator struct {
    accountKeeper ante.AccountKeeper
    signer         SignerVerifier
}

func NewCustomAuthDecorator(ak ante.AccountKeeper, signer SignerVerifier) CustomAuthDecorator {
    return CustomAuthDecorator{
        accountKeeper: ak,
        signer:        signer,
    }
}

// AnteHandle performs authentication
func (cad CustomAuthDecorator) AnteHandle(
    ctx sdk.Context,
    tx sdk.Tx,
    simulate bool,
    next sdk.AnteHandler,
) (sdk.Context, error) {
    sigTx, ok := tx.(authsigning.SigVerifiableTx)
    if !ok {
        return ctx, sdkerrors.Wrap(sdkerrors.ErrTxDecode, "invalid transaction type")
    }
    
    // Get signers
    signers := sigTx.GetSigners()
    
    // Verify each signer
    for _, signer := range signers {
        // Check if signer is authorized
        if !cad.signer.IsAuthorized(ctx, signer) {
            return ctx, sdkerrors.Wrap(
                sdkerrors.ErrUnauthorized,
                fmt.Sprintf("signer %s is not authorized", signer.String()),
            )
        }
        
        // Additional custom checks
        if err := cad.signer.VerifySignature(ctx, signer, sigTx); err != nil {
            return ctx, err
        }
    }
    
    return next(ctx, tx, simulate)
}
```

**Role-Based Access Control**:
```go
package keeper

// Role represents user role
type Role uint8

const (
    RoleUser Role = iota
    RoleAdmin
    RoleModerator
)

// HasRole checks if account has role
func (k Keeper) HasRole(ctx sdk.Context, addr sdk.AccAddress, role Role) bool {
    accountRole := k.GetAccountRole(ctx, addr)
    return accountRole >= role
}

// RequireRole ensures account has required role
func (k Keeper) RequireRole(ctx sdk.Context, addr sdk.AccAddress, role Role) error {
    if !k.HasRole(ctx, addr, role) {
        return sdkerrors.Wrap(
            sdkerrors.ErrUnauthorized,
            fmt.Sprintf("account %s does not have required role", addr.String()),
        )
    }
    return nil
}

// In handler
func (k Keeper) AdminAction(ctx sdk.Context, msg types.MsgAdminAction) error {
    // Check admin role
    if err := k.RequireRole(ctx, msg.Signer, RoleAdmin); err != nil {
        return err
    }
    
    // Perform admin action
    return k.DoAdminAction(ctx, msg)
}
```

---

## Q3: How do you implement custom fee models and fee distribution?

**Answer**:

**Custom Fee Decorator**:
```go
package ante

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
    "github.com/cosmos/cosmos-sdk/x/auth/ante"
)

// CustomFeeDecorator implements custom fee logic
type CustomFeeDecorator struct {
    accountKeeper ante.AccountKeeper
    bankKeeper    BankKeeper
    feeKeeper     FeeKeeper
}

func NewCustomFeeDecorator(ak ante.AccountKeeper, bk BankKeeper, fk FeeKeeper) CustomFeeDecorator {
    return CustomFeeDecorator{
        accountKeeper: ak,
        bankKeeper:    bk,
        feeKeeper:     fk,
    }
}

func (cfd CustomFeeDecorator) AnteHandle(
    ctx sdk.Context,
    tx sdk.Tx,
    simulate bool,
    next sdk.AnteHandler,
) (sdk.Context, error) {
    feeTx, ok := tx.(sdk.FeeTx)
    if !ok {
        return ctx, sdkerrors.Wrap(sdkerrors.ErrTxDecode, "invalid transaction type")
    }
    
    // Get fee
    fee := feeTx.GetFee()
    
    // Calculate custom fee based on message type
    customFee := cfd.calculateCustomFee(ctx, tx)
    
    // Use higher of standard or custom fee
    if customFee.IsAllGT(fee) {
        fee = customFee
    }
    
    // Get payer
    payer := feeTx.FeePayer()
    if payer == nil {
        payer = feeTx.GetSigners()[0]
    }
    
    // Deduct fee
    if err := cfd.bankKeeper.SendCoinsFromAccountToModule(
        ctx,
        payer,
        types.FeeCollectorName,
        fee,
    ); err != nil {
        return ctx, sdkerrors.Wrapf(sdkerrors.ErrInsufficientFee, "insufficient fee: %s", err)
    }
    
    // Distribute fee
    cfd.feeKeeper.DistributeFee(ctx, fee, tx)
    
    return next(ctx, tx, simulate)
}

func (cfd CustomFeeDecorator) calculateCustomFee(ctx sdk.Context, tx sdk.Tx) sdk.Coins {
    // Calculate fee based on message types
    var totalFee sdk.Coins
    
    for _, msg := range tx.GetMsgs() {
        switch m := msg.(type) {
        case *types.MsgComplexOperation:
            // Higher fee for complex operations
            totalFee = totalFee.Add(sdk.NewCoin("stake", sdk.NewInt(1000)))
        case *types.MsgSimpleOperation:
            // Lower fee for simple operations
            totalFee = totalFee.Add(sdk.NewCoin("stake", sdk.NewInt(100)))
        default:
            // Default fee
            totalFee = totalFee.Add(sdk.NewCoin("stake", sdk.NewInt(500)))
        }
    }
    
    return totalFee
}
```

**Fee Distribution**:
```go
package keeper

// DistributeFee distributes fees to validators and treasury
func (k Keeper) DistributeFee(ctx sdk.Context, fee sdk.Coins, tx sdk.Tx) {
    // Get distribution parameters
    params := k.GetParams(ctx)
    
    // Calculate shares
    validatorShare := fee.MulInt(sdk.NewIntFromUint64(params.ValidatorFeeShare))
    treasuryShare := fee.MulInt(sdk.NewIntFromUint64(params.TreasuryFeeShare))
    
    // Distribute to validators
    k.distributeToValidators(ctx, validatorShare)
    
    // Send to treasury
    k.bankKeeper.SendCoinsFromModuleToModule(
        ctx,
        types.FeeCollectorName,
        types.TreasuryModuleName,
        treasuryShare,
    )
}
```

---

## Q4: How do you implement custom query pagination and filtering?

**Answer**:

**Pagination**:
```go
package keeper

import (
    "github.com/cosmos/cosmos-sdk/store/prefix"
    "github.com/cosmos/cosmos-sdk/types/query"
)

// QueryAllPosts with pagination
func (k Keeper) QueryAllPosts(
    ctx sdk.Context,
    req *types.QueryAllPostsRequest,
) (*types.QueryAllPostsResponse, error) {
    if req == nil {
        return nil, status.Error(codes.InvalidArgument, "invalid request")
    }
    
    var posts []types.Post
    store := ctx.KVStore(k.storeKey)
    postStore := prefix.NewStore(store, types.PostKeyPrefix)
    
    // Parse pagination
    pageRes, err := query.Paginate(postStore, req.Pagination, func(key []byte, value []byte) error {
        var post types.Post
        if err := k.cdc.Unmarshal(value, &post); err != nil {
            return err
        }
        
        // Apply filters
        if req.Filter != nil {
            if !k.matchesFilter(post, req.Filter) {
                return nil // Skip this post
            }
        }
        
        posts = append(posts, post)
        return nil
    })
    
    if err != nil {
        return nil, status.Error(codes.Internal, err.Error())
    }
    
    return &types.QueryAllPostsResponse{
        Posts:      posts,
        Pagination: pageRes,
    }, nil
}

// matchesFilter checks if post matches filter criteria
func (k Keeper) matchesFilter(post types.Post, filter *types.PostFilter) bool {
    if filter == nil {
        return true
    }
    
    // Filter by creator
    if filter.Creator != "" {
        if post.Creator != filter.Creator {
            return false
        }
    }
    
    // Filter by title contains
    if filter.TitleContains != "" {
        if !strings.Contains(strings.ToLower(post.Title), strings.ToLower(filter.TitleContains)) {
            return false
        }
    }
    
    // Filter by date range
    if filter.CreatedAfter != nil {
        if post.CreatedAt.Before(*filter.CreatedAfter) {
            return false
        }
    }
    
    if filter.CreatedBefore != nil {
        if post.CreatedAt.After(*filter.CreatedBefore) {
            return false
        }
    }
    
    return true
}
```

**Index-Based Queries**:
```go
// Create index for efficient queries
func (k Keeper) IndexPostsByCreator(ctx sdk.Context) {
    store := ctx.KVStore(k.storeKey)
    postStore := prefix.NewStore(store, types.PostKeyPrefix)
    
    iterator := sdk.KVStorePrefixIterator(postStore, []byte{})
    defer iterator.Close()
    
    for ; iterator.Valid(); iterator.Next() {
        var post types.Post
        k.cdc.MustUnmarshal(iterator.Value(), &post)
        
        // Store in creator index
        creatorKey := types.GetPostByCreatorKey(post.Creator, post.Id)
        store.Set(creatorKey, iterator.Key())
    }
}

// QueryPostsByCreator uses index
func (k Keeper) QueryPostsByCreator(
    ctx sdk.Context,
    creator string,
    pagination *query.PageRequest,
) ([]types.Post, *query.PageResponse, error) {
    store := ctx.KVStore(k.storeKey)
    creatorStore := prefix.NewStore(store, types.GetPostByCreatorKeyPrefix(creator))
    
    var posts []types.Post
    
    pageRes, err := query.Paginate(creatorStore, pagination, func(key []byte, value []byte) error {
        // Get actual post key from index
        postKey := value
        
        // Get post
        postBytes := store.Get(postKey)
        var post types.Post
        k.cdc.MustUnmarshal(postBytes, &post)
        posts = append(posts, post)
        return nil
    })
    
    return posts, pageRes, err
}
```

---

## Q5: How do you implement upgrade handlers and migration logic?

**Answer**:

**Upgrade Handler**:
```go
package app

import (
    "github.com/cosmos/cosmos-sdk/types/module"
    upgradetypes "github.com/cosmos/cosmos-sdk/x/upgrade/types"
)

// RegisterUpgradeHandlers registers upgrade handlers
func (app *App) RegisterUpgradeHandlers() {
    app.UpgradeKeeper.SetUpgradeHandler("v2", func(ctx sdk.Context, plan upgradetypes.Plan, vm module.VersionMap) (module.VersionMap, error) {
        // Perform migrations
        return app.mm.RunMigrations(ctx, app.configurator, vm)
    })
}

// Custom migration
func (app *App) migrateV1ToV2(ctx sdk.Context) error {
    // Migrate state
    store := ctx.KVStore(app.keys[types.StoreKey])
    
    // Example: Migrate old format to new format
    iterator := sdk.KVStorePrefixIterator(store, []byte("old_prefix"))
    defer iterator.Close()
    
    for ; iterator.Valid(); iterator.Next() {
        key := iterator.Key()
        value := iterator.Value()
        
        // Transform data
        newKey := migrateKey(key)
        newValue := migrateValue(value)
        
        // Store in new format
        store.Set(newKey, newValue)
        
        // Delete old key
        store.Delete(key)
    }
    
    return nil
}

// Version-specific upgrade
func (app *App) RegisterUpgradeHandlers() {
    app.UpgradeKeeper.SetUpgradeHandler("v2.0.0", func(ctx sdk.Context, plan upgradetypes.Plan, vm module.VersionMap) (module.VersionMap, error) {
        // Run module migrations
        return app.mm.RunMigrations(ctx, app.configurator, vm)
    })
    
    app.UpgradeKeeper.SetUpgradeHandler("v2.1.0", func(ctx sdk.Context, plan upgradetypes.Plan, vm module.VersionMap) (module.VersionMap, error) {
        // Custom migration logic
        if err := app.migrateV1ToV2(ctx); err != nil {
            return vm, err
        }
        
        // Continue with module migrations
        return app.mm.RunMigrations(ctx, app.configurator, vm)
    })
}
```

**Module Migration**:
```go
package keeper

// RegisterStoreDecoder registers migration decoders
func (k Keeper) RegisterStoreDecoder(store sdk.StoreDecoderRegistry) {
    store[types.StoreKey] = func(version uint64) func(sdk.KVStore) error {
        switch version {
        case 0:
            return migrateV0ToV1
        case 1:
            return migrateV1ToV2
        default:
            return nil
        }
    }
}

func migrateV0ToV1(store sdk.KVStore) error {
    // Migration logic
    return nil
}
```

---

## Q6: How do you implement event indexing and event queries?

**Answer**:

**Event Emission**:
```go
package keeper

// EmitOrderEvent emits order-related events
func (k Keeper) EmitOrderEvent(
    ctx sdk.Context,
    eventType string,
    orderId uint64,
    attributes map[string]string,
) {
    event := sdk.NewEvent(
        eventType,
        sdk.NewAttribute(types.AttributeKeyOrderId, fmt.Sprintf("%d", orderId)),
    )
    
    // Add custom attributes
    for key, value := range attributes {
        event = event.AppendAttributes(sdk.NewAttribute(key, value))
    }
    
    ctx.EventManager().EmitEvent(event)
}

// EmitTypedEvent emits typed events (better for indexing)
func (k Keeper) EmitTypedEvent(ctx sdk.Context, event proto.Message) {
    ctx.EventManager().EmitTypedEvent(event)
}
```

**Event Querying**:
```go
package keeper

// QueryEvents queries events by type and attributes
func (k Keeper) QueryEvents(
    ctx sdk.Context,
    eventType string,
    attributes map[string]string,
    pagination *query.PageRequest,
) (*types.QueryEventsResponse, error) {
    // Query events from event store
    events := k.eventStore.QueryEvents(
        ctx,
        eventType,
        attributes,
        pagination,
    )
    
    return &types.QueryEventsResponse{
        Events:     events,
        Pagination: pagination,
    }, nil
}
```

**Event Indexing**:
```go
// Index events for efficient querying
type EventIndexer struct {
    store sdk.KVStore
}

func (ei EventIndexer) IndexEvent(event sdk.Event) {
    // Index by event type
    typeKey := types.GetEventTypeKey(event.Type)
    ei.store.Set(typeKey, []byte{1})
    
    // Index by attributes
    for _, attr := range event.Attributes {
        attrKey := types.GetEventAttributeKey(event.Type, attr.Key, attr.Value)
        ei.store.Set(attrKey, []byte{1})
    }
}
```

---

## Q7: How do you implement custom transaction validation and ordering?

**Answer**:

**Custom Mempool**:
```go
package mempool

import (
    "github.com/cosmos/cosmos-sdk/types"
    "github.com/cosmos/cosmos-sdk/types/mempool"
)

// PriorityMempool implements priority-based ordering
type PriorityMempool struct {
    transactions []sdk.Tx
    priorityFunc func(sdk.Tx) int64
}

func NewPriorityMempool(priorityFunc func(sdk.Tx) int64) *PriorityMempool {
    return &PriorityMempool{
        transactions: make([]sdk.Tx, 0),
        priorityFunc: priorityFunc,
    }
}

func (mp *PriorityMempool) Insert(ctx sdk.Context, tx sdk.Tx) error {
    // Calculate priority
    priority := mp.priorityFunc(tx)
    
    // Insert in sorted order
    inserted := false
    for i, existingTx := range mp.transactions {
        if mp.priorityFunc(existingTx) < priority {
            // Insert here
            mp.transactions = append(
                mp.transactions[:i],
                append([]sdk.Tx{tx}, mp.transactions[i:]...)...,
            )
            inserted = true
            break
        }
    }
    
    if !inserted {
        mp.transactions = append(mp.transactions, tx)
    }
    
    return nil
}

func (mp *PriorityMempool) Select(ctx sdk.Context, maxTxs [][]byte) [][]byte {
    selected := make([][]byte, 0, len(maxTxs))
    
    for i, tx := range mp.transactions {
        if i >= len(maxTxs) {
            break
        }
        selected = append(selected, tx.GetSigners()[0].Bytes())
    }
    
    return selected
}

// Custom validation
func (mp *PriorityMempool) ValidateTx(ctx sdk.Context, tx sdk.Tx) error {
    // Custom validation logic
    if err := mp.validateNonce(ctx, tx); err != nil {
        return err
    }
    
    if err := mp.validateGas(ctx, tx); err != nil {
        return err
    }
    
    return nil
}
```

---

## Q8: How do you implement inter-module communication and hooks?

**Answer**:

**Hooks Interface**:
```go
package types

// Hooks defines module hooks
type Hooks interface {
    AfterOrderCreated(ctx sdk.Context, orderId uint64)
    AfterOrderCancelled(ctx sdk.Context, orderId uint64)
    AfterOrderCompleted(ctx sdk.Context, orderId uint64)
}

var _ Hooks = Hooks{}

// Hooks is a wrapper struct
type Hooks struct{}

func (Hooks) AfterOrderCreated(ctx sdk.Context, orderId uint64)    {}
func (Hooks) AfterOrderCancelled(ctx sdk.Context, orderId uint64)  {}
func (Hooks) AfterOrderCompleted(ctx sdk.Context, orderId uint64)  {}
```

**Keeper with Hooks**:
```go
package keeper

type Keeper struct {
    storeKey sdk.StoreKey
    hooks    types.Hooks
}

// SetHooks sets hooks
func (k *Keeper) SetHooks(hooks types.Hooks) {
    if k.hooks != nil {
        panic("cannot set hooks twice")
    }
    k.hooks = hooks
}

// CreateOrder with hooks
func (k Keeper) CreateOrder(ctx sdk.Context, order types.Order) error {
    // Create order
    k.SetOrder(ctx, order)
    
    // Call hooks
    if k.hooks != nil {
        k.hooks.AfterOrderCreated(ctx, order.Id)
    }
    
    return nil
}
```

**Module Using Hooks**:
```go
package keeper

type NotificationKeeper struct {
    orderKeeper types.OrderKeeper
}

// Implement hooks
func (nk NotificationKeeper) AfterOrderCreated(ctx sdk.Context, orderId uint64) {
    // Send notification
    order, _ := nk.orderKeeper.GetOrder(ctx, orderId)
    nk.SendNotification(ctx, order.Buyer, "Order created")
}

// Register hooks
func (app *App) SetupHooks() {
    orderKeeper := app.OrderKeeper
    notificationKeeper := app.NotificationKeeper
    
    orderKeeper.SetHooks(notificationKeeper)
}
```

---

## Q9: How do you implement custom consensus parameters and governance?

**Answer**:

**Custom Parameters**:
```go
package types

// Params defines module parameters
type Params struct {
    MaxOrderAmount    sdk.Int
    MinOrderAmount    sdk.Int
    OrderTimeout      time.Duration
    FeePercentage     sdk.Dec
}

// DefaultParams returns default parameters
func DefaultParams() Params {
    return Params{
        MaxOrderAmount: sdk.NewInt(1000000),
        MinOrderAmount: sdk.NewInt(100),
        OrderTimeout:  24 * time.Hour,
        FeePercentage: sdk.NewDecWithPrec(1, 2), // 1%
    }
}
```

**Parameter Management**:
```go
package keeper

// GetParams gets parameters
func (k Keeper) GetParams(ctx sdk.Context) types.Params {
    store := ctx.KVStore(k.storeKey)
    bz := store.Get(types.ParamsKey)
    if bz == nil {
        return types.DefaultParams()
    }
    
    var params types.Params
    k.cdc.MustUnmarshal(bz, &params)
    return params
}

// SetParams sets parameters
func (k Keeper) SetParams(ctx sdk.Context, params types.Params) {
    store := ctx.KVStore(k.storeKey)
    bz := k.cdc.MustMarshal(&params)
    store.Set(types.ParamsKey, bz)
}
```

**Governance Proposal**:
```go
package types

// UpdateParamsProposal is a governance proposal
type UpdateParamsProposal struct {
    Title       string
    Description string
    Params      Params
}

// ValidateBasic validates proposal
func (p UpdateParamsProposal) ValidateBasic() error {
    if p.Title == "" {
        return sdkerrors.Wrap(sdkerrors.ErrInvalidRequest, "title cannot be empty")
    }
    if p.Description == "" {
        return sdkerrors.Wrap(sdkerrors.ErrInvalidRequest, "description cannot be empty")
    }
    return p.Params.Validate()
}

// Handler for proposal
func (k Keeper) HandleUpdateParamsProposal(ctx sdk.Context, p UpdateParamsProposal) error {
    k.SetParams(ctx, p.Params)
    return nil
}
```

---

## Q10: How do you optimize gas usage in Cosmos SDK transactions?

**Answer**:

**Gas Optimization Techniques**:

1. **Use uint64 instead of string for IDs**:
```go
// Bad: Uses more gas
type Post struct {
    Id      string  // Expensive
    Title   string
}

// Good: Uses less gas
type Post struct {
    Id      uint64  // Cheap
    Title   string
}
```

2. **Pack structs efficiently**:
```go
// Bad: Wastes storage
type Order struct {
    Id      uint64  // 8 bytes
    Status  uint8   // 1 byte (but takes 8 bytes due to alignment)
    Amount  sdk.Int
}

// Good: Packed efficiently
type Order struct {
    Id      uint64
    Status  uint8
    _       [7]byte // Padding
    Amount  sdk.Int
}
```

3. **Use iterator instead of loading all**:
```go
// Bad: Loads all into memory
func (k Keeper) GetAllOrders(ctx sdk.Context) []Order {
    store := ctx.KVStore(k.storeKey)
    iterator := sdk.KVStorePrefixIterator(store, []byte{})
    defer iterator.Close()
    
    var orders []Order
    for ; iterator.Valid(); iterator.Next() {
        var order Order
        k.cdc.MustUnmarshal(iterator.Value(), &order)
        orders = append(orders, order) // Expensive append
    }
    return orders
}

// Good: Process one at a time
func (k Keeper) ProcessOrders(ctx sdk.Context, processor func(Order) error) error {
    store := ctx.KVStore(k.storeKey)
    iterator := sdk.KVStorePrefixIterator(store, []byte{})
    defer iterator.Close()
    
    for ; iterator.Valid(); iterator.Next() {
        var order Order
        k.cdc.MustUnmarshal(iterator.Value(), &order)
        if err := processor(order); err != nil {
            return err
        }
    }
    return nil
}
```

4. **Cache expensive computations**:
```go
// Cache computed values
type Keeper struct {
    storeKey sdk.StoreKey
    cache    map[string]interface{}
}

func (k Keeper) GetExpensiveValue(ctx sdk.Context, key string) sdk.Int {
    // Check cache first
    if cached, exists := k.cache[key]; exists {
        return cached.(sdk.Int)
    }
    
    // Compute
    value := k.computeExpensiveValue(ctx, key)
    
    // Cache
    k.cache[key] = value
    
    return value
}
```

5. **Use events instead of storing redundant data**:
```go
// Bad: Store redundant data
type Order struct {
    Id          uint64
    CreatedAt   time.Time
    UpdatedAt   time.Time  // Redundant if we can derive from events
}

// Good: Use events for history
func (k Keeper) CreateOrder(ctx sdk.Context, order Order) {
    k.SetOrder(ctx, order)
    
    // Emit event (cheaper than storing)
    ctx.EventManager().EmitEvent(
        sdk.NewEvent(
            types.EventTypeOrderCreated,
            sdk.NewAttribute(types.AttributeKeyOrderId, fmt.Sprintf("%d", order.Id)),
            sdk.NewAttribute(types.AttributeKeyTimestamp, ctx.BlockTime().String()),
        ),
    )
}
```

---

