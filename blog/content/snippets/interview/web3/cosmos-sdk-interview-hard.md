---
title: "Cosmos SDK Interview Questions - Hard"
date: 2025-12-14
tags: ["cosmos", "cosmos-sdk", "interview", "hard", "blockchain", "golang"]
---

Hard-level Cosmos SDK interview questions covering advanced SDK internals, performance optimization, and complex module design.

## Q1: How do you implement a custom ABCI application with advanced state management?

**Answer**:

**Custom BaseApp**:
```go
package app

import (
    "github.com/cosmos/cosmos-sdk/baseapp"
    "github.com/cosmos/cosmos-sdk/types"
)

type CustomApp struct {
    *baseapp.BaseApp
    customState *CustomState
}

// CustomState manages application state
type CustomState struct {
    snapshots   *SnapshotManager
    checkpoints *CheckpointManager
    cache       *StateCache
}

// BeginBlock with custom logic
func (app *CustomApp) BeginBlock(req abci.RequestBeginBlock) abci.ResponseBeginBlock {
    // Custom pre-processing
    app.customState.cache.BeginBlock()
    
    // Standard BeginBlock
    res := app.BaseApp.BeginBlock(req)
    
    // Custom post-processing
    app.customState.checkpoints.CreateCheckpoint(req.Height)
    
    return res
}

// DeliverTx with custom validation
func (app *CustomApp) DeliverTx(req abci.RequestDeliverTx) abci.ResponseDeliverTx {
    ctx := app.NewContext(false, req.Header)
    
    // Custom validation
    if err := app.customState.ValidateTx(ctx, req.Tx); err != nil {
        return abci.ResponseDeliverTx{
            Code: uint32(sdkerrors.ErrInvalidRequest.ABCICode()),
            Log:  err.Error(),
        }
    }
    
    // Standard DeliverTx
    res := app.BaseApp.DeliverTx(req)
    
    // Custom post-processing
    app.customState.cache.CacheTx(req.Tx, res)
    
    return res
}

// Commit with snapshot creation
func (app *CustomApp) Commit() abci.ResponseCommit {
    // Standard commit
    res := app.BaseApp.Commit()
    
    // Create snapshot periodically
    if app.customState.snapshots.ShouldCreateSnapshot(app.LastBlockHeight()) {
        app.customState.snapshots.CreateSnapshot(app.LastBlockHeight())
    }
    
    return res
}
```

**State Snapshotting**:
```go
package state

// SnapshotManager manages state snapshots
type SnapshotManager struct {
    store    sdk.KVStore
    interval uint64
}

func (sm *SnapshotManager) CreateSnapshot(height int64) error {
    // Create snapshot of all stores
    snapshot := &Snapshot{
        Height: height,
        Stores: make(map[string][]byte),
    }
    
    // Snapshot each module store
    for _, storeKey := range sm.storeKeys {
        store := sm.getStore(storeKey)
        snapshot.Stores[storeKey.Name()] = sm.snapshotStore(store)
    }
    
    // Persist snapshot
    return sm.persistSnapshot(snapshot)
}

func (sm *SnapshotManager) RestoreSnapshot(height int64) error {
    snapshot, err := sm.loadSnapshot(height)
    if err != nil {
        return err
    }
    
    // Restore each store
    for storeKey, data := range snapshot.Stores {
        store := sm.getStore(storeKey)
        sm.restoreStore(store, data)
    }
    
    return nil
}
```

---

## Q2: How do you implement advanced transaction ordering and MEV protection?

**Answer**:

**MEV-Resistant Mempool**:
```go
package mempool

// EncryptedMempool encrypts transactions to prevent front-running
type EncryptedMempool struct {
    transactions map[string]*EncryptedTx
    decryptionKey []byte
}

type EncryptedTx struct {
    EncryptedData []byte
    Commitment    []byte
    Timestamp     time.Time
}

// Insert encrypts transaction before adding
func (mp *EncryptedMempool) Insert(ctx sdk.Context, tx sdk.Tx) error {
    // Encrypt transaction
    encrypted, err := mp.encryptTx(tx)
    if err != nil {
        return err
    }
    
    // Create commitment
    commitment := mp.createCommitment(tx)
    
    // Store encrypted
    mp.transactions[string(tx.Hash())] = &EncryptedTx{
        EncryptedData: encrypted,
        Commitment:    commitment,
        Timestamp:     ctx.BlockTime(),
    }
    
    return nil
}

// Reveal decrypts transactions at block proposal time
func (mp *EncryptedMempool) Reveal(ctx sdk.Context, txHashes [][]byte) ([]sdk.Tx, error) {
    var txs []sdk.Tx
    
    for _, hash := range txHashes {
        encryptedTx, exists := mp.transactions[string(hash)]
        if !exists {
            continue
        }
        
        // Verify commitment
        if !mp.verifyCommitment(encryptedTx.Commitment, encryptedTx.EncryptedData) {
            return nil, sdkerrors.Wrap(sdkerrors.ErrInvalidRequest, "invalid commitment")
        }
        
        // Decrypt
        tx, err := mp.decryptTx(encryptedTx.EncryptedData)
        if err != nil {
            return nil, err
        }
        
        txs = append(txs, tx)
    }
    
    return txs, nil
}
```

**Fair Ordering**:
```go
// FairOrderingMempool implements fair transaction ordering
type FairOrderingMempool struct {
    transactions []*PrioritizedTx
    maxPriority  int64
}

type PrioritizedTx struct {
    Tx       sdk.Tx
    Priority int64
    Nonce    uint64
    Sender   sdk.AccAddress
}

// Insert with fair priority calculation
func (mp *FairOrderingMempool) Insert(ctx sdk.Context, tx sdk.Tx) error {
    // Calculate priority based on fee and time
    priority := mp.calculatePriority(ctx, tx)
    
    // Get sender nonce
    sender := tx.GetSigners()[0]
    nonce := mp.getNonce(ctx, sender)
    
    prioritized := &PrioritizedTx{
        Tx:       tx,
        Priority: priority,
        Nonce:    nonce,
        Sender:   sender,
    }
    
    // Insert maintaining order
    mp.insertSorted(prioritized)
    
    return nil
}

// calculatePriority calculates fair priority
func (mp *FairOrderingMempool) calculatePriority(ctx sdk.Context, tx sdk.Tx) int64 {
    feeTx := tx.(sdk.FeeTx)
    fee := feeTx.GetFee()
    
    // Base priority from fee
    priority := fee.AmountOf("stake").Int64()
    
    // Reduce priority for recent transactions (prevent front-running)
    age := ctx.BlockTime().Unix() - mp.getTxTimestamp(tx).Unix()
    if age < 60 { // Less than 1 minute
        priority = priority / 2
    }
    
    return priority
}
```

---

## Q3: How do you implement cross-chain communication and IBC integration?

**Answer**:

**IBC Module Integration**:
```go
package keeper

import (
    "github.com/cosmos/ibc-go/v3/modules/core/keeper"
    channeltypes "github.com/cosmos/ibc-go/v3/modules/core/04-channel/types"
)

type Keeper struct {
    storeKey    sdk.StoreKey
    ibcKeeper   *ibckeeper.Keeper
    channelKeeper channelkeeper.Keeper
}

// SendPacket sends packet via IBC
func (k Keeper) SendPacket(
    ctx sdk.Context,
    sourcePort,
    sourceChannel string,
    packetData []byte,
    timeoutHeight clienttypes.Height,
    timeoutTimestamp uint64,
) error {
    // Create packet
    packet := channeltypes.NewPacket(
        packetData,
        k.getNextSequenceSend(ctx, sourcePort, sourceChannel),
        sourcePort,
        sourceChannel,
        k.getCounterpartyPort(sourcePort),
        k.getCounterpartyChannel(sourceChannel),
        timeoutHeight,
        timeoutTimestamp,
    )
    
    // Send via IBC
    return k.channelKeeper.SendPacket(ctx, packet)
}

// OnRecvPacket handles received IBC packet
func (k Keeper) OnRecvPacket(
    ctx sdk.Context,
    packet channeltypes.Packet,
    relayer sdk.AccAddress,
) (*sdk.Result, error) {
    // Unmarshal packet data
    var data MyPacketData
    if err := k.cdc.Unmarshal(packet.GetData(), &data); err != nil {
        return nil, err
    }
    
    // Process packet
    switch data.Type {
    case PacketTypeOrder:
        return k.handleOrderPacket(ctx, data, relayer)
    case PacketTypePayment:
        return k.handlePaymentPacket(ctx, data, relayer)
    default:
        return nil, sdkerrors.Wrap(sdkerrors.ErrUnknownRequest, "unknown packet type")
    }
}

// AcknowledgePacket handles packet acknowledgment
func (k Keeper) OnAcknowledgementPacket(
    ctx sdk.Context,
    packet channeltypes.Packet,
    acknowledgement []byte,
    relayer sdk.AccAddress,
) (*sdk.Result, error) {
    // Parse acknowledgement
    var ack channeltypes.Acknowledgement
    if err := k.cdc.Unmarshal(acknowledgement, &ack); err != nil {
        return nil, err
    }
    
    if ack.Success() {
        // Handle successful acknowledgement
        return k.handleSuccessAck(ctx, packet, relayer)
    } else {
        // Handle failure
        return k.handleFailureAck(ctx, packet, ack.Error(), relayer)
    }
}
```

**Cross-Chain State Sync**:
```go
// CrossChainStateSync synchronizes state across chains
type CrossChainStateSync struct {
    keeper      Keeper
    ibcKeeper   *ibckeeper.Keeper
}

// SyncState syncs state to target chain
func (ccs *CrossChainStateSync) SyncState(
    ctx sdk.Context,
    targetChain string,
    stateKey string,
) error {
    // Get state
    state := ccs.keeper.GetState(ctx, stateKey)
    
    // Create sync packet
    packetData := SyncPacketData{
        Type:   PacketTypeStateSync,
        Key:    stateKey,
        State:  state,
        Height: ctx.BlockHeight(),
    }
    
    // Send via IBC
    return ccs.keeper.SendPacket(
        ctx,
        "transfer",
        ccs.getChannel(targetChain),
        packetData,
        clienttypes.Height{},
        0,
    )
}
```

---

## Q4: How do you implement advanced caching and state optimization?

**Answer**:

**Multi-Level Cache**:
```go
package cache

// MultiLevelCache implements L1/L2 caching
type MultiLevelCache struct {
    l1Cache *sync.Map // In-memory
    l2Cache *LRUCache // Persistent
    store   sdk.KVStore
}

func (mlc *MultiLevelCache) Get(ctx sdk.Context, key []byte) ([]byte, error) {
    // Check L1 cache
    if value, ok := mlc.l1Cache.Load(string(key)); ok {
        return value.([]byte), nil
    }
    
    // Check L2 cache
    if value, ok := mlc.l2Cache.Get(string(key)); ok {
        // Promote to L1
        mlc.l1Cache.Store(string(key), value)
        return value.([]byte), nil
    }
    
    // Load from store
    value := mlc.store.Get(key)
    if value == nil {
        return nil, sdkerrors.Wrap(sdkerrors.ErrKeyNotFound, "key not found")
    }
    
    // Cache in L2
    mlc.l2Cache.Set(string(key), value)
    
    // Cache in L1
    mlc.l1Cache.Store(string(key), value)
    
    return value, nil
}

// Write-through cache
func (mlc *MultiLevelCache) Set(ctx sdk.Context, key, value []byte) {
    // Write to store first
    mlc.store.Set(key, value)
    
    // Update L2 cache
    mlc.l2Cache.Set(string(key), value)
    
    // Update L1 cache
    mlc.l1Cache.Store(string(key), value)
}
```

**State Compression**:
```go
// CompressedStore compresses values before storing
type CompressedStore struct {
    store     sdk.KVStore
    compressor *Compressor
}

func (cs *CompressedStore) Set(key, value []byte) {
    // Compress value
    compressed := cs.compressor.Compress(value)
    
    // Store compressed
    cs.store.Set(key, compressed)
}

func (cs *CompressedStore) Get(key []byte) []byte {
    // Get compressed
    compressed := cs.store.Get(key)
    if compressed == nil {
        return nil
    }
    
    // Decompress
    return cs.compressor.Decompress(compressed)
}
```

---

## Q5: How do you implement advanced validator set management and slashing?

**Answer**:

**Custom Validator Set**:
```go
package keeper

// ValidatorSetManager manages custom validator set
type ValidatorSetManager struct {
    keeper Keeper
}

// UpdateValidatorSet updates validators based on custom rules
func (vsm *ValidatorSetManager) UpdateValidatorSet(ctx sdk.Context) error {
    // Get current validators
    validators := vsm.keeper.GetAllValidators(ctx)
    
    // Calculate new set based on:
    // - Staking power
    // - Performance metrics
    // - Reputation
    newSet := vsm.calculateNewSet(ctx, validators)
    
    // Apply changes
    return vsm.applyValidatorSet(ctx, newSet)
}

// calculateNewSet calculates optimal validator set
func (vsm *ValidatorSetManager) calculateNewSet(
    ctx sdk.Context,
    validators []Validator,
) []Validator {
    // Score validators
    scored := make([]ScoredValidator, len(validators))
    for i, val := range validators {
        scored[i] = ScoredValidator{
            Validator: val,
            Score:     vsm.calculateScore(ctx, val),
        }
    }
    
    // Sort by score
    sort.Slice(scored, func(i, j int) bool {
        return scored[i].Score > scored[j].Score
    })
    
    // Select top N
    maxValidators := vsm.keeper.GetMaxValidators(ctx)
    selected := make([]Validator, 0, maxValidators)
    for i := 0; i < len(scored) && i < maxValidators; i++ {
        selected = append(selected, scored[i].Validator)
    }
    
    return selected
}

// calculateScore calculates validator score
func (vsm *ValidatorSetManager) calculateScore(
    ctx sdk.Context,
    val Validator,
) sdk.Dec {
    // Base score from staking power
    baseScore := sdk.NewDecFromInt(val.Tokens)
    
    // Performance multiplier
    uptime := vsm.keeper.GetUptime(ctx, val.OperatorAddress)
    performanceMultiplier := sdk.NewDec(1).Add(uptime.Mul(sdk.NewDecWithPrec(1, 1)))
    
    // Reputation multiplier
    reputation := vsm.keeper.GetReputation(ctx, val.OperatorAddress)
    reputationMultiplier := sdk.NewDec(1).Add(reputation.Mul(sdk.NewDecWithPrec(5, 2)))
    
    // Final score
    return baseScore.Mul(performanceMultiplier).Mul(reputationMultiplier)
}
```

**Advanced Slashing**:
```go
// SlashingManager implements advanced slashing
type SlashingManager struct {
    keeper Keeper
}

// SlashValidator slashes validator with custom logic
func (sm *SlashingManager) SlashValidator(
    ctx sdk.Context,
    validator sdk.ValAddress,
    infractionType InfractionType,
    evidence Evidence,
) error {
    // Get validator
    val, found := sm.keeper.GetValidator(ctx, validator)
    if !found {
        return sdkerrors.Wrap(sdkerrors.ErrNotFound, "validator not found")
    }
    
    // Calculate slash amount based on:
    // - Infraction type
    // - Validator history
    // - Evidence strength
    slashAmount := sm.calculateSlashAmount(ctx, val, infractionType, evidence)
    
    // Apply slash
    val.Tokens = val.Tokens.Sub(slashAmount)
    sm.keeper.SetValidator(ctx, val)
    
    // Distribute slashed tokens
    sm.distributeSlashedTokens(ctx, slashAmount, evidence)
    
    // Update reputation
    sm.keeper.DecreaseReputation(ctx, validator, infractionType)
    
    // Emit event
    ctx.EventManager().EmitEvent(
        sdk.NewEvent(
            types.EventTypeSlash,
            sdk.NewAttribute(types.AttributeKeyValidator, validator.String()),
            sdk.NewAttribute(types.AttributeKeySlashAmount, slashAmount.String()),
            sdk.NewAttribute(types.AttributeKeyInfractionType, string(infractionType)),
        ),
    )
    
    return nil
}
```

---

## Q6: How do you implement parallel transaction processing?

**Answer**:

**Parallel Execution**:
```go
package execution

// ParallelExecutor executes transactions in parallel
type ParallelExecutor struct {
    maxWorkers int
    semaphore  chan struct{}
}

func NewParallelExecutor(maxWorkers int) *ParallelExecutor {
    return &ParallelExecutor{
        maxWorkers: maxWorkers,
        semaphore:  make(chan struct{}, maxWorkers),
    }
}

// ExecuteBatch executes transactions in parallel
func (pe *ParallelExecutor) ExecuteBatch(
    ctx sdk.Context,
    txs []sdk.Tx,
) ([]sdk.Result, []error) {
    results := make([]sdk.Result, len(txs))
    errors := make([]error, len(txs))
    
    var wg sync.WaitGroup
    
    for i, tx := range txs {
        wg.Add(1)
        
        go func(idx int, transaction sdk.Tx) {
            defer wg.Done()
            
            // Acquire semaphore
            pe.semaphore <- struct{}{}
            defer func() { <-pe.semaphore }()
            
            // Execute transaction
            result, err := pe.executeTx(ctx, transaction)
            results[idx] = result
            errors[idx] = err
        }(i, tx)
    }
    
    wg.Wait()
    
    return results, errors
}

// executeTx executes single transaction
func (pe *ParallelExecutor) executeTx(ctx sdk.Context, tx sdk.Tx) (sdk.Result, error) {
    // Create isolated context for parallel execution
    isolatedCtx := ctx.WithIsolated(true)
    
    // Execute
    return pe.handler.DeliverTx(isolatedCtx, tx)
}
```

**Dependency Analysis**:
```go
// DependencyAnalyzer analyzes transaction dependencies
type DependencyAnalyzer struct{}

// AnalyzeDependencies finds transaction dependencies
func (da *DependencyAnalyzer) AnalyzeDependencies(txs []sdk.Tx) [][]int {
    graph := make(map[int][]int)
    
    // Build dependency graph
    for i, tx := range txs {
        deps := da.findDependencies(tx, txs)
        graph[i] = deps
    }
    
    // Topological sort
    return da.topologicalSort(graph)
}

// findDependencies finds transactions this tx depends on
func (da *DependencyAnalyzer) findDependencies(tx sdk.Tx, allTxs []sdk.Tx) []int {
    var deps []int
    
    // Check if tx reads/writes same accounts
    txAccounts := da.getAccounts(tx)
    
    for i, otherTx := range allTxs {
        if i == 0 {
            continue
        }
        
        otherAccounts := da.getAccounts(otherTx)
        
        // Check for overlap
        if da.hasOverlap(txAccounts, otherAccounts) {
            deps = append(deps, i)
        }
    }
    
    return deps
}
```

---

## Q7: How do you implement advanced event processing and indexing?

**Answer**:

**Event Indexer**:
```go
package indexing

// EventIndexer indexes events for efficient querying
type EventIndexer struct {
    store sdk.KVStore
}

// IndexEvent indexes an event
func (ei *EventIndexer) IndexEvent(ctx sdk.Context, event sdk.Event) error {
    // Index by type
    typeKey := ei.getTypeKey(event.Type)
    ei.addToIndex(typeKey, event)
    
    // Index by each attribute
    for _, attr := range event.Attributes {
        attrKey := ei.getAttributeKey(event.Type, attr.Key, attr.Value)
        ei.addToIndex(attrKey, event)
    }
    
    // Index by time range
    timeKey := ei.getTimeKey(ctx.BlockTime(), event.Type)
    ei.addToIndex(timeKey, event)
    
    return nil
}

// QueryEvents queries events with filters
func (ei *EventIndexer) QueryEvents(
    ctx sdk.Context,
    filters []EventFilter,
    pagination *query.PageRequest,
) ([]sdk.Event, *query.PageResponse, error) {
    // Build query plan
    plan := ei.buildQueryPlan(filters)
    
    // Execute query
    events, err := ei.executeQuery(ctx, plan, pagination)
    if err != nil {
        return nil, nil, err
    }
    
    return events, pagination, nil
}

// buildQueryPlan optimizes query execution
func (ei *EventIndexer) buildQueryPlan(filters []EventFilter) *QueryPlan {
    plan := &QueryPlan{}
    
    // Find most selective filter
    mostSelective := ei.findMostSelective(filters)
    plan.PrimaryFilter = mostSelective
    
    // Add secondary filters
    for _, filter := range filters {
        if filter != mostSelective {
            plan.SecondaryFilters = append(plan.SecondaryFilters, filter)
        }
    }
    
    return plan
}
```

---

## Q8: How do you implement advanced governance with quadratic voting?

**Answer**:

**Quadratic Voting**:
```go
package governance

// QuadraticVoting implements quadratic voting
type QuadraticVoting struct {
    keeper Keeper
}

// Vote with quadratic weighting
func (qv *QuadraticVoting) Vote(
    ctx sdk.Context,
    proposalId uint64,
    voter sdk.AccAddress,
    option VoteOption,
    votingPower sdk.Int,
) error {
    // Calculate cost (quadratic)
    cost := qv.calculateCost(votingPower)
    
    // Check if voter has enough tokens
    balance := qv.keeper.GetBalance(ctx, voter)
    if balance.AmountOf("stake").LT(cost) {
        return sdkerrors.Wrap(sdkerrors.ErrInsufficientFunds, "insufficient voting power")
    }
    
    // Deduct cost
    err := qv.keeper.SendCoinsFromAccountToModule(
        ctx,
        voter,
        types.GovernanceModuleName,
        sdk.NewCoins(sdk.NewCoin("stake", cost)),
    )
    if err != nil {
        return err
    }
    
    // Record vote
    vote := types.Vote{
        ProposalId:  proposalId,
        Voter:       voter.String(),
        Option:      option,
        VotingPower: votingPower,
        Cost:        cost,
    }
    
    qv.keeper.SetVote(ctx, vote)
    
    // Update proposal tally
    qv.keeper.UpdateTally(ctx, proposalId, option, votingPower)
    
    return nil
}

// calculateCost calculates quadratic cost
func (qv *QuadraticVoting) calculateCost(votingPower sdk.Int) sdk.Int {
    // Cost = votingPower^2
    return votingPower.Mul(votingPower)
}
```

---

## Q9: How do you implement state machine replication and recovery?

**Answer**:

**State Replication**:
```go
package replication

// StateReplicator replicates state across nodes
type StateReplicator struct {
    keeper Keeper
    peers  []Peer
}

// ReplicateState replicates state to peers
func (sr *StateReplicator) ReplicateState(ctx sdk.Context) error {
    // Get current state
    state := sr.getStateSnapshot(ctx)
    
    // Replicate to all peers
    var wg sync.WaitGroup
    errors := make(chan error, len(sr.peers))
    
    for _, peer := range sr.peers {
        wg.Add(1)
        go func(p Peer) {
            defer wg.Done()
            if err := p.SendState(state); err != nil {
                errors <- err
            }
        }(peer)
    }
    
    wg.Wait()
    close(errors)
    
    // Check for errors
    for err := range errors {
        if err != nil {
            return err
        }
    }
    
    return nil
}

// RecoverState recovers state from peers
func (sr *StateReplicator) RecoverState(ctx sdk.Context, targetHeight int64) error {
    // Request state from peers
    states := make([]StateSnapshot, 0)
    
    for _, peer := range sr.peers {
        state, err := peer.RequestState(targetHeight)
        if err != nil {
            continue
        }
        states = append(states, state)
    }
    
    // Verify and merge states
    verifiedState := sr.verifyAndMerge(states)
    
    // Restore state
    return sr.restoreState(ctx, verifiedState)
}
```

---

## Q10: How do you optimize Cosmos SDK application performance at scale?

**Answer**:

**Performance Optimization Strategies**:

1. **Connection Pooling**:
```go
// ConnectionPool manages database connections
type ConnectionPool struct {
    pool *sql.DB
    maxConnections int
}

func NewConnectionPool(maxConn int) *ConnectionPool {
    db, _ := sql.Open("postgres", "...")
    db.SetMaxOpenConns(maxConn)
    db.SetMaxIdleConns(maxConn / 2)
    return &ConnectionPool{pool: db, maxConnections: maxConn}
}
```

2. **Batch Processing**:
```go
// BatchProcessor processes operations in batches
func (k Keeper) BatchUpdate(ctx sdk.Context, updates []Update) error {
    batch := k.store.NewBatch()
    defer batch.Close()
    
    for _, update := range updates {
        batch.Set(update.Key, update.Value)
    }
    
    return batch.Write()
}
```

3. **Async Operations**:
```go
// AsyncProcessor processes operations asynchronously
type AsyncProcessor struct {
    queue chan Operation
    workers int
}

func (ap *AsyncProcessor) ProcessAsync(op Operation) {
    ap.queue <- op
}

func (ap *AsyncProcessor) worker() {
    for op := range ap.queue {
        ap.process(op)
    }
}
```

4. **State Pruning**:
```go
// StatePruner prunes old state
func (k Keeper) PruneState(ctx sdk.Context, keepHeight int64) error {
    currentHeight := ctx.BlockHeight()
    
    if currentHeight-keepHeight < 100 {
        return nil // Don't prune recent state
    }
    
    // Prune old state
    return k.pruneOldState(ctx, currentHeight-keepHeight)
}
```

---

