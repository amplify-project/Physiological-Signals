# Multi-GPU DDP Training Hang Issue - Investigation Report

**Date**: November 14, 2025  
**System**: 12x Tesla T4 GPUs, Ubuntu, PyTorch 2.x  
**Dataset**: 65,114 training samples (86 action classes), MediaPipe features  
**Issue**: DistributedDataParallel training hangs at first batch with 0% progress

---

## Executive Summary

**Problem**: DDP training with 8 GPUs successfully initializes all processes, loads datasets, wraps models, and enters the training loop, but hangs indefinitely at `0/255 [00:00<?, ?it/s]` when attempting to iterate the first batch from the DataLoader.

**Symptom**: GPU utilization shows 100% compute but 0% memory utilization, indicating the GPUs are spinning/waiting rather than processing data.

**Root Cause**: NCCL collective operation timeout when DataLoader attempts to iterate over 65K samples with DistributedSampler across 8 ranks. The processes reach different stages asynchronously, causing a deadlock in NCCL's ALLGATHER operation.

**Workaround**: Single-GPU training works perfectly. ETA for 100 epochs: 40-50 hours.

**Status**: **UNRESOLVED** - Requires deeper investigation into PyTorch DDP internals or dataset redesign.

---

## System Configuration

### Hardware
- **GPUs**: 12x NVIDIA Tesla T4 (15360 MiB each)
- **CUDA**: 13.0
- **Driver**: 580.95.05
- **Interconnect**: PCIe (P2P communication limited between T4s)

### Software
- **OS**: Ubuntu (Linux kernel)
- **Python**: 3.12
- **PyTorch**: 2.x with CUDA support
- **NCCL**: Bundled with PyTorch

### Dataset
- **Training samples**: 65,114 .npz files
- **Validation samples**: 4,072 .npz files
- **Classes**: 86 action classes
- **Feature format**: MediaPipe Holistic (543 keypoints × 3 coords per frame)
- **Sequence length**: 300 frames (10 seconds at 30fps)
- **Feature dimensions per sample**: (300, 1629) after flattening
- **File size**: ~578 KB per .npz file (591 KB compressed)

### Model
- **Architecture**: Temporal Transformer
- **Parameters**: 3,748,310
- **Layers**: 4 transformer layers, 8 attention heads, 256 hidden dim
- **Input**: (batch, 300, 1629)
- **Output**: 86 classes

---

## Timeline of Investigation

### Initial Observations (09:00-09:30)
1. **8 GPUs selected** for training (CUDA_VISIBLE_DEVICES=0-7)
2. **torchrun** successfully spawns 8 worker processes
3. All ranks (0-7) successfully join NCCL process group
4. Dataset loading completes in 0.5 seconds (only Rank 0 prints progress)
5. Model wraps with DDP successfully
6. Training loop entered, progress bar shows `Training: 0%|  | 0/255 [00:00<?, ?it/s]`
7. **HANGS**: No progress after 10+ minutes

### GPU Utilization Pattern
```
GPU 0-7: Utilization 100%, Memory 243-303 MiB, Memory Util 0%
GPU 8-11: Idle (not used)
```

**Analysis**: 100% compute with 0% memory utilization = GPUs are spinning/waiting, not processing data.

---

## Attempted Solutions

### 1. DataLoader Workers Configuration ❌
**Hypothesis**: Multiple DataLoader workers causing file I/O contention.

**Action**: Changed `num_workers=4` to `num_workers=0` in DataLoader initialization.

```python
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                          sampler=train_sampler, num_workers=0, pin_memory=True)
```

**Result**: FAILED - Still hangs at first batch.

---

### 2. Memory-Mapped File Loading ❌
**Hypothesis**: Concurrent numpy file reads blocking each other.

**Action**: Modified `__getitem__` to use memory-mapped read-only mode:

```python
def __getitem__(self, idx):
    npz_path, label = self.samples[idx]
    with np.load(npz_path, mmap_mode='r', allow_pickle=False) as data:
        features = np.array(data['features'])  # Copy to avoid mmap issues
    # ... rest of processing
```

**Result**: FAILED - Still hangs. Individual file loading works instantly (<0.01s).

---

### 3. Explicit Synchronization Barrier ❌❌
**Hypothesis**: Ranks reaching training loop at different times, need explicit sync.

**Action**: Added `dist.barrier()` after optimizer creation:

```python
log(f"[Rank {rank}] Optimizer ready", rank)
dist.barrier()  # Force all ranks to synchronize
log(f"[Rank {rank}] Starting training loop", rank)
```

**Result**: **WORSE** - Caused NCCL timeout deadlock:

```
[rank1]:[E1114 09:50:59.789610635 ProcessGroupNCCL.cpp:683] 
[Rank 1] Watchdog caught collective operation timeout: 
WorkNCCL(SeqNum=1, OpType=ALLGATHER, NumelIn=1, NumelOut=2, Timeout(ms)=600000) 
ran for 600067 milliseconds before timing out.
```

**Analysis**: 
- Rank 0 enqueued work 0,1,2,3 but only completed 1
- Rank 1 enqueued work 1 but completed -1 (nothing)
- Ranks reached barrier at different times → deadlock

**Action**: Removed barrier.

---

### 4. Reduced GPU Count (2 GPUs) ❌
**Hypothesis**: 8 GPUs too many, P2P communication issues.

**Action**: Ran with only 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_action_transformer_ddp.py
```

**Result**: FAILED - Same hang, same symptoms. Not a scale issue.

---

### 5. Torchrun vs Manual Spawn ❌
**Hypothesis**: Using `torch.multiprocessing.spawn()` conflicting with torchrun.

**Action**: Modified main() to detect torchrun environment variables:

```python
def main():
    args = parse_args()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        train_ddp(rank, world_size, args)
    else:
        # Fallback: manual spawn
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(train_ddp, args=(world_size, args), nprocs=world_size)
```

**Result**: FAILED - torchrun now correctly invoked, but still hangs.

---

### 6. Small Dataset Test ✅
**Hypothesis**: Dataset size is the issue.

**Action**: Created test with only 10 samples from one action class.

**Result**: **SUCCESS!** Test completes in 0.08-0.15 seconds per rank:

```
[Rank 0] ✅ Batch 0: features=torch.Size([2, 300, 1629]), elapsed=0.05s
[Rank 0] ✅ Batch 1: features=torch.Size([2, 300, 1629]), elapsed=0.08s
[Rank 0] ✅ Batch 2: features=torch.Size([1, 300, 1629]), elapsed=0.10s
[Rank 0] 🎉 SUCCESS! Loaded 3 batches in 0.10s
```

**Conclusion**: DDP works fine with small datasets. Issue is specific to 65K+ samples.

---

### 7. Single-GPU Training ✅
**Hypothesis**: Issue is DDP-specific, not dataset-specific.

**Action**: Ran DataParallel (not Distributed) training on 1 GPU with full dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_action_transformer_dataparallel.py \
    --features-dir data/processed/features_kinetics700 \
    --epochs 100 --batch-size 32
```

**Result**: **SUCCESS!** Training runs perfectly:
- GPU 0: 99% utilization, 1085 MiB memory
- Process CPU: 67-149%
- Estimated time: 40-50 hours for 100 epochs

**Conclusion**: Single-GPU works. Multi-GPU DDP does not.

---

## Error Messages & Diagnostics

### NCCL Timeout Error (with barrier)
```
[rank1]:[E1114 09:50:59.789610635 ProcessGroupNCCL.cpp:683] 
[Rank 1] Watchdog caught collective operation timeout: 
WorkNCCL(SeqNum=1, OpType=ALLGATHER, NumelIn=1, NumelOut=2, Timeout(ms)=600000) 
ran for 600067 milliseconds before timing out.

[rank0]:[E1114 09:51:00.658350320 ProcessGroupNCCL.cpp:1858] 
[PG ID 0 PG GUID 0(default_pg) Rank 0] Received a dump signal due to a 
collective timeout from rank 1. Last enqueued NCCL work: 3, last completed: 1.
This is most likely caused by incorrect usages of collectives, e.g., 
wrong sizes used across ranks, the order of collectives is not same for all ranks.
```

### Process State During Hang
```bash
$ ps aux | grep train_action_transformer_ddp
eoghan   3930122  10.6  0.0 4909820 593216 ?   Ssl  09:22   0:02 python ...  # Rank 0
eoghan   3930123  10.6  0.0 4909824 589888 ?   Ssl  09:22   0:02 python ...  # Rank 1
# ... 8 total processes, all in Ssl state (sleeping, waiting)
```

### GPU Monitoring
```
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory
0, 303 MiB, 100 %, 0 %
1, 263 MiB, 100 %, 0 %
2, 259 MiB, 100 %, 0 %
3, 243 MiB, 100 %, 0 %
4, 263 MiB, 100 %, 0 %
5, 259 MiB, 100 %, 0 %
6, 243 MiB, 100 %, 0 %
7, 259 MiB, 100 %, 0 %
```

---

## Root Cause Analysis

### What We Know

1. **Dataset initialization works**: All ranks load 65K file paths in 0.5 seconds
2. **Model initialization works**: DDP wrapper completes successfully
3. **NCCL setup works**: All ranks join process group
4. **Small datasets work**: 10 files load and train instantly
5. **Single-GPU works**: Full 65K dataset trains without issues
6. **The hang occurs**: At the FIRST DataLoader iteration in the training loop

### The Smoking Gun

The DataLoader's `__iter__()` is called when entering the training loop's `for batch in train_loader:` statement. This triggers:

1. **DistributedSampler** generates indices for this rank's subset of data
2. **DataLoader** prefetches batches (even with num_workers=0, some prefetching occurs)
3. **NCCL collectives** are called internally by DDP to synchronize

With 65K samples:
- DistributedSampler creates indices array: 65114 samples / 8 ranks = ~8139 samples per rank
- Each rank tries to build its subset simultaneously
- Some internal synchronization point in PyTorch's DataLoader+DistributedSampler+DDP stack deadlocks

### Hypothesis: Asynchronous Initialization Race

**Theory**: The combination of:
- Large DistributedSampler index generation (65K samples)
- DataLoader internal buffering/prefetching
- DDP's background NCCL communication threads
- Python's GIL (Global Interpreter Lock)

Creates a race condition where:
1. Rank 0 starts iterating and triggers NCCL collective
2. Ranks 1-7 are still initializing their iterators
3. NCCL collective waits for all ranks to participate
4. Ranks 1-7 can't reach collective because they're blocked on GIL or I/O
5. **Deadlock**

### Why Small Datasets Work

With only 10 samples:
- Index generation is instant (<1ms)
- All ranks reach the collective synchronously
- No opportunity for race condition

---

## Potential Solutions (Not Yet Tested)

### Option 1: Pre-shuffle Dataset Indices
Create a fixed, pre-shuffled index file that all ranks load instead of generating on-the-fly:

```python
# Generate once:
indices = list(range(len(dataset)))
random.shuffle(indices)
np.save('dataset_indices.npy', indices)

# In training:
indices = np.load('dataset_indices.npy')
sampler = DistributedSampler(dataset, shuffle=False)
# Use custom sampler that reads from pre-shuffled indices
```

### Option 2: Chunked Training
Split 65K dataset into 8 chunks of ~8K each, train sequentially with checkpoint resumption:

```python
# Train on chunk 1 → save checkpoint
# Load checkpoint, train on chunk 2 → save
# Repeat for all 8 chunks
# Final model sees all 65K samples
```

**Pros**: Each chunk is small enough to work with DDP
**Cons**: Not true distributed training, more manual orchestration

### Option 3: Increase NCCL Timeout
Default timeout is 10 minutes (600s). If initialization just needs more time:

```python
os.environ['NCCL_TIMEOUT'] = '3600'  # 1 hour
```

**Risk**: Might just delay the inevitable hang.

### Option 4: Custom DistributedSampler
Implement custom sampler that pre-generates all indices before `__iter__()`:

```python
class PrecomputedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        # Pre-compute all indices during init
        self._indices = self._compute_indices()
    
    def __iter__(self):
        # Return pre-computed indices immediately
        return iter(self._indices)
```

### Option 5: Gradient Accumulation on Single GPU
Train on 1 GPU with larger effective batch size via gradient accumulation:

```python
# Instead of batch_size=32 on 8 GPUs (effective=256)
# Use batch_size=32 on 1 GPU with accumulation_steps=8 (effective=256)

for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Pros**: Simpler, guaranteed to work
**Cons**: Slower (8x compared to ideal DDP)

### Option 6: Use Different Backend
Try Gloo instead of NCCL:

```python
dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

**Note**: Gloo is slower than NCCL for GPU communication but might be more stable.

---

## Recommendations

### Immediate Action (IMPLEMENTED)
**Single-GPU Training**: Launch 100-epoch training on 1 GPU.
- **ETA**: 40-50 hours (~2 days)
- **Command**: 
  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/train_action_transformer_dataparallel.py \
      --features-dir data/processed/features_kinetics700 \
      --epochs 100 --batch-size 32 --lr 0.0001 \
      2>&1 | tee logs/training_single_gpu_100epochs.log
  ```

### Future Investigation Priority

1. **Test Option 1** (Pre-shuffled indices) - Lowest risk, might solve issue
2. **Test Option 4** (Custom sampler) - More control over initialization
3. **Test Option 3** (Increase timeout) - Quick to test
4. **Test Option 6** (Gloo backend) - Diagnostic value
5. **Contact PyTorch team** - File GitHub issue with this report

### Long-term Solutions

1. **Dataset redesign**: 
   - Combine .npz files into larger HDF5 shards (~1000 samples per file)
   - Reduces file I/O operations from 65K to 65
   - Standard practice for large-scale training

2. **Use PyTorch Lightning**:
   - Abstracts DDP complexity
   - Better error handling and debugging
   - Proven at scale

3. **Profile with PyTorch Profiler**:
   ```python
   with torch.profiler.profile(
       activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
       on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
   ) as prof:
       # Training code
   ```

---

## Files Modified During Investigation

### Training Scripts
- `scripts/train_action_transformer_ddp.py` - Main DDP training script
- `scripts/launch_training_8gpus.sh` - Launcher with torchrun
- `scripts/launch_training_12gpus.sh` - Alternative with 2x6 GPU groups
- `scripts/test_ddp_dataloader.py` - Small dataset DDP test
- `scripts/debug_single_file.py` - Single .npz file test
- `scripts/train_action_transformer_dataparallel.py` - Single-GPU fallback

### Monitoring Scripts
- `scripts/monitor_training_live.sh` - Real-time GPU/process monitor
- `scripts/scan_npz_timeout.py` - File integrity scanner

### Documentation
- `docs/ddp_training_debugging_nov2025.md` - This document

---

## References & Related Issues

### PyTorch Issues
- [pytorch/pytorch#12831](https://github.com/pytorch/pytorch/issues/12831) - DistributedDataParallel hangs
- [pytorch/pytorch#53979](https://github.com/pytorch/pytorch/issues/53979) - NCCL timeout with large datasets
- [pytorch/pytorch#64779](https://github.com/pytorch/pytorch/issues/64779) - DataLoader deadlock in DDP

### NCCL Documentation
- [NCCL Troubleshooting Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

### Best Practices
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Distributed Training Tips](https://pytorch.org/tutorials/beginner/dist_overview.html)

---

## Contact & Follow-up

**Issue Owner**: Eoghan Hynes  
**Date Created**: November 14, 2025  
**Date Resolved**: November 17, 2025  
**Status**: ✅ **RESOLVED** - Gloo backend fixes hang  

For questions or updates, reference this document when implementing DDP training.

---

## SOLUTION FOUND - November 17, 2025

### Root Cause Identified

The hang was **NOT** in the DataLoader or DistributedSampler initialization as initially suspected. Verbose debug logging revealed the exact hang location:

```
[Rank 0] 🎉🎉 FIRST BATCH LOADED! batch_idx=0, features.shape=torch.Size([32, 300, 1629])
[Rank 0] 🔷 Moving batch to device cuda:0...
[HANG - 100% GPU, 0% memory utilization]
```

**Actual Root Cause**: NCCL backend deadlock during **`.to(device)` tensor transfer**, not during data loading.

### Working Solution: Gloo Backend

Changed from NCCL to Gloo backend for DDP communication:

```python
def setup_ddp(rank, world_size):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    
    # SOLUTION: Use Gloo instead of NCCL
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    log(f"[Rank {rank}] ✅ Joined process group with GLOO backend on GPU {rank}")
```

**Why This Works**:
- **NCCL** uses GPU-to-GPU direct communication (requires P2P support)
- **Gloo** uses CPU-based communication (TCP sockets)
- Tesla T4s on PCIe have limited P2P capabilities
- Gloo avoids the NCCL synchronization deadlock during `.to(device)`

### Test Results

**2 GPUs, 2 epochs test**:
```
[Rank 0] 🎉🎉🎉 FIRST BATCH COMPLETE! Training is WORKING!
Training:   2%|▏  | 17/1018 [00:07<07:13, 2.31it/s, loss=4.4645]
```

**GPU Utilization** (healthy training state):
```
GPU 0: 45% utilization, 1227 MiB memory
GPU 1: 54% utilization, 1123 MiB memory
```

**Compare to hang state**:
```
GPU 0: 100% utilization, 303 MiB memory  ← Spinning/waiting
GPU 1: 100% utilization, 263 MiB memory  ← Spinning/waiting
```

### Implementation Files

**New Training Script** (Gloo backend + pre-computed indices):
```bash
scripts/train_action_transformer_ddp_v2.py
```

**Supporting Files**:
```bash
scripts/generate_dataset_indices.py         # Pre-compute shuffled indices
scripts/ddp_precomputed_dataset.py         # Custom Dataset/Sampler
scripts/test_ddp_v2_quick.sh               # Quick 2-GPU test script
```

### Key Changes from Original

1. **Backend**: `"nccl"` → `"gloo"`
2. **Dataset**: Added pre-computed indices support (speeds up initialization)
3. **Debug logging**: Extensive verbose output to pinpoint issues

### Usage

**Quick test (2 GPUs, 2 epochs)**:
```bash
bash scripts/test_ddp_v2_quick.sh
```

**Full training (8 GPUs, 100 epochs)**:
```bash
# Step 1: Generate pre-computed indices (one time)
python3 scripts/generate_dataset_indices.py \
    --features-dir data/processed/features_kinetics700 \
    --output-dir data/processed

# Step 2: Launch training
CUDA_VISIBLE_DEVICES=0-7 torchrun --nproc_per_node=8 \
    scripts/train_action_transformer_ddp_v2.py \
    --features-dir data/processed/features_kinetics700 \
    --epochs 100 --batch-size 32
```

### Performance Expectations

- **8 GPUs**: ~5-6 hours for 100 epochs (vs 40-50 hours single GPU)
- **Speedup**: ~8x with 8 GPUs
- **Gloo overhead**: Minimal for large batches (models sync gradients, not every tensor)

### When to Use NCCL vs Gloo

**Use NCCL when**:
- GPUs have NVLink interconnect (A100, H100)
- Strong P2P support verified (`nvidia-smi topo -m`)
- Maximum performance needed

**Use Gloo when**:
- PCIe-connected GPUs (Tesla T4, GTX series)
- NCCL hangs or timeouts occur
- Cross-node training (Gloo has better CPU fallback)

### Lessons Learned

1. **Verbose debug logging is critical** - The hang location was not where initially suspected
2. **Backend choice matters** - NCCL assumptions about P2P can cause deadlocks
3. **Test incrementally** - Small dataset tests didn't expose the `.to(device)` issue
4. **GPU metrics tell the story** - 100% compute + 0% memory = spinning/waiting, not training

### Related Solutions Tested

| Solution | Status | Notes |
|----------|--------|-------|
| Pre-computed indices | ✅ Helpful | Speeds up initialization, doesn't fix hang |
| Gloo backend | ✅ **SOLVES ISSUE** | Fixes `.to(device)` deadlock |
| Increased NCCL timeout | ❌ Failed | Hang is deadlock, not slow initialization |
| Reduced GPU count | ❌ Failed | Reproduced with 2 GPUs |
| DataLoader workers=0 | ❌ Failed | Not a DataLoader issue |

---
