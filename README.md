# warp-vote-consensus

GPU warp-vote hardware as agent consensus primitive. 32-thread ballots map to ternary voting, with quorum tree and CRDT merge for fleet-wide decisions.

## Why This Matters

# warp-vote-consensus
Experiments with using GPU warp-vote hardware as an agent consensus primitive.
## Architecture
Real GPU warp hardware provides `__ballot_sync(mask, predicate)` — a single
instruction that collects 32 thread predicates into a 32-bit ballot word in

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub enum Vote
pub fn to_bits
pub fn from_bits
pub struct BallotWord
pub fn collect
pub fn tally
pub fn majority_agree
pub fn to_bytes
pub fn from_bytes
pub struct VoteTally
pub fn agreement_rate
pub struct Warp
```

## Usage

```toml
[dependencies]
warp-vote-consensus = "0.1.0"
```

```rust
use warp_vote_consensus::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/warp-vote-consensus.git
cd warp-vote-consensus
cargo test    # 23 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 23 |
| Lines of Rust | 729 |
| Public API | 35 items |

## License

Apache-2.0
