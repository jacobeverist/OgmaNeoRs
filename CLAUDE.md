# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AOgmaNeo implements Sparse Predictive Hierarchies (SPH) — a biologically-inspired neural architecture. The **Rust port is the active codebase** at the repository root. The original C++ source is preserved in `cpp_ref/` for reference. Licensed CC BY-NC-SA 4.0.

## Rust Codebase (Primary)

### Build & Test

```bash
# Build
cargo build --release

# Run all tests (6 integration tests)
cargo test

# Run a single test
cargo test test_hierarchy_create_and_step

# Run with output visible
cargo test -- --nocapture

# Lint (must be clean, 0 warnings)
cargo clippy

# Run the cartpole RL example
cargo run --release --example cartpole
```

### Source Layout

```
src/
  lib.rs           — module declarations only
  helpers.rs       — Int2/Int3/Float2, PCG32 RNG, VecWriter/SliceReader, CircleBuffer
  encoder.rs       — ART sparse coder (parallel via rayon)
  decoder.rs       — multi-dendrite perceptrons (parallel via rayon)
  actor.rs         — actor-critic RL with eligibility traces (sequential)
  image_encoder.rs — SOM for image data, supports reconstruct()
  hierarchy.rs     — top-level orchestrator
tests/
  smoke_test.rs    — 6 integration tests
examples/
  cartpole.rs      — CartPole RL demo
```

### Architecture

**Hierarchy** is the user-facing entry point. It holds a stack of Encoder layers with associated Decoders and optional Actors.

Configuration is split:
- **Structural** (`IoDesc`, `LayerDesc`) — set at `init_random()` time; cannot change afterwards.
- **Runtime** (`LayerParams` containing `DecoderParams`, `encoder::Params`) — adjustable anytime via `hierarchy.params`.

`IoType` on each `IoDesc` determines behavior:
- `None` — input-only (no decoder/actor)
- `Prediction` — encoder + decoder predicts the IO's next state
- `Action` — encoder + actor (RL); uses reward signal passed to `step()`

`step(&[input_cis], learn, reward, mimic)` drives one timestep. `mimic` is `f32` (not `bool`).

**Data representation**: all inputs/outputs are `Vec<i32>` of column indices (`cis`). A column index selects one active cell within each spatial column. Layer sizes are `Int3 { x, y, z }` where `x*y` is the number of columns and `z` is the column size (number of cells per column).

**Parallelism**: Encoder and Decoder forward passes use `rayon` (`into_par_iter()`). Actor and ImageEncoder are sequential. Parallel kernels collect per-column results into `Vec<ForwardResult>` to avoid shared mutable state.

**Serialization**: `VecWriter` / `SliceReader` do field-by-field little-endian I/O (not raw struct binary). Access the written bytes as `writer.data` (not `.into_bytes()`).

**RNG**: thread-local PCG32 via `global_rand()`. Column kernels seed a local RNG with `rand_get_state(seed)` for deterministic per-column randomness.

### Key Clippy Suppressions

`#![allow(clippy::needless_range_loop)]` is at the top of every compute module. Private column kernels carry `#[allow(clippy::too_many_arguments)]`. These are intentional and should be preserved.

## C++ Reference (`cpp_ref/`)

Original C++ source under namespace `aon`. Useful when porting logic or verifying algorithm correctness. Build with CMake (requires OpenMP):

```bash
# macOS (Apple Silicon)
brew install cmake llvm libomp
source setup_env.sh
mkdir build && cd build
cmake .. && make
```

Key abbreviations used throughout C++ and Rust: `vl` = visible layer, `hc` = hidden column, `ci` = column index, `cis` = column indices, `wi` = weight index, `diam` = diameter (2×radius+1).
