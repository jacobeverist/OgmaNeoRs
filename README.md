# AOgmaNeo

A Rust port of the [AOgmaNeo](https://github.com/ogmacorp/AOgmaNeo) library by [Ogma Intelligent Systems Corp](https://ogmacorp.com). AOgmaNeo implements **Sparse Predictive Hierarchies (SPH)** — a biologically-inspired online machine learning system with a low compute footprint that learns from streaming data without forgetting.

The original C++ source is preserved in `cpp_ref/` for reference.

Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## Features

- **Online learning** — learns from a data stream one sample at a time, in order, without forgetting
- **Sparse representations** — CSDRs (Columnar Sparse Distributed Representations) keep compute proportional to active content, not total network size
- **Short-term memory** — exponential memory via a clockwork hierarchy of layers
- **Reinforcement learning** — built-in actor-critic with eligibility traces
- **Image encoding** — self-organizing map pre-encoder with reconstruction
- **Serialization** — save/load weights, state, or both
- **Parallel forward pass** — encoder and decoder columns computed in parallel via `rayon`

---

## Quick Start

```toml
[dependencies]
aogmaneo = { path = "." }
```

```rust
use aogmaneo::helpers::Int3;
use aogmaneo::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

// Configure the hierarchy
let io_descs = vec![IoDesc {
    size: Int3::new(4, 4, 16), // 4×4 grid, 16 cells per column
    io_type: IoType::Prediction,
    ..IoDesc::default()
}];
let layer_descs = vec![LayerDesc {
    hidden_size: Int3::new(4, 4, 16),
    ..LayerDesc::default()
}];

let mut h = Hierarchy::new();
h.init_random(&io_descs, &layer_descs);

// Step with your input CSDR (Vec<i32>, values in [0, column_size))
let input_cis: Vec<i32> = vec![0i32; 4 * 4];
h.step(&[&input_cis], true, 0.0, 0.0);

// Read the next-step prediction
let prediction: &[i32] = h.get_prediction_cis(0);
```

---

## Building and Testing

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run a single test
cargo test test_hierarchy_create_and_step

# Lint
cargo clippy
```

---

## Examples

### Pure Rust (no extra dependencies)

```bash
# CartPole balancing via RL (built-in physics)
cargo run --release --example cartpole

# Wavy-line sequence prediction (ASCII recall output)
cargo run --release --example wave_prediction
```

### Gymnasium examples (requires Python)

These examples drive [Gymnasium](https://gymnasium.farama.org/) environments via PyO3.

**Setup** — create an ARM64 virtual environment and install gymnasium:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install gymnasium
pip install "gymnasium[box2d]"   # needed for LunarLander
```

**Run** — the examples automatically detect the `.venv` directory, so no activation is needed:

```bash
cargo run --release --example cartpole_env_runner --features gymnasium-examples
cargo run --release --example lunarlander --features gymnasium-examples
```

The examples check `$VIRTUAL_ENV` (set by `source .venv/bin/activate`) and fall back to a `.venv` directory in the current working directory. Activating the venv first also works if you prefer.

---

## How It Works

AOgmaNeo processes data as a stream of **CSDRs** — flat `Vec<i32>` arrays where each integer selects the active cell in a column of a 2D grid. The hierarchy encodes inputs upward through a stack of sparse coding layers, then decodes predictions back downward.

Each timestep:
1. **Up pass** — input CSDRs are encoded layer by layer into progressively more abstract sparse representations.
2. **Down pass** — decoders reconstruct predictions of the next input from the current hidden state.

Higher layers clock less frequently, giving the system exponential temporal memory with near-constant added compute per layer.

See [`doc/AOgmaNeo_User_Guide.md`](doc/AOgmaNeo_User_Guide.md) for a full explanation of the concepts and API.

---

## Documentation

| Document | Description |
|---|---|
| [`doc/AOgmaNeo_User_Guide.md`](doc/AOgmaNeo_User_Guide.md) | Full user guide: concepts, API reference, RL example |
| [`doc/TuningGuide.md`](doc/TuningGuide.md) | Parameter descriptions and tuning advice |
| [`doc/NameReference.md`](doc/NameReference.md) | Variable naming glossary for reading the source |
| [`doc/CppToRust.md`](doc/CppToRust.md) | Mapping from C++ names/types to Rust equivalents |

---

## Source Layout

```
src/
  helpers.rs       — Int2/Int3, PCG32 RNG, VecWriter/SliceReader, CircleBuffer
  encoder.rs       — ART sparse coder (parallel)
  decoder.rs       — multi-dendrite perceptrons (parallel)
  actor.rs         — actor-critic RL with eligibility traces
  image_encoder.rs — SOM for images with reconstruct()
  hierarchy.rs     — top-level orchestrator
tests/
  smoke_test.rs    — integration tests
examples/
  cartpole.rs             — CartPole balancing via RL (pure Rust)
  wave_prediction.rs      — wavy-line sequence prediction (pure Rust)
  cartpole_env_runner.rs  — CartPole-v1 via gymnasium (requires --features gymnasium-examples)
  lunarlander.rs          — LunarLander-v3 via gymnasium (requires --features gymnasium-examples)
cpp_ref/           — original C++ source (reference only)
doc/               — documentation
```

---

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

AOgmaNeo Copyright © 2020-2025 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
Contact: licenses@ogmacorp.com for commercial licensing.
