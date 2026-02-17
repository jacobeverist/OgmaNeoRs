# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AOgmaNeo is a C++14 library implementing Sparse Predictive Hierarchies (SPH) — a biologically-inspired neural architecture. It targets desktop, embedded, and microcontroller (Arduino) platforms. Python bindings exist in the separate [PyAOgmaNeo](https://github.com/ogmacorp/PyAOgmaNeo) repo. Licensed CC BY-NC-SA 4.0.

## Build Commands

```bash
# macOS (Apple Silicon) — set up OpenMP environment first
brew install cmake
brew install llvm
brew install libomp
source setup_env.sh

# Standard build
mkdir build && cd build
cmake ..
make

# Install/uninstall
make install
make uninstall

# Shared library (recommended for Python bindings on Linux)
cmake .. -DBUILD_SHARED_LIBS=ON
```

Requires CMake 3.13+ and OpenMP. On macOS with Homebrew, `setup_env.sh` configures the LLVM/OpenMP paths.

## Architecture

All code lives in `source/aogmaneo/` under namespace `aon`. The library builds as a static library (`libAOgmaNeo.a`) by default.

### Core Classes

**Hierarchy** (`hierarchy.h/cpp`) — The top-level orchestrator. Coordinates a stack of Encoder/Decoder/Actor layers. Configured via `IO_Desc` (input/output ports) and `Layer_Desc` (hidden layers). Each IO port has a type: `none` (input-only), `prediction`, or `action` (RL). Runtime parameters are adjusted via `Hierarchy::Params`.

**Encoder** (`encoder.h/cpp`) — Sparse coding layer using Adaptive Resonance Theory (ART). Computes sparse activations column-wise via vigilance matching + lateral inhibition. Uses 8-bit unsigned weights (`Byte_Buffer`).

**Decoder** (`decoder.h/cpp`) — Predictive reconstruction layer with multi-dendrite perceptrons. Uses 8-bit signed weights (`S_Byte_Buffer`). Predicts input from hidden sparse representations.

**Actor** (`actor.h/cpp`) — Reinforcement learning layer implementing actor-critic with eligibility traces and a circular history buffer for temporal credit assignment. Largest source file (~1000 lines).

**Image_Encoder** (`image_encoder.h/cpp`) — Self-organizing map variant specialized for image data. Supports `reconstruct()` for decoding back to original resolution.

### Data Flow

```
IO Inputs → Encoder[0] → Encoder[1] → ... → Encoder[N]
                ↓              ↓                  ↓
           Decoder[0]     Decoder[1]         Decoder[N]
           Actor[0]*
              ↓
         Predictions/Actions
```

Layers support optional recurrent connections (controlled by `recurrent_radius` in `Layer_Desc`). Exponential memory is achieved via `ticks_per_update` — each layer processes at a different temporal stride.

### Supporting Code

**helpers.h/cpp** — Math functions (custom fast implementations or `USE_STD_MATH`), PCG32 RNG, activation functions (sigmoid, tanh, symlog/symexp), vector types (`Int2`, `Int3`, `Float2`, etc.), `Circle_Buffer`, stream serialization base classes.

**array.h** — `Array<T>` (owning) and `Array_View<T>` (non-owning) container templates.

### Key Type Aliases

- `Byte_Buffer` / `S_Byte_Buffer` — 8-bit unsigned/signed weight storage
- `Int_Buffer` / `Float_Buffer` — integer/float data buffers
- `Int3` — `Vec3<int>`, used for layer sizes as (width, height, column_size)

## Naming Conventions

C++ uses Pascal case for structs (`Layer_Desc`) and snake_case for everything else. See `NameReference.md` for the full variable naming glossary. Key abbreviations: `vl` = visible layer, `hc` = hidden column, `ci` = column index, `cis` = column indices, `wi` = weight index, `diam` = diameter (2*radius+1).

## Design Patterns

- **Descriptor/Params separation**: Structural config (`IO_Desc`, `Layer_Desc`) set at creation; runtime params (`Encoder::Params`, `Decoder::Params`, `Actor::Params`) adjustable anytime via `hierarchy.params`.
- **Column-wise parallelism**: All compute kernels operate per-column, parallelized with OpenMP via `PARALLEL_FOR` / `ATOMIC` macros. Gracefully degrades without OpenMP.
- **8-bit weight quantization**: Encoder/Decoder use byte-sized weights for memory efficiency on embedded targets.
- **Serialization**: All classes implement `write()`/`read()`, `write_state()`/`read_state()`, `write_weights()`/`read_weights()` via abstract `Stream_Writer`/`Stream_Reader` interfaces.

## Compile Definitions

- `USE_OMP` — Enables OpenMP parallelization (set in CMakeLists.txt)
- `USE_STD_MATH` — Uses standard library math instead of custom fast-math implementations
