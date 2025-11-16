# ngp-minimal: Standalone Minimal NeRF Training

A minimal, standalone implementation of instant-ngp for NeRF-synthetic dataset training.

## Overview

This is a **completely independent** implementation that:
- ✅ Only supports NeRF-synthetic (Blender) dataset training
- ✅ Has NO dependencies on `../include` or `../src` from the full version
- ✅ Does NOT link against the `ngp-baseline-nv` library
- ✅ Uses the same class/function names as the full version for easy comparison
- ✅ Implements ONLY the training path - no GUI, VR, SDF, or other modes

## Goals

1. **Independence**: Fully self-contained implementation
2. **Minimal**: ~70-80% code reduction compared to full version
3. **Functional**: Successfully trains NeRF-synthetic datasets
4. **Consistent**: Training results match the full version
5. **Educational**: Clear, understandable code for learning purposes

## Implementation Status

### Phase 1: Baseline Analysis ✅
- [x] Identified core training call graph
- [x] Analyzed required data structures
- [x] Determined minimal feature set

### Phase 2: Project Skeleton ✅
- [x] Independent CMake configuration
- [x] Directory structure
- [x] CLI argument parsing
- [x] Compiles successfully

### Phase 3: Data Loading ✅
- [x] Math types (`vec2/3/4`, `mat3/4x3`, `BoundingBox`)
- [x] `GPUMemory<T>` CUDA memory management
- [x] `NerfDataset` structure
- [x] Blender NeRF-synthetic JSON parser
- [x] Image loading (PNG via stb_image)
- [x] **Verified**: Successfully loaded lego dataset (400 images, 800x800)

### Phase 4: Network & Training Core (TODO)
- [ ] `NerfNetwork<T>` wrapper (tiny-cuda-nn)
- [ ] Volume rendering kernels
- [ ] Training step implementation
- [ ] Loss computation

### Phase 5: Testbed Integration (TODO)
- [ ] `Testbed` class (NeRF-only)
- [ ] Training loop
- [ ] Density grid management
- [ ] Main loop integration

### Phase 6: Cleanup & Validation (TODO)
- [ ] Code cleanup
- [ ] Full validation against complete version
- [ ] Performance benchmarking
- [ ] Documentation

## Building

The project is built as part of the parent `ngp-baseline-nv` project:

```bash
cd ngp-baseline-nv
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j
```

The executable will be at: `build/ngp-minimal/ngp-minimal-app.exe`

## Usage

Once implementation is complete, usage will be:

```bash
# Train on lego scene
ngp-minimal-app --scene data/nerf-synthetic/lego --config configs/nerf/base.json

# Load from snapshot
ngp-minimal-app --scene data/nerf-synthetic/lego --snapshot checkpoints/lego.msgpack

# Test data loading without training
ngp-minimal-app --scene data/nerf-synthetic/lego --config configs/nerf/base.json --no-train
```

## What's NOT Included

This minimal version **does not** implement:
- ❌ GUI, rendering window, interactive camera
- ❌ VR support
- ❌ Multi-GPU training
- ❌ DLSS
- ❌ SDF, Volume, Image modes
- ❌ Environment map training
- ❌ Camera parameter optimization
- ❌ Marching Cubes mesh extraction
- ❌ Error map importance sampling (uses uniform sampling)

## What IS Included

This minimal version **does** implement:
- ✅ NeRF-synthetic (Blender) data loading
- ✅ HashGrid position encoding
- ✅ MLP network (via tiny-cuda-nn)
- ✅ Volume rendering
- ✅ RGB loss computation
- ✅ Density grid / Occupancy grid
- ✅ Training loop with Adam optimizer

## Architecture

```
ngp-minimal/
├── CMakeLists.txt          # Independent build config
├── README.md               # This file
├── include/
│   └── ngp-minimal/
│       ├── common.h        # Math types, enums
│       ├── common_device.cuh  # Device-side utilities
│       ├── gpu_memory.h    # CUDA memory management
│       ├── nerf_loader.h   # Dataset structures
│       ├── nerf_network.h  # Network wrapper
│       └── testbed.h       # Training orchestration
└── src/
    ├── main.cu             # CLI entry point
    ├── nerf_loader.cu      # Data loading
    ├── testbed.cu          # Main training logic
    └── testbed_nerf.cu     # NeRF-specific training
```

## Dependencies

**Required:**
- tiny-cuda-nn (FetchContent from parent)
- CUDA Toolkit
- nlohmann/json (from parent vcpkg)
- stb_image (vendor)
- args.hxx (vendor)

**Total:** ~5 dependencies vs ~15 in full version

## License

Same as parent project (NVIDIA proprietary license).

## Contributing

This is a minimal educational implementation. For production use, refer to the full `ngp-baseline-nv` implementation.

