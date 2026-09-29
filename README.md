# GPU Kernel Portfolio — Projects 7-13 (C++23)

Second batch, same treatment as the first (`gpu-portfolio.zip`, projects 1-6): analyzed, corrected,
completed. Project 6 (Convolution) was re-submitted in this batch's source material and already covered;
this package starts at project 7.

## Critical constraints: no GPU, no PyTorch

Same hard constraint as batch 1 — **no NVIDIA GPU and no CUDA toolkit (`nvcc`) in this sandbox** — plus a
new one specific to this batch: **no PyTorch installed either** (`import torch` fails). Projects 10, 11,
and 13 are PyTorch C++ extensions, so this second constraint matters specifically to them.

Consequences, applied consistently:
1. **Projects 7, 8, 9, 12's core algorithms** were verified the same way as batch 1's CUDA projects: the
   exact block/thread/warp structure transcribed into plain C++, checked against a trusted reference, run
   in this sandbox. The `.cu`/`.cuh` files themselves were never compiled or run.
2. **Projects 10 and 11** wrap a kernel whose *math* is independently checkable without PyTorch: project
   10's tiling is the same algorithm already verified for project 9; project 11's SiLU forward/backward
   was checked against finite differences in plain C++. The PyTorch binding layer (pybind11, tensor
   checks, autograd wiring) was reviewed by hand but not compiled or run.
3. **Project 13** (FlashAttention) — see below; its core recurrence has the strongest independent
   verification in this batch.

## A finding about the verification method itself

While building project 13's test, a deliberately-injected bug (a sign error in the online-softmax
rescaling factor) **did not get caught** by the first two versions of the test, despite the underlying
algorithm being wrong. The cause was in the *test*, not the algorithm: the diff-tracking pattern
`worst = std::max(worst, diff)`, used throughout this project and copied from patterns used in batch 1,
silently drops `NaN`: `std::max(a, NaN)` returns `a` unchanged, because every comparison against `NaN` is
false. The injected bug made the simulated algorithm output `NaN` immediately (an `exp()` of a huge
number, multiplied into an accumulator that started at `-FLT_MAX`), and the max-tracking loop let that
`NaN` vanish instead of reporting it, so the check read `worst == 0` and passed.

Fixed in `13_flash_attention/tests/test_flash_attention_cpu_sim.cpp` with an explicit `std::isfinite`
check, confirmed to catch the same mutation afterward, and then **applied retroactively** to every other
file using the same pattern, in both batches: `07_softmax`, `09_gemm`, `12_wmma_tensorcores` in this batch,
and `02_matmul_cpu`, `06_convolution` in batch 1 (whose zip was rebuilt with this fix and should be
considered the current version). None of those files had an actual bug hiding behind this blind spot —
the point is that they *could* have, and now can't.

## What was found and fixed

| Project | Bug in the original draft | Fix | Verified how |
|---|---|---|---|
| 7. Softmax | none found | — | CPU simulation of both v0 (3-pass) and v1 (online) vs a double-precision reference: 105 checks, degenerate (constant-row) and near-overflow inputs included |
| 8. LayerNorm/RMSNorm | reduction only correct for exactly 32 threads/block — silently wrong for the 128/256 threads the draft's own comments recommend for D=4096 (confirmed: ~75% relative error reproduced at 128 threads); float4 path dropped the last D%4 elements | added a proper multi-warp block reduction (shared-memory cross-warp combine); added a scalar tail pass | CPU simulation, 34,796 checks across block sizes 32-256 and D both aligned and not; both original bugs reproduced by hand before fixing |
| 9. GEMM | none found | — | CPU simulation of the tiling/zero-padding boundary logic for M/N/K not multiples of the tile size (never exercised by the original's fixed 2048³ benchmark): 8 shapes, 0 failures |
| 10. PyTorch matmul | same algorithm as project 9 (already verified); no launch-error check; no explicit dtype check | added `cudaGetLastError()` + `TORCH_CHECK`; added explicit dtype checks | reduces to project 9's verification; not independently re-run (no PyTorch here) |
| 11. SiLU autograd | no launch-error check; test script used `grad_out = ones`, which cannot expose a wrong chain-rule multiplication | added launch-error checks; forward/backward formulas checked against finite differences (9,014 checks, all `x` regimes including saturation); test script fixed to use random `grad_out` | finite-difference check run in this sandbox (no PyTorch needed for the math itself); mutation-tested (a broken derivative formula was confirmed to fail 3,529 of 9,014 checks) |
| 12. WMMA Tensor Cores | **grid computed with plain integer division, no ceiling** — for M=100, N=50, confirmed 36 of 100 rows and 2 of 50 columns of the output were never touched by any block at all; separately, even covered warps could read/write past the buffer when a tile straddled the true M/N/K boundary | pad A/B (and the output) to the tile-aligned size with zeros before launch, slice the true M×N region back out afterward | CPU simulation of the padding invariant (padding cannot change any answer in the valid region): 6 shapes, 0 failures; the original grid-undercoverage bug reproduced numerically by hand |
| 13. FlashAttention | none found in the core recurrence (see below); no launch-error check; no shared-memory budget request for large head dimensions | added launch-error check; added `cudaFuncSetAttribute` for the extended dynamic shared memory large `d` needs | CPU simulation of the exact tiled online-softmax recurrence vs naive full attention: 10 checks across N not a multiple of the KV block size and several head dimensions; the recurrence itself mutation-tested (a sign-flipped rescaling factor confirmed to fail 10/10 checks once the test's own NaN-blindness bug, above, was fixed) |

## Build and test

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

Builds and runs every CPU-side simulation (projects 7, 8, 9, 11's math, 12, 13). On a machine with a CUDA
toolkit, the same commands additionally build `07_softmax_gpu`, `08_norm_gpu`, `09_gemm_gpu`,
`12_wmma_gpu` (12 additionally needs a Volta-or-newer GPU for actual Tensor Core execution).

**Projects 10, 11, 13 are PyTorch C++ extensions and are intentionally not part of this CMake build** —
that's not an oversight; it's how PyTorch extensions are normally built, since `torch.utils.cpp_extension`
knows how to locate LibTorch and pybind11 in a way a plain `CMakeLists.txt` doesn't without assuming a
Python + PyTorch environment just to *configure* the rest of this batch. With PyTorch and a CUDA toolkit
available:

```bash
cd 10_pytorch_matmul && pip install . && python test_matmul.py
cd 11_pytorch_silu_autograd && pip install . && python test_silu.py
cd 13_flash_attention && pip install . && python test_flash_attention.py
```

## Per-project detail

- **`07_softmax/`** — `tests/test_softmax_cpu_sim.cpp` (105 checks). `src/main.cu` (GPU-only) validates
  every row against a CPU reference, not just row 0 as the original did.
- **`08_norm/`** — `tests/test_norm_cpu_sim.cpp` (34,796 checks, both original bugs reproduced and
  fixed). `src/main.cu` (GPU-only) deliberately uses 128 threads/block, the exact configuration that
  broke the original.
- **`09_gemm/`** — `tests/test_gemm_cpu_sim.cpp` (8 checks, non-tile-aligned shapes). `src/main.cu`
  (GPU-only) validates every element against a CPU reference; the original validated nothing.
- **`10_pytorch_matmul/`** — `csrc/`, `setup.py`, `test_matmul.py` (all untested here). Test script fixed
  to include shapes not a multiple of the tile size.
- **`11_pytorch_silu_autograd/`** — `tests/test_silu_math.cpp` (9,014 checks, run in this sandbox,
  mutation-tested). `csrc/`, `setup.py`, `test_silu.py` (untested here; test script fixed to use random
  `grad_out`).
- **`12_wmma_tensorcores/`** — `tests/test_wmma_padding_sim.cpp` (6 checks). `include/wmma_gemm.cuh` /
  `src/main.cu` (GPU-only, needs Tensor Cores) implement the padding fix.
- **`13_flash_attention/`** — `tests/test_flash_attention_cpu_sim.cpp` (10 checks, the strongest
  independent verification in this batch, including the NaN-blindness fix described above).
  `csrc/`, `setup.py`, `test_flash_attention.py` (untested here).

## What to do with a GPU (and, for 10/11/13, PyTorch) available

1. Raw-CUDA projects (7, 8, 9, 12): `cmake --build build` builds the `*_gpu` targets; run each with no
   arguments, every one validates its own output against a CPU reference before printing any benchmark
   numbers.
2. PyTorch extensions (10, 11, 13): `pip install .` then run the project's test script.
3. If anything mismatches, the CPU-side simulations in `tests/` isolate the same algorithm from
   CUDA/PyTorch-specific concerns and are fast to iterate on without a GPU.
