#!/usr/bin/env python3
"""Mutation check for project 2 (matmul): inject one bug at a time and confirm the test suite catches it."""
import subprocess, shutil, sys, os, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "02_matmul_cpu")

MUTATIONS = [
 ("tiled: k-loop uses j0 instead of k0",           "for (int k = k0; k < k_max; ++k) {\n            const float a = A[static_cast<std::size_t>(i) * K + k];\n            const float* brow = B.data() + static_cast<std::size_t>(k) * N;\n            float* crow = C.data() + static_cast<std::size_t>(i) * N;\n            for (int j = j0; j < j_max; ++j) crow[j] += a * brow[j];\n          }\n      }\n}\n\n#if defined(__AVX2__)", "for (int k = j0; k < k_max; ++k) {\n            const float a = A[static_cast<std::size_t>(i) * K + k];\n            const float* brow = B.data() + static_cast<std::size_t>(k) * N;\n            float* crow = C.data() + static_cast<std::size_t>(i) * N;\n            for (int j = j0; j < j_max; ++j) crow[j] += a * brow[j];\n          }\n      }\n}\n\n#if defined(__AVX2__)"),
 ("ikj: C not zeroed before accumulation",         "std::fill(C.begin(), C.begin() + static_cast<std::ptrdiff_t>(M) * N, 0.f);\n  for (int i = 0; i < M; ++i)\n    for (int k = 0; k < K; ++k) {\n      const float a = A[static_cast<std::size_t>(i) * K + k];", "for (int i = 0; i < M; ++i)\n    for (int k = 0; k < K; ++k) {\n      const float a = A[static_cast<std::size_t>(i) * K + k];"),
 ("AVX2: vc/vb operand order swapped in FMA (wrong accumulator)", "vc = _mm256_fmadd_ps(va, vb, vc);", "vc = _mm256_fmadd_ps(va, vc, vb);"),
 ("AVX2: scalar cleanup uses old `a` not tile-local", "const float a = A[static_cast<std::size_t>(i) * K + k];\n            for (; j < j_max; ++j) crow[j] += a * brow[j];\n          }\n      }\n}\n\n// tiled_avx2, with the outer i-tile loop", "for (; j < j_max; ++j) crow[j] += 0.f * brow[j];\n          }\n      }\n}\n\n// tiled_avx2, with the outer i-tile loop"),
 ("tiled: i-tile bound uses M directly, ignoring i_max clamp", "const int i_max = std::min(i0 + tile, M), k_max = std::min(k0 + tile, K), j_max = std::min(j0 + tile, N);\n        for (int i = i0; i < i_max; ++i)\n          for (int k = k0; k < k_max; ++k) {\n            const float a = A[static_cast<std::size_t>(i) * K + k];\n            const float* brow = B.data() + static_cast<std::size_t>(k) * N;\n            float* crow = C.data() + static_cast<std::size_t>(i) * N;\n            for (int j = j0; j < j_max; ++j) crow[j] += a * brow[j];\n          }\n      }\n}\n\n#if defined(__AVX2__)", "const int i_max = std::min(i0 + tile, M), k_max = std::min(k0 + tile, K), j_max = std::min(j0 + tile, N);\n        for (int i = i0; i < M; ++i)\n          for (int k = k0; k < k_max; ++k) {\n            const float a = A[static_cast<std::size_t>(i) * K + k];\n            const float* brow = B.data() + static_cast<std::size_t>(k) * N;\n            float* crow = C.data() + static_cast<std::size_t>(i) * N;\n            for (int j = j0; j < j_max; ++j) crow[j] += a * brow[j];\n          }\n      }\n}\n\n#if defined(__AVX2__)"),
]

for name, old, new in MUTATIONS:
    d = os.path.join(tempfile.gettempdir(), "matmul_mut")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(os.path.join(SRC, "include"), os.path.join(d, "include"))
    shutil.copytree(os.path.join(SRC, "tests"), os.path.join(d, "tests"))
    p = os.path.join(d, "include", "matmul.hpp")
    s = open(p).read()
    if old not in s:
        print(f"!! pattern not found: {name}")
        continue
    open(p, "w").write(s.replace(old, new, 1))
    c = subprocess.run(["g++", "-std=c++23", "-O2", "-march=native", "-Wall", "-I", os.path.join(d, "include"),
                        os.path.join(d, "tests", "test_matmul.cpp"), "-o", os.path.join(d, "t")],
                       capture_output=True, text=True)
    if c.returncode:
        print(f"!! compile error: {name}\n{c.stderr[:400]}")
        continue
    r = subprocess.run([os.path.join(d, "t")], capture_output=True, text=True)
    print(("CAUGHT " if r.returncode != 0 else "MISSED!") + f" {name}")
