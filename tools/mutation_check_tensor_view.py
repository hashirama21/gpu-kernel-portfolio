#!/usr/bin/env python3
"""Mutation check for project 3 (TensorView)."""
import subprocess, shutil, os, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "03_tensor_view")

MUTATIONS = [
 ("reshape: contiguity check removed (the original bug)",
  'if (!is_contiguous())\n      throw std::logic_error("reshape: view is not contiguous (make a copy first, or use a compatible slice)");\n    const std::size_t new_numel',
  'const std::size_t new_numel'),
 ("numel(): empty shape returns 0 again (the original bug)",
  'return std::accumulate(shape_.begin(), shape_.end(), std::size_t{1}, std::multiplies<>());',
  'if (shape_.empty()) return 0; return std::accumulate(shape_.begin(), shape_.end(), std::size_t{1}, std::multiplies<>());'),
 ("broadcast_to: size-1 axis not forced to stride 0", 'new_strides[i] = 0;  // broadcast axis: stride 0 makes every index alias the single element', 'new_strides[i] = src.strides()[src_idx];'),
 ("contiguous_strides: row-major formula reversed (off-by-one direction)", 's[i] = acc;\n      acc *= shape[i];', 's[i] = acc;\n      acc *= shape[i] + 1;'),
 ("slice: forgets to advance the data pointer", 'return TensorView(data_ptr_ + index * strides_[0], std::move(new_shape), std::move(new_strides));', 'return TensorView(data_ptr_, std::move(new_shape), std::move(new_strides));'),
 ("bounds check: off-by-one allows idx == shape[i]", 'if (idx[i] >= shape_[i]) throw std::out_of_range("TensorView: index out of bounds");', 'if (idx[i] > shape_[i]) throw std::out_of_range("TensorView: index out of bounds");'),
]

for name, old, new in MUTATIONS:
    d = os.path.join(tempfile.gettempdir(), "tv_mut")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(os.path.join(SRC, "include"), os.path.join(d, "include"))
    shutil.copytree(os.path.join(SRC, "tests"), os.path.join(d, "tests"))
    p = os.path.join(d, "include", "tensor_view.hpp")
    s = open(p).read()
    if old not in s:
        print(f"!! pattern not found: {name}")
        continue
    open(p, "w").write(s.replace(old, new, 1))
    c = subprocess.run(["g++", "-std=c++23", "-O2", "-Wall", "-I", os.path.join(d, "include"),
                        os.path.join(d, "tests", "test_tensor_view.cpp"), "-o", os.path.join(d, "t")],
                       capture_output=True, text=True)
    if c.returncode:
        print(f"CAUGHT (compile error) {name}")
        continue
    r = subprocess.run([os.path.join(d, "t")], capture_output=True, text=True)
    print(("CAUGHT " if r.returncode != 0 else "MISSED!") + f" {name}")
