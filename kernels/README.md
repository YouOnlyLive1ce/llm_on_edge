quip(pull4773)
gptq
sparsegpt
kernels gpu

torch tensor -> triton kernel -> cuda kernel

# Results

|          | Torch   | Triton    | Cuda   | Cublass/Cutlass |
|----------|----------|----------|--------|--------|
| Kernel_name | Cell 2 | Cell 3 |  Cell 2 |   2    |
| Kernel_name | Cell 5 | Cell 6 |  Cell 2 |   3    |