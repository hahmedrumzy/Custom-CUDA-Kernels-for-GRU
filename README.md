# Custom CUDA kernel for GRU inference (in progress)

Work in progress: a fused CUDA kernel for the gate maths of the GRU layer in my commodity-liquidity model, to be benchmarked against a naive PyTorch loop and cuDNN's `nn.GRU`.

| Part | Status |
|---|---|
| Naive GRU, checked against `nn.GRU` | done |
| Benchmark (naive vs `nn.GRU`) | done |
| Fused CUDA kernel | in progress |
| Benchmark against the kernel | not started |
