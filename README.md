# Custom CUDA kernel for GRU inference (in progress)

Work in progress: a fused CUDA kernel for the gate maths of the GRU layer in my commodity-liquidity model, to be benchmarked against a naive PyTorch loop and cuDNN's `nn.GRU`.

| Part | Status |
|---|---|
| Naive GRU, checked against `nn.GRU` | done |
| Benchmark (naive vs `nn.GRU`) | done |
| Fused CUDA kernel | in progress |
| Benchmark against the kernel | not started |




Developing custom CUDA kernels for GRU inference that fuse the per-timestep gate computations into a single kernel,
targeting the launch-overhead and memory-traffic bottlenecks of small-batch serving.
Building a benchmark harness against naive PyTorch and cuDNN-backed nn.GRU across batch sizes and sequence lengths,
with correctness tests against torch.nn.GRU.
