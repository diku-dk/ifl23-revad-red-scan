Use futhark benchmark to run the programs:

```
futhark bench --backend=cuda test-scan-mm5.fut --pass-option=--default-group-size=512
```

Observations:

1. Best results for scan (with CUDA backend) seem to be obtained with a CUDA block size equal to 512.

2. Since the paper was written, due to various compiler changes (e.g., to the implementation of scan) the performance results reported in the paper may not be entirely accurate anymore. For example, on an A100 GPU, currently, on `matMul3x3` PPAD is faster by `1.06x`, on `matMul5x5` Ours is faster by `1.32x` and on `matMul2x2` they seem to have equal performance.
