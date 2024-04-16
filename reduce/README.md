Use futhark benchmark to run the programs:

```
futhark bench --backend=cuda red-basic-ops-vec.fut
```

The performance reported in the paper might not be entirely accurate with the latest Futhark compiler, for example because improvements in the code generation of reduce with non-commutative operator have significantly improved the performance of the primal in such cases.
