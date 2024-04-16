Run `mkDatasets.sh` to make the datasets and then use futhark benchmark to run the programs:

```
futhark bench --backend=cuda histo-add-min-mul-vec.fut
```
