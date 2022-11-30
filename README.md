# repo-team13

Download and split PCQM4Mv2 dataset
```bash
python3 dataset/pace_ice_split.py --name PCQM4Mv2 --train_split 0.05 --method "scaffold"
```

To run fedml:

```bash
cd qcf_bench
conda env export > environment.yml
./run_bench.sh 1
```
