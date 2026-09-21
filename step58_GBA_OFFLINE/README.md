# Step58-GBA-v2 Offline Audit

This version separates fixed CA-LGM training from gradient diagnostics.

1. `trace`: runs the original fixed `lambda=0.05` CA-LGM update and saves
   state/batch snapshots. It performs no GBA `autograd.grad` calls.
2. `audit`: starts only after training, loads the frozen snapshots, and
   computes PGE `g_H`, `g_LP`, injection ratio, and cosine.

Shared dependencies remain in `../src`, `../data`, and `../step58_CA_LGM`.

## First validation experiment

PyCharm script:

`E:\CODE\LAPT -ONLY\step58_GBA_OFFLINE\run_pipeline.py`

Parameters:

```text
--stage all --dataset computers --rate 0.005 --seed 1 --gpu 0 --epochs 600 --lp-weight 0.05 --eval-interval 60 --eval-runs 10
```

For separate machines/stages, first run `--stage trace`, copy the complete
trace output, and then run the same parameters with `--stage audit`.

Outputs:

`E:\CODE\LAPT -ONLY\outputs\step58_GBA_OFFLINE\computers\r_0.005\seed_1`

The formal F1 is written by the trace run. Offline gradient statistics are in
`offline_audit/offline_gradient_diagnostics.csv` and
`offline_audit/offline_audit_summary.txt`.

## Optional trajectory validation

After the first Computers run, compare it with the existing original CA-LGM:

```text
E:\Anaconda\envs\py310_hydro\python.exe "E:\CODE\LAPT -ONLY\step58_GBA_OFFLINE\validate_trace.py" --reference "E:\CODE\LAPT -ONLY\outputs\step58_CA_LGM\computers\r_0.005\pilot_600\v1_lgm_w005" --trace "E:\CODE\LAPT -ONLY\outputs\step58_GBA_OFFLINE\computers\r_0.005\seed_1\trace"
```

The report compares loss rows, selected adjacency, selected features, and F1.
