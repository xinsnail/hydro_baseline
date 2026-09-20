# Step58-GBA — Fixed CA-LGM Gradient Behavior Audit

This is a diagnostic replay of the frozen Step58 CA-LGM method. It reuses
the shared `src/`, `data/`, and `step58_CA_LGM/` folders.

## Objective

Measure `r_t = 0.05 * ||g_LP|| / (||g_H|| + eps)` and
`C_t = cos(g_H, g_LP)` on trainable PGE parameters.

## Frozen behavior

- CA-LGM weight remains 0.05.
- No PCGrad, clipping, scaling, RankGM, or LAPT.
- Diagnostics do not modify `.grad` or the optimizer update.
- Training remains 600 epochs with validation every 60 epochs.
- Only `outer_loop=0` is audited in each adjacency-update epoch, producing
  about 120 samples in a 600-epoch replay.

## Representative cases

1. Computers R=0.005 Seed 1 (positive)
2. Computers R=0.005 Seed 2 (negative)
3. Photo R=0.05 Seed 1 (negative)
4. Cora R=1.0 Seed 1 (negative)

## Run from PyCharm

Script path:

`E:\CODE\LAPT -ONLY\step58_GBA\run_pipeline.py`

Run the following parameter sets sequentially on one GPU:

```text
--dataset computers --rate 0.005 --seed 1 --gpu 0 --epochs 600 --lp-weight 0.05 --audit-outer 0 --eval-interval 60 --eval-runs 10
--dataset computers --rate 0.005 --seed 2 --gpu 0 --epochs 600 --lp-weight 0.05 --audit-outer 0 --eval-interval 60 --eval-runs 10
--dataset photo --rate 0.05 --seed 1 --gpu 0 --epochs 600 --lp-weight 0.05 --audit-outer 0 --eval-interval 60 --eval-runs 10
--dataset cora --rate 1.0 --seed 1 --gpu 0 --epochs 600 --lp-weight 0.05 --audit-outer 0 --eval-interval 60 --eval-runs 10
```

There is no `--stage` argument in this launcher.

## Outputs

- `gba_gradient_diagnostics.csv`
- `gba_phase_summary.csv`
- `gba_state_trajectory.csv`
- `trajectory_snapshots/`
- `gba_summary.txt`
- `gba.log`

After all four cases finish, run:

```text
E:\Anaconda\envs\py310_hydro\python.exe "E:\CODE\LAPT -ONLY\step58_GBA\summarize_gba_cases.py"
```

This creates `gba_cross_case_summary.csv` and `.txt` under the Step58-GBA
output root. It intentionally performs no formal correlation test with only
four cases.

The diagnostic values must not be used to change a running case.
