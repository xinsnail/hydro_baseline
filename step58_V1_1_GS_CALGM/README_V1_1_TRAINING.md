# Step58-V1.1 Gradient-Safe CA-LGM Training

This training branch keeps the original HyDRO gradient intact and adds only a
small link-task correction during PGE updates.

- Link objective: positive/negative direction-only CA-LGM.
- Conflict control: project the link gradient against the HyDRO gradient when
  their dot product is negative.
- Injection budget: the projected link gradient is scaled to 2% of the HyDRO
  gradient norm (`rho=0.02`).
- Scale safety bound: `max_scale=5.0`; the completed baseline audit showed the
  required scale remained below 4.0.
- Checkpoint contract: GNN training uses normalized propagation, while saved
  checkpoints contain the raw PGE adjacency expected by the evaluator.
- No RankGM and no propagation-budget penalty are used in this version.

The first comparison is a 120-epoch Computers, R=0.005, seed-1 paired pilot.

## Portable command

Run `run_pipeline.py` in PyCharm with:

```text
--stage train --dataset computers --rate 0.005 --seed 1 --gpu 0 --epochs 120 --rho 0.02 --max-scale 5 --eval-interval 60 --eval-runs 10
```

For the full condensation gate, change only `--epochs 120` to
`--epochs 600`.  The output path is generated relative to the shared
`outputs` directory; no machine-specific project path is required.
