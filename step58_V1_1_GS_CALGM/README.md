# Step58-V1.1 Gradient-Safe CA-LGM

This version begins with a non-intervening full PGE-gradient trajectory audit.
It follows the original HyDRO optimizer and records all 300 PGE update points
for a 120-epoch Computers run.  No link loss is added during this stage.

PyCharm script:

`run_pipeline.py`

Parameters:

`--dataset computers --rate 0.005 --seed 1 --gpu 0 --epochs 120 --rho 0.02 --lambda-max 0.05`

Outputs are CSV, TXT, and LOG files; no JSON is produced.
