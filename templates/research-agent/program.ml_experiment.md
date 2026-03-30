## backend
ml_experiment

## goal
Improve val_bpb on nanochat. Baseline: measure first.

## directions
- Learning rate schedule (cosine vs flat vs warmup_cosine)
- DEPTH param (4 vs 6)
- Batch size (8 vs 16)
- Weight decay values (0.1, 0.01, 0.001)
- Optimizer beta values

## known_constraints
- Do not increase CONTEXT_LEN -- timeouts at 10min budget on home lab GPU

## constraints
gpu: 1
minutes_per_experiment: 10
max_experiments: 20
revert_on_no_improvement: true
vram_budget_gb: 16
