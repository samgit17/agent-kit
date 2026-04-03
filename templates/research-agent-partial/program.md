## backend
# Options: web | ml_experiment
web

## goal
What are the best practices for securing multi-agent LLM systems in 2025?

## success_criteria
- At least 5 credible sources
- Prefer sources from last 6 months
- Focus on agentic attack surfaces, not general LLM safety

## constraints
max_iterations: 3

---
# ML EXPERIMENT SETTINGS (only used when backend: ml_experiment)
# Uncomment and set backend to ml_experiment to use

# ## backend
# ml_experiment

# ## goal
# Improve val_bpb on nanochat. Baseline: measure first.

# ## directions
# - Learning rate schedule (cosine vs flat vs warmup-cosine)
# - DEPTH param (4 vs 6 vs 8) — stay within VRAM budget for selected gpu
# - Batch size (8, 16, 32)
# - Muon vs AdamW optimizer

# ## constraints
# gpu: 0                        # CUDA device index — 0=4070 Super, 1=5060 Ti
# minutes_per_experiment: 10    # 10min for home lab (vs 5min on H100)
# max_experiments: 20
# revert_on_no_improvement: true
# vram_budget_gb: 12            # executor will warn if estimated usage exceeds this
