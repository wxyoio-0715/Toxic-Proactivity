# Misalignment Simulation

Codebase for generating misalignment scenarios, running agent/environment simulations, and summarizing results.

## Quick Start

1) Create and activate a virtual environment (optional but recommended).

2) Install dependencies:
```
python -m pip install -r requirements.txt
```

3) Configure API keys (local only):
Edit `config/config.local.yaml` and fill in your keys. This file is ignored by git.

4) Generate a scenario (single):
```
python -m scenario_generation.workflows.misalignment_context_workflow_en --domain Healthcare --category Loyalty
```

5) Run simulation (default: all generated scenarios):
```
python -m misalignment_simulation.main
```

## Configuration

Primary config: `config/config.yaml`  
Local secrets: `config/config.local.yaml` (gitignored)

The workflow supports two modes:
- Base mode (default): `enable_experiments: false`
- Experimental mode: `enable_experiments: true` and set variables

Example (in `config/config.yaml`):
```
workflow:
  enable_experiments: true
  stakes: "high"
  feedback_probability: "medium"
  goal_clarity: "implicit"
  ethical_framework: "utilitarian"
  liability: "agent_liable"
```

## Batch Generation

```
python -m scenario_generation.workflows.misalignment_context_workflow_en --batch-file scenario_generation/workflows/batch.json
```

## Summaries and Plots

Summarize statistics:
```
python -m misalignment_simulation.summarize_statistics --output table
```

Generate plots:
```
python scripts/analyze_misalign.py --csv <path_to_all_runs_summary.csv>
python scripts/plot_stress_final.py --outdir outputs_misalign
```

## Outputs

- Generated scenarios: `scenario_generation/output/`
- Run logs: `logs/`
- Summaries: `logs/statistics/summary/`

## Reproducibility Notes

Pinned dependencies are in `requirements.txt`. Model outputs vary across providers; set model profiles in `config/model_profiles.yaml` and fix model versions where possible.
