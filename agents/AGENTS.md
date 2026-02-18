# monday8am Skills — Agent Instructions

This file aggregates all skills for agents that use a single instruction file (e.g., OpenAI Codex).

## Available Skills

### functiongemma-trainer

Fine-tune Google's FunctionGemma (270M) for on-device function calling. Full pipeline: dataset validation, SFT training on HF Jobs, evaluation, and LiteRT-LM export for Android.

See [skills/functiongemma-trainer/SKILL.md](../skills/functiongemma-trainer/SKILL.md) for complete instructions.

**Quick reference:**

```bash
# Validate dataset
uv run validate_functiongemma_dataset.py --dataset USERNAME/DATASET --tools tools.json

# Train on HF Jobs
hf jobs run --flavor t4-small --timeout 1h \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run train_functiongemma.py \
    --dataset USERNAME/DATASET --tools tools.json \
    --output-repo USERNAME/MODEL --epochs 3

# Evaluate
uv run evaluate_functiongemma.py --model USERNAME/MODEL --dataset USERNAME/DATASET --tools tools.json

# Export for Android
hf jobs run --flavor t4-small --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run export_litertlm.py --model USERNAME/MODEL
```

Key: always use `--max-length=1280+` for FunctionGemma (tool schemas alone are ~825 tokens).
