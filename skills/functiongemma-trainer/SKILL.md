---
name: functiongemma-trainer
description: Fine-tune FunctionGemma for on-device function calling using SFT on Hugging Face Jobs. Handles FunctionGemma-specific prompt formatting, CSV dataset validation, training on HF GPUs, evaluation of function-calling accuracy, and LiteRT-LM export for Android deployment. Use this skill when training FunctionGemma models for custom tool schemas.
---

# FunctionGemma Trainer

Fine-tune Google's FunctionGemma (270M) for custom on-device function calling. This skill handles the full pipeline: dataset validation, FunctionGemma prompt formatting, SFT training on HF Jobs, evaluation, and LiteRT-LM export for Android.

## Overview

FunctionGemma is a 270M parameter model built on Gemma 3, fine-tuned for function calling. It achieves 58% accuracy zero-shot and 70-85% after fine-tuning on domain-specific tools (depending on data quality and training epochs). This skill automates the fine-tuning process for your custom tool schemas.

**Key facts:**
- Base model: `google/functiongemma-270m-it`
- Size: 288 MB (dynamic int8 quantization)
- Latency: 0.3s time-to-first-token on Samsung S25 Ultra
- Fine-tuning is NOT optional — 58% base accuracy is not production-ready
- Export format: LiteRT-LM (`.litertlm`) for Android, not GGUF

## Real-World Results

**Cycling Copilot** (February 2026):
- Dataset: 942 examples across 6 tools
- Training: 3 epochs, ~21 min on t4-small (~$0.14)
- Results: **70.1% combined accuracy** (tool + arguments)
  - Tool selection: 78.1%
  - Argument accuracy: 71.1%
  - Best tool: get_segment_ahead (83.0%)
  - Weakest tool: get_route_alternatives (54.1%)
- Export: 284 MB `.litertlm` file ready for Android
- Model: https://huggingface.co/monday8am/cycling-copilot-functiongemma

**Key lesson**: Initial training with `max_length=512` caused 0% eval accuracy due to prompt truncation. FunctionGemma tool schemas can be ~825 tokens, requiring `max_length=1280+`.

## Prerequisites

- Hugging Face Pro account (for Jobs access)
- Write-access HF token: `hf auth login` or `export HF_TOKEN=hf_xxx`
- Accept the FunctionGemma license at https://huggingface.co/google/functiongemma-270m-it
- Dataset uploaded to HF Hub in the expected CSV format

## Dataset Format

The training dataset must be a CSV on HF Hub with exactly two columns:

```csv
user_message,tool_calls
"What's the weather?","[{""name"": ""get_weather"", ""args"": {""query"": ""current""}}]"
"Find a coffee shop","[{""name"": ""find_poi"", ""args"": {""query"": ""cafe""}}]"
```

**Rules:**
- `user_message`: Natural language input, quoted
- `tool_calls`: JSON array with exactly ONE tool call object containing `name` and `args`
- Each tool should have a single string parameter called `query` for maximum compatibility
- No empty tool calls (`[]`) — every row must map to a tool
- Recommended: 400-500 examples, ~10 per tool per variation type

### Validate a dataset before training

Use the validation script to check format before spending GPU time:

```bash
uv run validate_functiongemma_dataset.py \
  --dataset USERNAME/DATASET_NAME \
  --tools path/to/tools.json \
  --split train
```

## Tool Schema Format

Tools must follow this JSON format. Every tool takes a single `query` string parameter:

```json
[
  {
    "type": "function",
    "function": {
      "name": "tool_name",
      "description": "What this tool does and when to call it.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Parameter description with valid values"
          }
        },
        "required": ["query"]
      },
      "return": {
        "type": "string"
      }
    }
  }
]
```

## FunctionGemma Prompt Format

FunctionGemma uses a specific prompt structure during training and inference. The skill handles this automatically, but for reference:

```
<start_of_turn>system
You are a helpful assistant with access to the following tools:

[TOOL_SCHEMAS_JSON]

When a tool is needed, respond ONLY with the tool call in this format:
[{"name": "function_name", "args": {"query": "value"}}]
<end_of_turn>
<start_of_turn>user
USER_MESSAGE
<end_of_turn>
<start_of_turn>model
TOOL_CALLS_JSON
<end_of_turn>
```

The training script formats each CSV row into this structure automatically.

## Training

### Hardware Selection

FunctionGemma is 270M parameters. Use the cheapest GPU available:

| Hardware | Hourly Cost | Training Time (~500 examples) | Total Cost |
|----------|------------|-------------------------------|------------|
| t4-small | $0.40/hr | ~20-30 min | ~$0.15-0.20 |
| l4-small | $0.80/hr | ~10-15 min | ~$0.15-0.20 |

**Always use `t4-small` unless you have a reason not to.** FunctionGemma is tiny and trains fast.

### Run Training

```bash
hf jobs run \
  --flavor t4-small \
  --timeout 1h \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run train_functiongemma.py \
    --dataset USERNAME/DATASET_NAME \
    --tools path/to/tools.json \
    --output-repo USERNAME/MODEL_NAME \
    --epochs 3 \
    --lr 2e-4 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --max-length 1280
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | required | HF dataset repo with CSV |
| `--tools` | required | Path to tool schemas JSON |
| `--output-repo` | required | HF repo for the fine-tuned model |
| `--epochs` | 3 | Number of training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--batch-size` | 1 | Training batch size (use 1 with longer sequences) |
| `--gradient-accumulation-steps` | 8 | Gradient accumulation steps (effective batch size = batch-size × this) |
| `--max-length` | 1280 | Max sequence length (**CRITICAL**: must be >925 tokens for FunctionGemma with tool schemas) |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--test-split` | 0.1 | Fraction of data for evaluation |
| `--trackio-project` | none | Optional Trackio project name for monitoring |

### ⚠️ Critical Parameter: max-length

**The `--max-length` parameter is CRITICAL for FunctionGemma training.** If set too low, it will catastrophically truncate training examples, causing fake high training accuracy but 0% evaluation accuracy.

For the cycling copilot tool schemas:
- Tool schemas JSON: ~825 tokens
- System prompt: ~50 tokens
- User message: ~20 tokens
- Model response: ~30 tokens
- **Total prompt length: ~925+ tokens**

**Never use `--max-length=512` with FunctionGemma.** Always use at least 1280 tokens.

Symptoms of incorrect max_length:
- Training shows 100% accuracy but evaluation shows 0%
- Model generates natural language instead of tool calls
- All training examples are truncated before the tool call response

### Quick Test Run

Always test before a full run:

```bash
hf jobs run \
  --flavor t4-small \
  --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run train_functiongemma.py \
    --dataset USERNAME/DATASET_NAME \
    --tools path/to/tools.json \
    --output-repo USERNAME/MODEL_NAME-test \
    --epochs 1 \
    --max-examples 50
```

### Monitor Training

Check job status:
```bash
hf jobs logs JOB_ID
```

If Trackio is configured, view real-time metrics at:
```
https://huggingface.co/spaces/USERNAME/trackio
```

## Evaluation

After training, evaluate function-calling accuracy:

```bash
uv run evaluate_functiongemma.py \
  --model USERNAME/MODEL_NAME \
  --dataset USERNAME/DATASET_NAME \
  --tools path/to/tools.json \
  --split test
```

This reports:
- **Tool selection accuracy**: Did the model pick the right tool?
- **Argument accuracy**: Did the model pass the correct query value?
- **Combined accuracy**: Both tool and argument correct
- **Per-tool breakdown**: Accuracy for each individual tool

Target: 85%+ combined accuracy after fine-tuning (vs 58% base).

## LiteRT-LM Export

Convert the fine-tuned model to `.litertlm` for Android deployment. This uses Google's `ai-edge-torch` library, which:
1. Merges LoRA adapters (if applicable)
2. Builds the PyTorch model via `gemma3.build_model_270m()`
3. Converts to `.litertlm` in a single call via `converter.convert_to_litert()`

The conversion writes a FunctionGemma-specific metadata textproto that configures stop tokens (`<end_of_turn>` and `<start_function_response>`) and model type (`function_gemma`).

**By default, the `.litertlm` file is uploaded to the same repo as the model** (consolidated structure). You can optionally specify a different `--output-repo` if needed.

**Important**: This requires `ai-edge-torch-nightly` and `ai-edge-litert-nightly`. The checkpoint directory must contain a `tokenizer.model` file (SentencePiece format).

```bash
# Export to same repo (recommended - consolidated structure)
hf jobs run \
  --flavor t4-small \
  --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run export_litertlm.py \
    --model USERNAME/MODEL_NAME \
    --output-name-prefix cycling-copilot

# Or export to different repo (optional)
hf jobs run \
  --flavor t4-small \
  --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run export_litertlm.py \
    --model USERNAME/MODEL_NAME \
    --output-repo USERNAME/EXPORT_REPO \
    --output-name-prefix cycling-copilot
```

The exported model can be loaded with the LiteRT-LM Android SDK or deployed via the Google AI Edge Gallery app.

### Export Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | required | HF repo of the fine-tuned model |
| `--output-repo` | same as --model | HF repo for LiteRT-LM export (default: same repo as model) |
| `--output-name-prefix` | cycling-copilot | Prefix for output files |
| `--prefill-seq-len` | 256 | Prefill sequence length |
| `--kv-cache-max-len` | 1024 | KV cache max length (context window) |
| `--quantize` | dynamic_int8 | Quantization method |

## Full Pipeline Example

```bash
# 1. Validate dataset
uv run validate_functiongemma_dataset.py \
  --dataset monday8am/cycling-copilot-dataset \
  --tools tools.json

# 2. Quick test run
hf jobs run --flavor t4-small --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run train_functiongemma.py \
    --dataset monday8am/cycling-copilot-dataset \
    --tools tools.json \
    --output-repo monday8am/cycling-copilot-functiongemma-test \
    --epochs 1 --max-examples 50

# 3. Full training
hf jobs run --flavor t4-small --timeout 1h \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run train_functiongemma.py \
    --dataset monday8am/cycling-copilot-dataset \
    --tools tools.json \
    --output-repo monday8am/cycling-copilot-functiongemma \
    --epochs 3

# 4. Evaluate
uv run evaluate_functiongemma.py \
  --model monday8am/cycling-copilot-functiongemma \
  --dataset monday8am/cycling-copilot-dataset \
  --tools tools.json

# 5. Export for Android
hf jobs run --flavor t4-small --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run export_litertlm.py \
    --model monday8am/cycling-copilot-functiongemma \
    --output-name-prefix cycling-copilot
```

## Troubleshooting

**"Model not found" error**: Make sure you accepted the FunctionGemma license at https://huggingface.co/google/functiongemma-270m-it

**High training accuracy but 0% eval accuracy**: This is almost always caused by `max_length` being too small. If your tool schemas are ~825 tokens and you use `max_length=512`, all training examples are truncated before the model response. The model learns the system prompt but never sees tool calls. **Solution**: Use `--max-length=1280` (or higher).

**Low accuracy after training (but eval is working)**: Check that your dataset has enough examples per tool (minimum 30-40 each). Verify tool descriptions are clear and non-overlapping. Try increasing epochs to 5-10 or adding more training data.

**Out of memory**: With `max_length=1280`, you may need `--batch-size=1` on t4-small. Use `--gradient-accumulation-steps=8` to maintain effective batch size.

**LiteRT-LM export fails**: Ensure `ai-edge-torch-nightly` and `ai-edge-litert-nightly` are installed. Check that `tokenizer.model` exists in the checkpoint directory. The conversion requires a GPU runtime.

**Missing tokenizer.model**: FunctionGemma checkpoints from HF Hub should include `tokenizer.model`. If your fine-tuned model doesn't have it, copy it from the base `google/functiongemma-270m-it` checkpoint.

## References

- [FunctionGemma overview](https://ai.google.dev/gemma/docs/functiongemma)
- [FunctionGemma model card](https://ai.google.dev/gemma/docs/functiongemma/model_card)
- [Official fine-tuning + export notebook](https://github.com/google-gemini/gemma-cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb)
- [HF-to-MediaPipe conversion guide](https://ai.google.dev/gemma/docs/conversions/hf-to-mediapipe-task)
- [LiteRT-LM documentation](https://github.com/google-ai-edge/LiteRT-LM)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [HF Jobs documentation](https://huggingface.co/docs/hub/jobs-overview)
