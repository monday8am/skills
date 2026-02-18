# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.57.0",
#   "datasets",
#   "peft",
#   "huggingface_hub",
# ]
# ///
"""
FunctionGemma Evaluation Script
================================
Evaluates a fine-tuned FunctionGemma model on function-calling accuracy.
Reports tool selection accuracy, argument accuracy, and per-tool breakdown.
"""

import argparse
import json
import os
import urllib.request
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FunctionGemma")
    parser.add_argument("--model", required=True, help="HF model repo")
    parser.add_argument("--dataset", required=True, help="HF dataset repo")
    parser.add_argument("--tools", default=None, help="Local path to tool schemas JSON")
    parser.add_argument("--tools-url", default=None, help="URL to tool schemas JSON (for HF Jobs)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def build_inference_prompt(user_message: str, tools_json: str) -> str:
    """Build prompt for inference (no model turn)."""
    return (
        f"<start_of_turn>system\n"
        f"You are a helpful assistant with access to the following tools:\n\n"
        f"{tools_json}\n\n"
        f"When a tool is needed, respond ONLY with the tool call in this format:\n"
        f'[{{"name": "function_name", "args": {{"query": "value"}}}}]\n'
        f"<end_of_turn>\n"
        f"<start_of_turn>user\n"
        f"{user_message}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def parse_model_output(output: str):
    """Try to parse model output as a tool call."""
    output = output.strip()
    # Remove any trailing tokens
    if "<end_of_turn>" in output:
        output = output[:output.index("<end_of_turn>")]
    output = output.strip()

    try:
        calls = json.loads(output)
        if isinstance(calls, list) and len(calls) > 0:
            return calls[0]
    except json.JSONDecodeError:
        pass
    return None


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load tools
    if args.tools:
        with open(args.tools) as f:
            tools = json.load(f)
    elif args.tools_url:
        print(f"Fetching tool schemas from {args.tools_url}...")
        with urllib.request.urlopen(args.tools_url) as resp:
            tools = json.loads(resp.read().decode())
    else:
        raise ValueError("Either --tools (local path) or --tools-url (URL) is required")
    tools_json = json.dumps(tools, indent=2)

    # Load dataset
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    # Load model (supports both LoRA adapters and full models)
    BASE_MODEL = "google/functiongemma-270m-it"
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    try:
        # Load base model first, then apply LoRA adapter manually
        # This avoids AutoPeftModelForCausalLM's embedding resize which
        # corrupts the embedding matrix with random values
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, args.model)
        model = model.merge_and_unload()
        print("  Loaded as PEFT/LoRA model (merged into base)")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        print("  Loaded as full model")
    model.eval()

    # Evaluate
    results = {
        "total": 0,
        "tool_correct": 0,
        "args_correct": 0,
        "combined_correct": 0,
        "parse_failures": 0,
        "per_tool": {},
    }

    for row in ds:
        results["total"] += 1

        # Parse expected
        expected = json.loads(row["tool_calls"])[0]
        expected_tool = expected["name"]
        expected_args = expected["args"]

        # Initialize per-tool tracking
        if expected_tool not in results["per_tool"]:
            results["per_tool"][expected_tool] = {"total": 0, "correct": 0}
        results["per_tool"][expected_tool]["total"] += 1

        # Generate
        prompt = build_inference_prompt(row["user_message"], tools_json)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        predicted = parse_model_output(generated)

        if predicted is None:
            results["parse_failures"] += 1
            if results["parse_failures"] <= 5:
                print(f"  PARSE FAIL #{results['parse_failures']}:")
                print(f"    Input: {row['user_message'][:80]}")
                print(f"    Expected: {row['tool_calls'][:80]}")
                print(f"    Generated: {repr(generated[:200])}")
            continue

        # Check tool name
        tool_match = predicted.get("name") == expected_tool
        if tool_match:
            results["tool_correct"] += 1

        # Check args
        args_match = predicted.get("args") == expected_args
        if args_match:
            results["args_correct"] += 1

        # Combined
        if tool_match and args_match:
            results["combined_correct"] += 1
            results["per_tool"][expected_tool]["correct"] += 1

    # Report
    total = results["total"]
    print(f"\n{'='*50}")
    print("EVALUATION REPORT")
    print(f"{'='*50}")
    print(f"Total examples: {total}")

    if total == 0:
        print("No examples to evaluate.")
        return

    print(f"Parse failures: {results['parse_failures']}")
    print(f"")
    print(f"Tool selection accuracy: {results['tool_correct']}/{total} ({results['tool_correct']/total*100:.1f}%)")
    print(f"Argument accuracy:       {results['args_correct']}/{total} ({results['args_correct']/total*100:.1f}%)")
    print(f"Combined accuracy:       {results['combined_correct']}/{total} ({results['combined_correct']/total*100:.1f}%)")
    print(f"")
    print(f"Per-tool breakdown:")
    for tool, stats in sorted(results["per_tool"].items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {tool:30s} {stats['correct']:3d}/{stats['total']:3d} ({acc:5.1f}%)")

    target_met = results["combined_correct"] / total >= 0.85
    print(f"\nTarget (85%): {'MET' if target_met else 'NOT MET'}")


if __name__ == "__main__":
    main()
