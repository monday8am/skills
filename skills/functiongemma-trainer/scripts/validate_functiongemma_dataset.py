# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets",
# ]
# ///
"""
FunctionGemma Dataset Validator
===============================
Validates a CSV dataset for FunctionGemma fine-tuning.
Checks column format, JSON validity, tool name coverage, and parameter values.
"""

import argparse
import json
import sys
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Validate FunctionGemma dataset")
    parser.add_argument("--dataset", required=True, help="HF dataset repo")
    parser.add_argument("--tools", required=True, help="Path to tool schemas JSON")
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load tools
    with open(args.tools) as f:
        tools = json.load(f)
    valid_tool_names = {t["function"]["name"] for t in tools}

    # Load dataset
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Dataset: {args.dataset} ({len(ds)} rows)")
    print(f"Columns: {ds.column_names}")

    # Check columns
    required = {"user_message", "tool_calls"}
    if not required.issubset(set(ds.column_names)):
        print(f"FAIL: Missing columns. Expected {required}, got {set(ds.column_names)}")
        sys.exit(1)

    errors = []
    tool_counts = {name: 0 for name in valid_tool_names}
    unknown_tools = set()

    for i, row in enumerate(ds):
        # Check user_message
        if not row["user_message"] or not row["user_message"].strip():
            errors.append(f"Row {i}: Empty user_message")
            continue

        # Check tool_calls JSON
        try:
            calls = json.loads(row["tool_calls"])
        except json.JSONDecodeError as e:
            errors.append(f"Row {i}: Invalid JSON in tool_calls: {e}")
            continue

        # Check it's a list
        if not isinstance(calls, list):
            errors.append(f"Row {i}: tool_calls must be a JSON array, got {type(calls)}")
            continue

        # Check exactly one tool call
        if len(calls) == 0:
            errors.append(f"Row {i}: Empty tool_calls array (every row must have a tool call)")
            continue

        if len(calls) > 1:
            errors.append(f"Row {i}: Multiple tool calls ({len(calls)}), expected exactly 1")
            continue

        call = calls[0]

        # Check structure
        if "name" not in call:
            errors.append(f"Row {i}: Tool call missing 'name' field")
            continue
        if "args" not in call:
            errors.append(f"Row {i}: Tool call missing 'args' field")
            continue

        # Check tool name
        if call["name"] not in valid_tool_names:
            errors.append(f"Row {i}: Unknown tool '{call['name']}'")
            unknown_tools.add(call["name"])
            continue

        tool_counts[call["name"]] += 1

        # Check args has query field
        if "query" not in call["args"]:
            errors.append(f"Row {i}: Tool args missing 'query' field")

    # Report
    print(f"\n{'='*50}")
    print("VALIDATION REPORT")
    print(f"{'='*50}")
    print(f"Total rows: {len(ds)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"\nFirst 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print(f"\nTool distribution:")
    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        pct = count / len(ds) * 100 if len(ds) > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {name:30s} {count:4d} ({pct:5.1f}%) {bar}")

    if unknown_tools:
        print(f"\nUnknown tools found: {', '.join(sorted(unknown_tools))}")

    # Warnings
    low_count_tools = [n for n, c in tool_counts.items() if c < 30]
    if low_count_tools:
        print(f"\nWARNING: Tools with <30 examples (may underfit):")
        for t in low_count_tools:
            print(f"  {t}: {tool_counts[t]} examples")

    if len(errors) == 0:
        print(f"\n DATASET READY for FunctionGemma fine-tuning")
    else:
        print(f"\n DATASET HAS ERRORS — fix before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
