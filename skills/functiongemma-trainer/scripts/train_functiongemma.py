# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.57.0",
#   "trl>=0.25.0",
#   "datasets",
#   "peft",
#   "huggingface_hub",
#   "trackio",
# ]
# ///
"""
FunctionGemma Fine-Tuning Script
================================
Fine-tunes google/functiongemma-270m-it on a CSV dataset of
user_message -> tool_calls mappings using SFT with LoRA.

Designed to run as a HF Job via:
  hf jobs uv run --flavor t4-small --secrets HF_TOKEN \\
    "https://huggingface.co/USER/REPO/resolve/main/train_functiongemma.py" \\
    -- --dataset USER/DATASET --tools-url USER/REPO/resolve/main/tools.json \\
    --output-repo USER/MODEL --epochs 3
"""

import argparse
import json
import os
import urllib.request

import trackio
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FunctionGemma")
    parser.add_argument("--dataset", required=True, help="HF dataset repo (CSV format)")
    parser.add_argument("--tools", default=None, help="Local path to tool schemas JSON file")
    parser.add_argument("--tools-url", default=None, help="URL to tool schemas JSON file (for HF Jobs)")
    parser.add_argument("--output-repo", required=True, help="HF repo for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1280)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples for test runs")
    parser.add_argument("--trackio-project", type=str, default=None)
    parser.add_argument("--trackio-run-name", type=str, default=None)
    parser.add_argument("--trackio-space-id", type=str, default=None, help="HF Space for Trackio dashboard")
    return parser.parse_args()


def load_tools(tools_path: str = None, tools_url: str = None) -> str:
    """Load tool schemas from local path or URL, return as formatted JSON string."""
    if tools_path:
        with open(tools_path, "r") as f:
            tools = json.load(f)
    elif tools_url:
        print(f"Fetching tool schemas from {tools_url}...")
        with urllib.request.urlopen(tools_url) as resp:
            tools = json.loads(resp.read().decode())
    else:
        raise ValueError("Either --tools (local path) or --tools-url (URL) is required")
    return json.dumps(tools, indent=2)


def format_functiongemma_prompt(user_message: str, tool_calls: str, tools_json: str) -> str:
    """Format a single example into FunctionGemma's expected prompt structure."""
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
        f"{tool_calls}\n"
        f"<end_of_turn>"
    )


def prepare_dataset(dataset_repo: str, tools_json: str, test_split: float, max_examples: int = None):
    """Load CSV dataset and format into FunctionGemma prompts."""
    ds = load_dataset(dataset_repo, split="train")

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    ds = ds.map(
        lambda row: {"text": format_functiongemma_prompt(
            user_message=row["user_message"],
            tool_calls=row["tool_calls"],
            tools_json=tools_json,
        )},
        remove_columns=ds.column_names,
    )

    if test_split > 0:
        split = ds.train_test_split(test_size=test_split, seed=42)
        return split["train"], split["test"]
    return ds, None


def main():
    args = parse_args()

    # Authenticate
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load tools
    tools_json = load_tools(args.tools, args.tools_url)
    print(f"Loaded tool schemas ({len(json.loads(tools_json))} tools)")

    # Prepare dataset
    print(f"Loading dataset from {args.dataset}...")
    train_dataset, eval_dataset = prepare_dataset(
        args.dataset, tools_json, args.test_split, args.max_examples
    )
    print(f"Training examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation examples: {len(eval_dataset)}")

    # Load model and tokenizer
    model_name = "google/functiongemma-270m-it"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Trackio setup
    use_trackio = False
    if args.trackio_project:
        try:
            space_id = args.trackio_space_id or "monday8am/trackio"
            trackio.init(project=args.trackio_project, space_id=space_id)
            use_trackio = True
            print(f"Trackio enabled: {args.trackio_project} -> https://huggingface.co/spaces/{space_id}")
        except Exception as e:
            print(f"Trackio init failed ({e}), continuing without monitoring")

    # Training config
    training_args = SFTConfig(
        output_dir="/tmp/functiongemma-output",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        max_length=args.max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=True,
        hub_model_id=args.output_repo,
        hub_token=hf_token,
        hub_strategy="every_save",
        report_to="trackio" if use_trackio else "none",
        run_name=args.trackio_run_name or args.trackio_project,
        fp16=torch.cuda.is_available(),
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    # Save and push
    print(f"Pushing model to {args.output_repo}...")
    trainer.push_to_hub()

    # Finish Trackio
    if use_trackio:
        trackio.finish()

    print(f"Training complete! Model at: https://huggingface.co/{args.output_repo}")


if __name__ == "__main__":
    main()
