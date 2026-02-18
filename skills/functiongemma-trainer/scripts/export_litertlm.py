# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.57.0",
#   "peft",
#   "huggingface_hub",
#   "ai-edge-torch-nightly",
#   "ai-edge-litert-nightly",
# ]
# ///
"""
FunctionGemma LiteRT-LM Export Script
=====================================
Converts a fine-tuned FunctionGemma model to .litertlm format
for on-device Android deployment via the LiteRT-LM SDK.

Pipeline (matches the official gemma-cookbook notebook):
  1. Merge LoRA adapters into base model (if applicable)
  2. Write FunctionGemma metadata textproto
  3. Build PyTorch model via gemma3.build_model_270m()
  4. Convert to .litertlm via converter.convert_to_litert() (single call)
  5. Upload to HF Hub (by default, to the same repo as the model)

Usage:
  # Export to same repo as model (recommended - consolidated structure)
  python export_litertlm.py --model USER/MODEL

  # Export to different repo (optional)
  python export_litertlm.py --model USER/MODEL --output-repo USER/EXPORT_REPO

Reference:
  google-gemini/gemma-cookbook/FunctionGemma/
  [FunctionGemma]Finetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import login, HfApi, upload_folder, hf_hub_download
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = "google/functiongemma-270m-it"

# FunctionGemma-specific metadata for the .litertlm bundle.
# This tells the LiteRT-LM runtime how to handle the model:
#   - start_token: BOS token id (2 for Gemma)
#   - stop_tokens: <end_of_turn> (normal stop) and <start_function_response>
#     (stops generation so the app can execute the function call)
#   - llm_model_type: function_gemma (enables FunctionGemma-specific behavior)
FUNCTIONGEMMA_METADATA = r"""start_token: {
    token_ids: {
        ids: [ 2 ]
    }
}
stop_tokens: {
    token_str: "<end_of_turn>"
}
stop_tokens: {
    token_str: "<start_function_response>"
}
llm_model_type: {
    function_gemma: {}
}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Export FunctionGemma to LiteRT-LM")
    parser.add_argument("--model", required=True,
                        help="HF repo of fine-tuned model (or local path)")
    parser.add_argument("--output-repo", required=False, default=None,
                        help="HF repo for LiteRT-LM export (default: same as --model)")
    parser.add_argument("--output-name-prefix", default="cycling-copilot",
                        help="Prefix for output files (default: cycling-copilot)")
    parser.add_argument("--prefill-seq-len", type=int, default=256,
                        help="Prefill sequence length (default: 256)")
    parser.add_argument("--kv-cache-max-len", type=int, default=1024,
                        help="KV cache max length (default: 1024)")
    parser.add_argument("--quantize", default="dynamic_int8",
                        choices=["dynamic_int8"])
    return parser.parse_args()


def step1_merge_lora(model_repo: str, output_dir: Path) -> Path:
    """Merge LoRA adapters into base model and save as full checkpoint."""
    print("=" * 60)
    print("STEP 1: Prepare checkpoint")
    print("=" * 60)

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_repo)

    try:
        # Try loading as LoRA adapter on top of base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, model_repo)
        print("  Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
    except Exception:
        # If not a LoRA model, load directly (full fine-tune)
        print("  No LoRA adapters found, loading as full model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_repo, torch_dtype=torch.float32,
        )

    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    # Ensure tokenizer.model (SentencePiece) exists — needed for LiteRT-LM
    tokenizer_model_path = merged_dir / "tokenizer.model"
    if not tokenizer_model_path.exists():
        print("  tokenizer.model missing, downloading from base model...")
        src = hf_hub_download(BASE_MODEL, "tokenizer.model")
        shutil.copy(src, tokenizer_model_path)
        print(f"  Copied tokenizer.model from {BASE_MODEL}")

    print(f"  Checkpoint saved to {merged_dir}")
    return merged_dir


def step2_convert_to_litertlm(
    checkpoint_dir: Path,
    output_dir: Path,
    output_name_prefix: str,
    prefill_seq_len: int,
    kv_cache_max_len: int,
    quantize: str,
) -> Path:
    """Convert checkpoint to .litertlm using the official ai-edge-torch pipeline."""
    print("=" * 60)
    print("STEP 2: Convert to .litertlm")
    print("=" * 60)

    from ai_edge_torch.generative.examples.gemma3 import gemma3
    from ai_edge_torch.generative.utilities import converter
    from ai_edge_torch.generative.utilities.export_config import ExportConfig
    from ai_edge_torch.generative.layers import kv_cache

    litertlm_dir = output_dir / "litertlm"
    litertlm_dir.mkdir(parents=True, exist_ok=True)

    # Write the FunctionGemma metadata textproto
    metadata_path = litertlm_dir / "base_llm_metadata.textproto"
    with open(metadata_path, "w") as f:
        f.write(FUNCTIONGEMMA_METADATA)
    print(f"  Wrote metadata to {metadata_path}")

    # Build the PyTorch model from the checkpoint
    print(f"  Building PyTorch model from {checkpoint_dir}...")
    pytorch_model = gemma3.build_model_270m(str(checkpoint_dir))

    # Setup export configuration
    export_config = ExportConfig()
    export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
    export_config.mask_as_input = True

    # Locate the tokenizer.model file
    tokenizer_model_path = checkpoint_dir / "tokenizer.model"
    if not tokenizer_model_path.exists():
        raise FileNotFoundError(
            f"tokenizer.model not found at {tokenizer_model_path}. "
            "FunctionGemma requires the SentencePiece tokenizer.model file."
        )

    # Single-call conversion to .litertlm
    print(f"  Converting to .litertlm (quantize={quantize}, "
          f"kv_cache={kv_cache_max_len}, prefill={prefill_seq_len})...")
    converter.convert_to_litert(
        pytorch_model,
        output_path=str(litertlm_dir),
        output_name_prefix=output_name_prefix,
        prefill_seq_len=prefill_seq_len,
        kv_cache_max_len=kv_cache_max_len,
        quantize=quantize,
        export_config=export_config,
        tokenizer_model_path=str(tokenizer_model_path),
        base_llm_metadata_path=str(metadata_path),
        output_format="litertlm",
    )

    # Find the output file
    expected_name = f"{output_name_prefix}_q8_ekv{kv_cache_max_len}.litertlm"
    litertlm_path = litertlm_dir / expected_name
    if litertlm_path.exists():
        size_mb = litertlm_path.stat().st_size / (1024 ** 2)
        print(f"  Bundle created: {litertlm_path} ({size_mb:.2f} MB)")
        return litertlm_path

    # Fallback: find any .litertlm file
    litertlm_files = list(litertlm_dir.glob("*.litertlm"))
    if litertlm_files:
        litertlm_path = litertlm_files[0]
        size_mb = litertlm_path.stat().st_size / (1024 ** 2)
        print(f"  Bundle created: {litertlm_path} ({size_mb:.2f} MB)")
        return litertlm_path

    raise FileNotFoundError("No .litertlm file produced — check logs above")


def step3_upload(output_dir: Path, output_repo: str, hf_token: str):
    """Upload .litertlm and metadata to HF Hub."""
    print("=" * 60)
    print("STEP 3: Upload to HF Hub")
    print("=" * 60)

    litertlm_dir = output_dir / "litertlm"
    api = HfApi()
    api.create_repo(output_repo, exist_ok=True, token=hf_token)
    upload_folder(
        folder_path=str(litertlm_dir),
        repo_id=output_repo,
        token=hf_token,
    )
    print(f"  Uploaded to https://huggingface.co/{output_repo}")


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Default output repo to same as input model (consolidated structure)
    output_repo = args.output_repo or args.model

    output_dir = Path("/tmp/litertlm-export")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline
    checkpoint_dir = step1_merge_lora(args.model, output_dir)
    litertlm_path = step2_convert_to_litertlm(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        output_name_prefix=args.output_name_prefix,
        prefill_seq_len=args.prefill_seq_len,
        kv_cache_max_len=args.kv_cache_max_len,
        quantize=args.quantize,
    )
    step3_upload(output_dir, output_repo, hf_token)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print(f"  Model:  {args.model}")
    print(f"  Output: {litertlm_path.name}")
    print(f"  Hub:    https://huggingface.co/{output_repo}")
    if output_repo == args.model:
        print(f"  Note:   LiteRT-LM file added to same repo as model")
    print("=" * 60)


if __name__ == "__main__":
    main()
