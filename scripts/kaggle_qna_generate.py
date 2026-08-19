"""Generate Q&A answers from a prompt batch on GPU hardware (Kaggle/Colab).

Copy the body of this file into a Kaggle notebook cell, or add it as a
Kaggle *utility script* and call ``main()``. It is intentionally
dependency-light (transformers + torch only) and does NOT import ``mvr`` --
the GPU host only needs the prompt batch JSON, never the repository or any
competition media.

Input : prompts.json  (format "mvr-qna-prompt-batch-v1", from
        `python scripts/run_qna_pipeline.py --export-prompts ...`)
Output: answers.json  (format "mvr-qna-answer-batch-v1", consumed by
        `python scripts/import_qna_answers.py`)

The output stores each model reply verbatim as ``raw_output``. Parsing and
validation happen locally in ``mvr.enrichment.qna_batch`` so there is one
source of truth for the answer schema, and so a bad reply can be inspected
rather than silently reshaped here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    # Kaggle/Colab are UTF-8 already; this only matters when the script is
    # exercised from a Windows console, whose legacy codepage cannot encode
    # Vietnamese question/answer text.
    sys.stdout.reconfigure(encoding="utf-8")

PROMPT_BATCH_FORMAT = "mvr-qna-prompt-batch-v1"
ANSWER_BATCH_FORMAT = "mvr-qna-answer-batch-v1"

# Qwen2.5-7B-Instruct has solid Vietnamese and fits T4 x2 (2x16GB) in
# bfloat16. On a single 16GB GPU use load_in_4bit=True, or drop to
# Qwen/Qwen2.5-3B-Instruct.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 256
RETRIES = 1  # one deterministic retry when the reply is not valid JSON


def load_model(model_name: str = DEFAULT_MODEL, load_in_4bit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"dtype": torch.bfloat16, "device_map": "auto", "low_cpu_mem_usage": True}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    generated = output_ids[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _looks_like_answer_json(raw: str) -> bool:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and "answer" in payload and "source" in payload


def run_batch(
    prompts_path: str | Path,
    answers_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    load_in_4bit: bool = False,
    loaded: tuple | None = None,
) -> Path:
    """Answer one prompt batch.

    Pass ``loaded=load_model(...)`` when processing several batches so the
    model is loaded once instead of once per file -- loading a 7B model
    takes minutes, which dominates the run for small batches.
    """
    batch = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    if batch.get("format") != PROMPT_BATCH_FORMAT:
        raise ValueError(f"unexpected prompt batch format: {batch.get('format')!r}")

    print(f"query_id : {batch['query_id']}")
    print(f"question : {batch['question']}")
    print(f"prompts  : {len(batch['items'])}")

    tokenizer, model = loaded or load_model(model_name, load_in_4bit=load_in_4bit)

    items = []
    started = time.time()
    for position, item in enumerate(batch["items"], start=1):
        raw = generate(tokenizer, model, item["prompt"])
        for _ in range(RETRIES):
            if _looks_like_answer_json(raw):
                break
            raw = generate(
                tokenizer,
                model,
                item["prompt"]
                + '\nYour previous reply was not valid JSON. Output ONLY: '
                '{"answer": "...", "source": "context-grounded"}',
            )
        items.append({"index": item["index"], "raw_output": raw})
        print(f"[{position}/{len(batch['items'])}] {raw[:110]}")

    payload = {
        "format": ANSWER_BATCH_FORMAT,
        "query_id": batch["query_id"],
        "model": model_name,
        "items": items,
    }
    answers_path = Path(answers_path)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    unparsed = sum(1 for row in items if not _looks_like_answer_json(row["raw_output"]))
    print(f"\ndone in {time.time() - started:.0f}s -> {answers_path}")
    if unparsed:
        print(f"WARNING: {unparsed}/{len(items)} replies still unparseable; "
              "those candidates will be dropped on import")
    return answers_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Q&A answers on GPU")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    run_batch(args.prompts, args.answers, args.model, load_in_4bit=args.load_in_4bit)
