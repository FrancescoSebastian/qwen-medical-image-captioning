"""Clean ImageCLEFmedical captions and prepare the input to paraphrasing."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
PROMPT_PATH = Path(__file__).with_name("cleaning_prompt.txt")
MAX_WORDS = 300


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    """Publish completed JSON files so interrupted writes can be retried."""
    temporary = NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            json.dump(data, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def extract_clean_answer(response):
    start = response.find("<answer>")
    end = response.find("</answer>")
    if start != -1 and end > start:
        return response[start + len("<answer>"):end].strip().strip('"')
    parts = response.split("</think>")
    return (parts[1] if len(parts) > 1 else response).strip().strip('"')


def create_generator(prompt, batch_size):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.generation_config = GenerationConfig.from_pretrained(MODEL_ID)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    device = model.get_input_embeddings().weight.device

    def generate(items, block_number):
        cleaned = {}
        for start in tqdm(range(0, len(items), batch_size), desc=f"Block {block_number}"):
            batch = items[start:start + batch_size]
            # Preserve the original prompt interpolation for string and list values.
            messages = [
                [{"role": "user", "content": prompt.format(caption=caption)}]
                for _, caption in batch
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                padding=True, truncation=True, return_dict=False,
            ).to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id),
                    max_new_tokens=1000, temperature=0.3, do_sample=True,
                )
            responses = tokenizer.batch_decode(
                outputs[:, input_ids.shape[1]:], skip_special_tokens=True
            )
            for (image_id, _), response in zip(batch, responses, strict=True):
                cleaned[image_id] = extract_clean_answer(response)
        return cleaned

    return generate


def collect_blocks(original, output_dir, block_size, batch_size, reuse_only):
    items = list(original.items())
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    manifest_path = output_dir / "cleaning_run.json"
    manifest = {
        "input_sha256": hashlib.sha256(json.dumps(items, ensure_ascii=False).encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model_id": MODEL_ID,
        "block_size": block_size,
        "batch_size": batch_size,
    }
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("Cleaning settings or input changed. Use a new output directory.")
    elif not reuse_only:
        if any(output_dir.glob("subset_*.json")):
            raise ValueError("Existing blocks have no run metadata. Use --reuse-blocks-only or a new directory.")
        write_json(manifest_path, manifest)

    combined = {}
    generate = None
    for number, start in enumerate(range(0, len(items), block_size)):
        block = items[start:start + block_size]
        path = output_dir / f"subset_{number:03d}.json"
        if path.is_file():
            cleaned = read_json(path)
            if (not isinstance(cleaned, dict) or cleaned.keys() != {key for key, _ in block}
                    or not all(isinstance(value, str) for value in cleaned.values())):
                raise ValueError(f"Unexpected IDs or caption values in {path}")
        elif reuse_only:
            raise FileNotFoundError(f"Missing cleaning block: {path}")
        else:
            if generate is None:
                generate = create_generator(prompt, batch_size)
            cleaned = generate(block, number)
            write_json(path, cleaned)
        combined.update((key, [cleaned[key]]) for key, _ in block)
    return combined


def prepare_paraphrasing_input(original, combined, output_dir):
    retained = {}
    rejected = {"empty": [], "na": [], "over_300_words": []}
    write_json(output_dir / "combined_output.json", combined)

    with (output_dir / "cleaning_word_counts.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "original_words", "cleaned_words", "word_count_change"])
        for image_id, values in combined.items():
            caption = values[0].strip()
            words = len(caption.split())
            if not caption:
                rejected["empty"].append(image_id)
            elif caption.upper() == "NA":
                rejected["na"].append(image_id)
            elif words > MAX_WORDS:
                rejected["over_300_words"].append(image_id)
            else:
                retained[image_id] = [caption]

            if caption.upper() != "NA":
                original_text = original[image_id]
                if isinstance(original_text, list):
                    original_text = " ".join(str(value) for value in original_text)
                original_words = len(str(original_text).split())
                writer.writerow([image_id, original_words, words, words - original_words])

    write_json(output_dir / "cleaning_rejections.json", rejected)
    write_json(output_dir / "filtered_na_records.json", {
        image_id: original[image_id] for image_id in rejected["na"]
    })
    path = output_dir / "paraphrasing_input.json"
    write_json(path, retained)
    print(f"Retained {len(retained):,} of {len(combined):,} captions: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--reuse-blocks-only", action="store_true",
        help="Build the paraphrasing input from existing subset_*.json files without loading the model.",
    )
    args = parser.parse_args()
    if min(args.block_size, args.batch_size) < 1:
        parser.error("Block size and batch size must be positive.")
    original = read_json(args.input_json)
    if not isinstance(original, dict) or not original:
        parser.error("The input JSON must be a nonempty mapping from image IDs to captions.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = collect_blocks(
        original, args.output_dir, args.block_size, args.batch_size, args.reuse_blocks_only
    )
    prepare_paraphrasing_input(original, combined, args.output_dir)


if __name__ == "__main__":
    main()
