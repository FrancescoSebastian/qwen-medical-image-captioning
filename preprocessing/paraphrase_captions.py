"""Paraphrase cleaned ImageCLEFmedical captions and extract valid answers."""

import argparse
import re
from pathlib import Path

from preprocessing.clean_captions import MAX_WORDS, collect_blocks, read_json, write_json


PROMPT_PATH = Path(__file__).with_name("paraphrasing_prompt.txt")
MIN_WORDS = 3
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_text_after_think(response):
    parts = response.split("</think>")
    return parts[1].strip().strip('"') if len(parts) > 1 else response.strip()


def select_paraphrasing_inputs(cleaned):
    selected = {}
    for image_id, captions in cleaned.items():
        if not isinstance(captions, list) or len(captions) != 1 or not isinstance(captions[0], str):
            raise ValueError(f"Expected a one-element caption list for {image_id!r}.")
        if len(captions[0].split()) >= MIN_WORDS:
            selected[image_id] = captions
    return selected


def prepare_paraphrases(combined):
    paraphrases = {}
    rejected = {"missing_answer_tags": [], "empty_answer": [], "over_300_words": []}
    for image_id, responses in combined.items():
        match = ANSWER_PATTERN.search(responses[0])
        if match is None:
            rejected["missing_answer_tags"].append(image_id)
            continue
        caption = match.group(1).strip()
        if not caption:
            rejected["empty_answer"].append(image_id)
        elif len(caption.split()) > MAX_WORDS:
            rejected["over_300_words"].append(image_id)
        else:
            paraphrases[image_id] = [caption]
    return paraphrases, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reuse-blocks-only", action="store_true")
    args = parser.parse_args()
    if min(args.block_size, args.batch_size) < 1:
        parser.error("Block size and batch size must be positive.")
    cleaned = read_json(args.input_json)
    if not isinstance(cleaned, dict) or not cleaned:
        parser.error("Expected a nonempty mapping from image IDs to cleaned caption lists.")
    selected = select_paraphrasing_inputs(cleaned)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = collect_blocks(
        selected, args.output_dir, args.block_size, args.batch_size, args.reuse_blocks_only,
        prompt_path=PROMPT_PATH, block_prefix="p_subset", manifest_name="paraphrasing_run.json",
        extract_response=extract_text_after_think,
    )
    paraphrases, rejected = prepare_paraphrases(combined)
    write_json(args.output_dir / "p_combined_output.json", combined)
    write_json(args.output_dir / "paraphrasing_rejections.json", rejected)
    output_path = args.output_dir / "paraphrased_captions.json"
    write_json(output_path, paraphrases)
    print(f"Selected {len(selected):,} of {len(cleaned):,} cleaned captions.")
    print(f"Retained {len(paraphrases):,} paraphrases: {output_path}")


if __name__ == "__main__":
    main()
