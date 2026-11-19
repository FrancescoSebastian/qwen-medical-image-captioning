"""Combine cleaned and paraphrased captions into the Qwen C&P training CSV."""

import argparse
import csv
from pathlib import Path

from preprocessing.clean_captions import MAX_WORDS, read_json
from preprocessing.paraphrase_captions import MIN_WORDS


def load_captions(path):
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping from image IDs to caption lists in {path}.")
    captions = {}
    for image_id, values in data.items():
        if not image_id.strip() or not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"Expected one caption for image ID {image_id!r} in {path}.")
        caption = values[0].strip() if isinstance(values[0], str) else ""
        if not caption or len(caption.split()) > MAX_WORDS:
            raise ValueError(f"Invalid caption for {image_id!r} in {path}. Use the filtered stage output.")
        captions[image_id] = caption
    return captions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-json", type=Path, required=True)
    parser.add_argument("--paraphrased-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    cleaned = load_captions(args.cleaned_json)
    paraphrased = load_captions(args.paraphrased_json)
    if not cleaned or any(caption.upper() == "NA" for caption in cleaned.values()):
        parser.error("--cleaned-json must contain valid cleaned captions.")
    for image_id in paraphrased:
        if image_id not in cleaned or len(cleaned[image_id].split()) < MIN_WORDS:
            parser.error(f"Paraphrase has no eligible cleaned caption: {image_id!r}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Caption"])
        # Both caption variants are training rows, even when their wording is identical.
        writer.writerows(cleaned.items())
        writer.writerows(paraphrased.items())

    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Paraphrased rows: {len(paraphrased):,}")
    print(f"Total training rows: {len(cleaned) + len(paraphrased):,}")
    print(f"Training CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
