"""Measure conversational leakage in raw Qwen test predictions."""

import argparse
from pathlib import Path

import pandas as pd

from qwen.evaluate_qwen import MARKERS, PREDICTION_COL


def analyze_file(path, model):
    predictions = pd.read_csv(path)
    if PREDICTION_COL not in predictions:
        raise ValueError(f"{path} is missing the {PREDICTION_COL!r} column.")
    if "status" in predictions:
        predictions = predictions.loc[predictions["status"].eq("ok")]

    captions = predictions[PREDICTION_COL].dropna()
    if captions.empty:
        raise ValueError(f"No eligible raw captions in {path}")

    n_leakage = sum(any(marker in str(caption) for marker in MARKERS) for caption in captions)
    return {
        "model": model,
        "n_total": len(captions),
        "n_leakage": n_leakage,
        "pct_leakage": 100 * n_leakage / len(captions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-predictions", type=Path, help="Raw Qwen-O inference CSV.")
    parser.add_argument("--clean-paraphrase-predictions", type=Path, help="Raw Qwen-C&P inference CSV.")
    parser.add_argument("--output-csv", type=Path, help="Optional path for the summary CSV.")
    args = parser.parse_args()

    inputs = (("Qwen-O", args.original_predictions), ("Qwen-C&P", args.clean_paraphrase_predictions))
    selected = [(model, path) for model, path in inputs if path is not None]
    if not selected:
        parser.error("Provide at least one raw predictions CSV.")

    try:
        summary = pd.DataFrame([analyze_file(path, model) for model, path in selected])
        if args.output_csv is not None:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.output_csv.open("x", encoding="utf-8", newline="") as output:
                summary.to_csv(output, index=False)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(summary.to_string(index=False, formatters={"pct_leakage": "{:.2f}".format}))


if __name__ == "__main__":
    main()
