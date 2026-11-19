"""Clean Qwen test predictions and export caption metrics to Excel."""

import argparse
from pathlib import Path


# 1. Configuration

VARIANTS = ("original", "clean-paraphrase")
ID_COL = "image_id"
REFERENCE_COL = "reference_caption"
PREDICTION_COL = "qwen_caption"
RAW_COL = "qwen_caption_raw"
CLEAN_COL = "qwen_caption_clean"
MARKERS = ("\nassistant", "\nuser", "<think>")
METRICS = ("BERTScore_F1", "ROUGE_L", "BLEU_1", "METEOR")


# 2. Prediction cleaning

def clean_generation(text):
    """Keep the caption before the first chat continuation marker."""
    end = min((text.index(marker) for marker in MARKERS if marker in text), default=len(text))
    return text[:end].strip()


def load_predictions(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {ID_COL, REFERENCE_COL, PREDICTION_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")

    total = len(df)
    if "status" in df.columns:
        df = df.loc[df["status"].eq("ok")]
    present = df[[ID_COL, REFERENCE_COL, PREDICTION_COL]].ne("").all(axis=1)
    df = df.loc[present].copy()
    df[RAW_COL] = df[PREDICTION_COL]
    df[CLEAN_COL] = df[RAW_COL].map(clean_generation)
    df["was_modified"] = df[RAW_COL].ne(df[CLEAN_COL])
    df["raw_words"] = df[RAW_COL].str.split().str.len()
    df["clean_words"] = df[CLEAN_COL].str.split().str.len()

    empty_after_cleaning = int(df[CLEAN_COL].eq("").sum())
    scored = df.loc[df[CLEAN_COL].ne("")].reset_index(drop=True)
    if scored.empty:
        raise ValueError(f"No eligible captions to score in {csv_path}")
    return scored, total, empty_after_cleaning


# 3. Caption metrics

def calculate_classical_metrics(df):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    smoothing = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bleu_scores, rouge_scores, meteor_scores = [], [], []
    for reference, prediction in zip(df[REFERENCE_COL], df[CLEAN_COL]):
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        bleu_scores.append(sentence_bleu(
            [reference_tokens], prediction_tokens,
            weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoothing,
        ))
        rouge_scores.append(rouge.score(reference, prediction)["rougeL"].fmeasure)
        meteor_scores.append(meteor_score([reference_tokens], prediction_tokens))

    df["ROUGE_L_clean"] = rouge_scores
    df["BLEU_1_clean"] = bleu_scores
    df["METEOR_clean"] = meteor_scores


def calculate_bertscore(df):
    import torch
    from bert_score import score
    from bert_score import utils

    original_sent_encode = utils.sent_encode

    def encode_with_length_limit(tokenizer, sentence):
        # Some Transformers versions expose a sentinel instead of a usable limit.
        original_limit = tokenizer.model_max_length
        try:
            tokenizer.model_max_length = 512
            return original_sent_encode(tokenizer, sentence)
        finally:
            tokenizer.model_max_length = original_limit

    utils.sent_encode = encode_with_length_limit
    try:
        _, _, f1 = score(
            df[CLEAN_COL].tolist(),
            df[REFERENCE_COL].tolist(),
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=16,
            verbose=True,
        )
    finally:
        utils.sent_encode = original_sent_encode

    df["BERTScore_F1_clean"] = f1.cpu().numpy()


# 4. Excel export

def save_excel(df, output_path):
    import pandas as pd

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        worksheet = writer.sheets["Results"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        float_columns = {index for index, column in enumerate(df.columns, start=1)
                         if pd.api.types.is_float_dtype(df[column])}
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                if cell.data_type == "f":
                    cell.data_type = "s"
                if cell.column in float_columns:
                    cell.number_format = "0.00000000"


# 5. Evaluation entry point

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-csv", type=Path)
    parser.add_argument("--clean-paraphrase-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/qwen_evaluation"))
    args = parser.parse_args()

    inputs = {variant: path for variant, path in
              zip(VARIANTS, (args.original_csv, args.clean_paraphrase_csv)) if path is not None}
    if not inputs:
        parser.error("Provide --original-csv, --clean-paraphrase-csv, or both.")
    for path in inputs.values():
        if not path.is_file():
            parser.error(f"Prediction CSV was not found: {path}")

    summary_path = args.output_dir / "qwen_test_metrics_summary.xlsx"
    details = {variant: args.output_dir / f"{variant}_per_sample_metrics.xlsx" for variant in inputs}
    for path in (summary_path, *details.values()):
        if path.exists():
            parser.error(f"{path} already exists. Choose another --output-dir.")

    import pandas as pd

    runs = {variant: load_predictions(path) for variant, path in inputs.items()}
    if len(runs) == 2:
        original = runs["original"][0][[ID_COL, REFERENCE_COL]]
        preprocessed = runs["clean-paraphrase"][0][[ID_COL, REFERENCE_COL]]
        if not original.value_counts().sort_index().equals(
            preprocessed.value_counts().sort_index()
        ):
            print("Warning: the variants have different scored image IDs or references.")

    summary = []
    for variant, (df, total, empty_after_cleaning) in runs.items():
        print(f"Scoring {variant}: {len(df):,} of {total:,} rows")
        calculate_bertscore(df)
        calculate_classical_metrics(df)
        summary.append({
            "variant": variant,
            "input_rows": total,
            "n_samples": len(df),
            "excluded_rows": total - len(df),
            "empty_after_cleaning": empty_after_cleaning,
            "modified_captions": int(df["was_modified"].sum()),
            "modified_percent": 100 * df["was_modified"].mean(),
            "mean_raw_words": df["raw_words"].mean(),
            "mean_clean_words": df["clean_words"].mean(),
            **{metric: df[f"{metric}_clean"].mean() for metric in METRICS},
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant, (df, _, _) in runs.items():
        save_excel(df, details[variant])
    summary_df = pd.DataFrame(summary)
    save_excel(summary_df, summary_path)
    print(summary_df.to_string(index=False))
    print(f"Excel reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
