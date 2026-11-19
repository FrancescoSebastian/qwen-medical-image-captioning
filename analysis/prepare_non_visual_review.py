"""Prepare the manual review workbook for the non-visual content analysis."""

import argparse
from pathlib import Path

# 1. Sampling configuration

CATEGORIES = (
    "patient_reference",
    "age",
    "sex_gender",
    "temporal_context",
    "figure_document",
    "clinical_context",
)
SEED = 42


# 2. Review samples

def select_review_cases(cases):
    import pandas as pd

    clean_paraphrase = cases.loc[cases["source"].eq("Qwen_C&A")]
    original = cases.loc[cases["source"].eq("Qwen_original")]
    clean_references = cases.loc[cases["source"].eq("GT_clean")]

    # Draw up to 20 original predictions from each category, then deduplicate.
    original_samples = []
    for category in CATEGORIES:
        subset = original.loc[original["matched_categories"].str.contains(category, regex=False)]
        if not subset.empty:
            original_samples.append(subset.sample(n=min(20, len(subset)), random_state=SEED))
    original_review = original.iloc[:0]
    if original_samples:
        original_review = pd.concat(original_samples, ignore_index=True).drop_duplicates(
            subset="image_id"
        )

    return {
        "Qwen_C&A_ALL": clean_paraphrase,
        "Qwen_original_sample": original_review,
        "GT_clean_sample": clean_references.sample(n=min(50, len(clean_references)), random_state=SEED),
    }


# 3. Workbook export

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Detected positive cases.")
    parser.add_argument("--output-xlsx", type=Path, required=True, help="Manual review workbook.")
    args = parser.parse_args()

    if not args.input_csv.is_file():
        parser.error(f"Positive cases CSV was not found: {args.input_csv}")
    if args.output_xlsx.suffix.lower() != ".xlsx":
        parser.error("--output-xlsx must end in .xlsx.")
    if args.output_xlsx.exists():
        parser.error(f"{args.output_xlsx} already exists. Choose a new filename.")

    import pandas as pd

    cases = pd.read_csv(args.input_csv, dtype={"image_id": str}, keep_default_na=False)
    missing = {"source", "image_id", "matched_categories"} - set(cases.columns)
    if missing:
        parser.error(f"Missing CSV columns: {', '.join(sorted(missing))}")

    print(f"Positive cases: {len(cases):,}")
    print(cases["source"].value_counts().to_string())
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with args.output_xlsx.open("xb") as output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet_name, sample in select_review_cases(cases).items():
                review = sample.assign(manual_classification="", manual_comment="")
                review.to_excel(writer, sheet_name=sheet_name, index=False)
                # Keep any caption beginning with "=" as text in Excel.
                for row in writer.sheets[sheet_name].iter_rows(min_row=2):
                    for cell in row:
                        if cell.data_type == "f":
                            cell.data_type = "s"
                print(f"{sheet_name}: {len(review):,} cases for review")
    print(f"Workbook saved to {args.output_xlsx}")


if __name__ == "__main__":
    main()
