"""Evaluate modality agreement between reference captions and Qwen predictions."""

import argparse
import re
from pathlib import Path

import pandas as pd

from analysis.non_visual_content import load_json_captions, normalize_image_id


# 1. Canonical modality patterns

MODALITY_PATTERNS = {


    # ========================================================
    # CT
    # ========================================================

    "CT": [

        r"\bCT\b",

        r"\bCT[- ]?scan\b",

        r"\bcomputed\s+tomograph(?:y|ic)\b",

        r"\bcomputed[- ]tomography\b",

        r"\bCTA\b",

        r"\bCT\s+angiograph(?:y|ic)\b",

        r"\bcomputed\s+tomographic\s+angiograph(?:y|ic)\b",

        r"\bHRCT\b",

        r"\bCBCT\b",

        r"\bcone[- ]beam\s+computed\s+tomography\b",

        r"\bCTPA\b",

        r"\bCT[- ]?angiogram\b",
    ],


    # ========================================================
    # MRI
    # ========================================================

    "MRI": [

        r"\bMRI\b",

        r"\bMR\s+imaging\b",

        r"\bMR\s+image\b",

        r"\bMR\s+images\b",

        r"\bmagnetic\s+resonance\s+imag(?:e|es|ing)\b",

        r"\bmagnetic\s+resonance\b",

        r"\bMRA\b",

        r"\bMR\s+angiograph(?:y|ic)\b",

        r"\bmagnetic\s+resonance\s+angiograph(?:y|ic)\b",

        r"\bMRCP\b",

        r"\bfMRI\b",
    ],


    # ========================================================
    # X-RAY / RADIOGRAPHY
    # ========================================================

    "X-ray": [

        r"\bx[- ]?ray\b",

        r"\bradiograph\b",

        r"\bradiographs\b",

        r"\bradiography\b",

        r"\bCXR\b",

        r"\bplain\s+film\b",

        r"\bplain\s+radiograph\b",

        r"\bpanoramic\s+radiograph\b",

        r"\bpanoramic\s+radiography\b",

        r"\borthopantomograph(?:y|ic)\b",

        r"\bOPG\b",

        r"\bDEXA\b",

        r"\bdual[- ]energy\s+x[- ]ray\s+absorptiometry\b",
    ],


    # ========================================================
    # ULTRASOUND
    # ========================================================

    "Ultrasound": [

        r"\bultrasound\b",

        r"\bultrasonography\b",

        r"\bsonography\b",

        r"\bsonographic\b",

        r"\bDoppler\s+ultrasound\b",

        r"\bduplex\s+ultrasound\b",

        r"\bechocardiogram\b",

        r"\bechocardiography\b",

        r"\bechocardiographic\b",

        r"\bEUS\b",

        r"\bendoscopic\s+ultrasound\b",
    ],


    # ========================================================
    # PET
    # ========================================================

    "PET": [

        r"\bPET\b",

        r"\bPET[- /]?CT\b",

        r"\bFDG[- ]?PET\b",

        r"\bFDG[- ]?PET[- /]?CT\b",

        r"\bpositron\s+emission\s+tomography\b",
    ],


    # ========================================================
    # SPECT
    # ========================================================

    "SPECT": [

        r"\bSPECT\b",

        r"\bSPECT[- /]?CT\b",

        r"\bsingle[- ]photon\s+emission\s+computed\s+tomography\b",
    ],


    # ========================================================
    # MAMMOGRAPHY
    # ========================================================

    "Mammography": [

        r"\bmammogram\b",

        r"\bmammography\b",

        r"\bmammographic\b",
    ],


    # ========================================================
    # FLUOROSCOPY
    # ========================================================

    "Fluoroscopy": [

        r"\bfluoroscopy\b",

        r"\bfluoroscopic\b",

        r"\bfluorogram\b",
    ],


    # ========================================================
    # ANGIOGRAPHY / DSA
    #
    # CTA and MRA are also detected by the CT and MRI patterns.
    # This group covers generic angiography and DSA.
    # ========================================================

    "Angiography": [

        r"\bDSA\b",

        r"\bdigital\s+subtraction\s+angiograph(?:y|ic)\b",

        r"\bangiogram\b",

        r"\bangiographic\b",

        r"\bangiography\b",

        r"\barteriogram\b",

        r"\barteriography\b",

        r"\bvenogram\b",

        r"\bvenography\b",

        r"\bcholangiogram\b",

        r"\bcholangiography\b",
    ],


    # ========================================================
    # SCINTIGRAPHY
    # ========================================================

    "Scintigraphy": [

        r"\bscintigraphy\b",

        r"\bscintigraphic\b",

        r"\bscintigram\b",
    ],


    # ========================================================
    # OCT
    # ========================================================

    "OCT": [

        r"\bOCT\b",

        r"\boptical\s+coherence\s+tomography\b",

        r"\bAS[- ]?OCT\b",
    ],
}

COMPILED_MODALITIES = {
    modality: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for modality, patterns in MODALITY_PATTERNS.items()
}


# 2. Detection and classification

def extract_modalities(text):
    modalities, matched_terms = [], []
    for modality, patterns in COMPILED_MODALITIES.items():
        terms = [
            match.group(0)
            for pattern in patterns
            for match in pattern.finditer(str(text))
        ]
        if terms:
            modalities.append(modality)
            matched_terms.extend(f"{modality}:{term}" for term in terms)
    return sorted(modalities), matched_terms


def add_modality_columns(frame, prefix):
    detections = frame[f"{prefix}_caption"].map(extract_modalities)
    frame[f"{prefix}_modalities"] = detections.map(lambda result: " | ".join(result[0]))
    frame[f"{prefix}_matched_terms"] = detections.map(lambda result: " | ".join(result[1]))
    frame[f"{prefix}_has_modality"] = detections.map(lambda result: bool(result[0]))
    return detections.map(lambda result: set(result[0]))


def classify_prediction(reference, prediction):
    if not reference:
        return "GT_NOT_EVALUABLE"
    if not prediction:
        return "OMITTED"
    return "MATCH" if reference & prediction else "MISMATCH"


def percentage(count, total):
    return 100 * count / total if total else None


# 3. Input and pairing

def load_predictions(path, prefix):
    read = pd.read_excel if path.suffix.lower() == ".xlsx" else pd.read_csv
    frame = read(path, dtype={"image_id": str}, keep_default_na=False)
    missing = {"image_id", "qwen_caption_clean"} - set(frame)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    if "status" in frame:
        frame = frame.loc[frame["status"].eq("ok")].copy()
    frame["image_id"] = frame["image_id"].map(normalize_image_id)
    return frame[["image_id", "qwen_caption_clean"]].rename(
        columns={"qwen_caption_clean": f"{prefix}_caption"}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-references", type=Path, required=True)
    parser.add_argument("--original-predictions", type=Path)
    parser.add_argument("--clean-paraphrase-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    inputs = (
        ("Qwen_original", "original", args.original_predictions),
        ("Qwen_C&A", "canda", args.clean_paraphrase_predictions),
    )
    selected = [(model, prefix, path) for model, prefix, path in inputs if path is not None]
    if not selected:
        parser.error("Provide at least one predictions file.")
    for path in (args.original_references, *(path for _, _, path in selected)):
        if not path.is_file():
            parser.error(f"Input file was not found: {path}")

    references = load_json_captions(args.original_references, "GT_original")
    if references.empty:
        parser.error("No reference captions to analyze.")
    paired = references.drop(columns="source").rename(columns={"caption": "GT_caption"})
    if paired["image_id"].duplicated().any():
        parser.error("Reference image IDs must be unique after normalization.")
    for _, prefix, path in selected:
        predictions = load_predictions(path, prefix)
        if predictions["image_id"].duplicated().any():
            parser.error(f"Prediction image IDs must be unique after normalization in {path}")
        paired = paired.merge(predictions, on="image_id", validate="one_to_one")
    if paired.empty:
        parser.error("No common image IDs between the selected inputs.")

    reference_sets = add_modality_columns(paired, "GT")
    strict_mask = paired["GT_has_modality"].copy()
    mismatch_mask = pd.Series(False, index=paired.index)
    for _, prefix, _ in selected:
        prediction_sets = add_modality_columns(paired, prefix)
        paired[f"{prefix}_status"] = [
            classify_prediction(ref, pred) for ref, pred in zip(reference_sets, prediction_sets)
        ]
        paired[f"{prefix}_exact_match"] = [
            bool(ref) and ref == pred for ref, pred in zip(reference_sets, prediction_sets)
        ]
        strict_mask &= paired[f"{prefix}_has_modality"]
        mismatch_mask |= paired[f"{prefix}_status"].eq("MISMATCH")

    # 4. Strict and reference-evaluable summaries
    strict = paired.loc[strict_mask]
    evaluable = paired.loc[paired["GT_has_modality"]]
    strict_rows, full_rows = [], []
    for model, prefix, _ in selected:
        status = strict[f"{prefix}_status"]
        matched = int(status.eq("MATCH").sum())
        mismatched = int(status.eq("MISMATCH").sum())
        exact = int(strict[f"{prefix}_exact_match"].sum())
        strict_rows.append({
            "model": model,
            "n_samples": len(strict),
            "n_modality_consistent": matched,
            "pct_modality_consistent": percentage(matched, len(strict)),
            "n_modality_inconsistent": mismatched,
            "pct_modality_inconsistent": percentage(mismatched, len(strict)),
            "n_exact_modality_match": exact,
            "pct_exact_modality_match": percentage(exact, len(strict)),
        })
        counts = evaluable[f"{prefix}_status"].value_counts()
        row = {"model": model, "n_GT_evaluable": len(evaluable)}
        for label in ("match", "mismatch", "omitted"):
            count = int(counts.get(label.upper(), 0))
            row[f"n_{label}"] = count
            row[f"pct_{label}"] = percentage(count, len(evaluable))
        full_rows.append(row)

    frequency_rows = []
    for modality in MODALITY_PATTERNS:
        count = sum(modality in modalities for modalities in reference_sets)
        frequency_rows.append({
            "modality": modality, "n_GT": count, "pct_GT": percentage(count, len(paired)),
        })
    frequency = pd.DataFrame(frequency_rows).sort_values("n_GT", ascending=False)

    # 5. Export reports
    output_dir = args.output_dir or Path("runs/modality_consistency")
    if args.output_dir is None and len(selected) == 1:
        output_dir /= selected[0][0]
    output_dir.mkdir(parents=True, exist_ok=True)
    strict_summary = pd.DataFrame(strict_rows)
    reports = {
        "Strict_summary": ("modality_strict_summary.csv", strict_summary),
        "Match_Mismatch_Omit": ("modality_full_summary.csv", pd.DataFrame(full_rows)),
        "GT_modality_frequency": ("modality_frequency.csv", frequency),
        "Strict_cases": ("modality_strict_subset.csv", strict),
        "GT_evaluable": ("modality_GT_evaluable.csv", evaluable),
        "Mismatch_review": ("modality_mismatch_cases_review.csv", paired.loc[mismatch_mask]),
        "Per_sample": ("modality_per_sample.csv", paired),
    }
    if len(selected) == 2:
        before, after = (row["pct_modality_consistent"] for row in strict_rows)
        change = None if before is None else after - before
        reports["Main_result"] = ("modality_comparison.csv", pd.DataFrame([{
            "comparison": "Qwen Original -> Qwen C&A",
            "original_consistency_pct": before,
            "canda_consistency_pct": after,
            "absolute_change_percentage_points": change,
            "relative_change_pct": percentage(change, before) if change is not None else None,
        }]))

    methodology = pd.DataFrame([
        ("Strict subset", "Explicit modality in the reference and every selected prediction."),
        ("Consistency", "At least one canonical modality is shared with the reference."),
        ("Exact match", "The reference and prediction have the same nonempty modality set."),
        ("Reference-evaluable", "Explicit modality in the reference, including prediction omissions."),
    ], columns=["Parameter", "Definition"])

    with pd.ExcelWriter(output_dir / "modality_analysis_complete.xlsx", engine="openpyxl") as writer:
        methodology.to_excel(writer, sheet_name="Methodology", index=False)
        for sheet, (filename, frame) in reports.items():
            frame.to_csv(output_dir / filename, index=False)
            frame.to_excel(writer, sheet_name=sheet, index=False)
            for row in writer.sheets[sheet].iter_rows(min_row=2):
                for cell in row:
                    if cell.data_type == "f":
                        cell.data_type = "s"

    print(strict_summary.to_string(index=False))
    print(f"Reports saved to {output_dir}")


if __name__ == "__main__":
    main()
