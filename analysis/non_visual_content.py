"""Analyze non-visual content in ImageCLEFmedical references and Qwen predictions."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# 1. Input columns

IMAGE_COL = "image_id"
PREDICTION_COL = "qwen_caption_clean"
STATUS_COL = "status"

# 2. Non-visual content patterns

PATTERNS = {


    # ========================================================
    # A. Patient references
    # ========================================================

    "patient_reference": [

        (
            "patient",
            r"\bpatient\b"
        ),

        (
            "patients",
            r"\bpatients\b"
        ),

        (
            "same_patient",
            r"\bsame\s+patient\b"
        ),

        (
            "proband",
            r"\bproband\b"
        ),

        (
            "participant",
            r"\bparticipants?\b"
        ),

        # Match "case" only when it appears to identify a specific case,
        # so general clinical usage is excluded.

        (
            "case_number",
            r"\bcase\s*(?:no\.?|#)?\s*\d+\b"
        ),
    ],


    # ========================================================
    # B. Age
    # ========================================================

    "age": [

        # 45-year-old
        # 45 year old
        # 45-years-old

        (
            "numeric_year_old",
            r"\b\d{1,3}\s*[- ]\s*years?\s*[- ]\s*old\b"
        ),


        # 45year-old / 45years-old

        (
            "compact_numeric_year_old",
            r"\b\d{1,3}\s*years?[- ]old\b"
        ),


        # 45 years old

        (
            "years_old",
            r"\b\d{1,3}\s+years?\s+old\b"
        ),


        # at 64 years of age

        (
            "years_of_age",
            r"\b\d{1,3}\s+years?\s+of\s+age\b"
        ),


        # age 64
        # aged 64
        # age of 64

        (
            "age_numeric",
            r"\b(?:age|aged|age\s+of)\s*[:\-]?\s*\d{1,3}\b"
        ),


        # Twenty-one-year-old
        # eighteen-year-old

        (
            "written_year_old",
            r"\b[a-z]+(?:-[a-z]+)*-year-old\b"
        ),
    ],


    # ========================================================
    # C. Sex and gender
    # ========================================================

    "sex_gender": [

        (
            "male",
            r"\bmale\b"
        ),

        (
            "female",
            r"\bfemale\b"
        ),

        (
            "man",
            r"\bman\b"
        ),

        (
            "men",
            r"\bmen\b"
        ),

        (
            "woman",
            r"\bwoman\b"
        ),

        (
            "women",
            r"\bwomen\b"
        ),

        (
            "boy",
            r"\bboy\b"
        ),

        (
            "girl",
            r"\bgirl\b"
        ),
    ],


    # ========================================================
    # D. Temporal context
    # ========================================================

    "temporal_context": [

        (
            "preoperative",
            r"\bpre[- ]?operative(?:ly)?\b"
        ),
        
        (
        "intraoperative",
        r"\bintra[- ]?operative(?:ly)?\b"
        ),

        (
            "postoperative",
            r"\bpost[- ]?operative(?:ly)?\b"
        ),

        (
            "preop",
            r"\bpre[- ]?op\b"
        ),

        (
            "postop",
            r"\bpost[- ]?op\b"
        ),

        (
            "follow_up",
            r"\bfollow[- ]?up\b"
        ),

        (
            "on_admission",
            r"\bon\s+admission\b"
        ),

        (
            "at_admission",
            r"\bat\s+admission\b"
        ),

        (
            "upon_admission",
            r"\bupon\s+admission\b"
        ),

        (
            "on_presentation",
            r"\bon\s+presentation\b"
        ),

        (
            "first_visit",
            r"\bfirst\s+visit\b"
        ),

        (
            "initial_visit",
            r"\binitial\s+visit\b"
        ),

        # 3 days after
        # 6 weeks after
        # 10 months after
        # 2 years after

        (
            "time_after",
            r"\b\d+\s*[- ]?\s*"
            r"(?:days?|weeks?|months?|years?)\s+"
            r"(?:after|following)\b"
        ),

        # 3 days before surgery
        # 2 months before

        (
            "time_before",
            r"\b\d+\s*[- ]?\s*"
            r"(?:days?|weeks?|months?|years?)\s+before\b"
        ),

        (
            "after_surgery",
            r"\bafter\s+(?:the\s+)?surgery\b"
        ),

        (
            "after_operation",
            r"\bafter\s+(?:the\s+)?operation\b"
        ),

        (
            "following_surgery",
            r"\bfollowing\s+(?:the\s+)?surgery\b"
        ),

        (
            "following_operation",
            r"\bfollowing\s+(?:the\s+)?operation\b"
        ),

        (
            "before_surgery",
            r"\bbefore\s+(?:the\s+)?surgery\b"
        ),

        (
            "before_operation",
            r"\bbefore\s+(?:the\s+)?operation\b"
        ),

        (
            "post_treatment",
            r"\bpost[- ]?treatment\b"
        ),

        (
            "postprocedure",
            r"\bpost[- ]?procedure\b"
        ),
    ],


    # ========================================================
    # E. Figure and document references
    # ========================================================

    "figure_document": [

        (
            "figure",
            r"\bfigure\s*\d*\b"
        ),

        (
            "fig",
            r"\bfig\.?\s*\d+\b"
        ),

        (
            "table",
            r"\btable\s*\d+\b"
        ),

        (
            "supplementary",
            r"\bsupplementary\s+"
            r"(?:figure|fig\.?|image|table)\b"
        ),
    ],


    # ========================================================
    # F. Clinical and procedural context
    # ========================================================

    "clinical_context": [

        (
            "history_of",
            r"\bhistory\s+of\b"
        ),

        (
            "presented_with",
            r"\bpresented\s+with\b"
        ),

        (
            "presentation_with",
            r"\bpresentation\s+with\b"
        ),

        (
            "complained_of",
            r"\bcomplain(?:ed|ing)?\s+of\b"
        ),

        (
            "diagnosed_with",
            r"\bdiagnosed\s+with\b"
        ),

        (
            "underwent",
            r"\bunderwent\b"
        ),

        (
            "status_post",
            r"\bstatus\s+post\b"
        ),

        (
            "at_surgery",
            r"\bat\s+surgery\b"
        ),

        (
            "during_surgery",
            r"\bduring\s+surgery\b"
        ),

        (
            "at_operation",
            r"\bat\s+operation\b"
        ),

        (
            "during_operation",
            r"\bduring\s+operation\b"
        ),
    ],
}


# ============================================================
# 3. Compile regular expressions
# ============================================================

COMPILED_PATTERNS = {}


for category, pattern_list in PATTERNS.items():

    COMPILED_PATTERNS[
        category
    ] = [

        (
            label,
            re.compile(
                pattern,
                flags=re.IGNORECASE,
            )
        )

        for label, pattern
        in pattern_list
    ]


# ============================================================
# 4. Normalize image IDs
# ============================================================

def normalize_image_id(
    image_id,
):

    image_id = str(
        image_id
    ).strip()


    if image_id.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
        )
    ):

        return image_id


    return (
        image_id
        + ".jpg"
    )


# ============================================================
# 5. Load reference captions
# ============================================================

def load_json_captions(
    path,
    source_name,
):

    print(
        f"\nLoading {source_name}:"
    )

    print(
        path
    )


    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )


    rows = []


    for image_id, captions in data.items():

        if not captions:

            continue


        # ----------------------------------------------------
        # Expected format:
        #
        # "image.jpg": [
        #     "caption"
        # ]
        # ----------------------------------------------------

        if isinstance(
            captions,
            list,
        ):

            if len(captions) == 0:

                continue

            caption = captions[0]

        else:

            caption = captions


        caption = str(
            caption
        ).strip()


        if caption == "":

            continue


        rows.append(
            {
                "source":
                    source_name,

                "image_id":
                    normalize_image_id(
                        image_id
                    ),

                "caption":
                    caption,
            }
        )


    df = pd.DataFrame(
        rows
    )


    print(
        f"Captions loaded: "
        f"{len(df):,}"
    )


    return df


# ============================================================
# 6. Load Qwen predictions
# ============================================================

def load_qwen_predictions(
    path,
    source_name,
):

    print(
        f"\nLoading {source_name}:"
    )

    print(
        path
    )


    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, dtype={IMAGE_COL: str}, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype={IMAGE_COL: str}, keep_default_na=False)


    required = [
        IMAGE_COL,
        PREDICTION_COL,
    ]


    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"Missing column "
                f"'{col}' in:\n{path}"
            )


    # --------------------------------------------------------
    # Keep successful inferences
    # --------------------------------------------------------

    if STATUS_COL in df.columns:

        df = (
            df[
                df[
                    STATUS_COL
                ] == "ok"
            ]
            .copy()
        )


    # --------------------------------------------------------
    # Build the prediction table
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "source": [
                source_name
            ] * len(df),

            "image_id": (
                df[
                    IMAGE_COL
                ]
                .apply(
                    normalize_image_id
                )
                .tolist()
            ),

            "caption": (
                df[
                    PREDICTION_COL
                ]
                .astype(str)
                .str.strip()
                .tolist()
            ),
        }
    )


    # --------------------------------------------------------
    # Drop empty captions
    # --------------------------------------------------------

    result = (
        result[
            result[
                "caption"
            ] != ""
        ]
        .reset_index(
            drop=True
        )
    )


    print(
        f"Predictions loaded: "
        f"{len(result):,}"
    )


    print(
        f"Source: "
        f"{source_name}"
    )


    return result


# ============================================================
# 7. Detect non-visual content
# ============================================================

def analyze_caption(
    caption,
):

    caption = str(
        caption
    )


    result = {}


    all_categories = []

    all_labels = []

    all_matches = []


    for category, patterns in COMPILED_PATTERNS.items():

        category_found = False


        for label, regex in patterns:

            matches = list(
                regex.finditer(
                    caption
                )
            )


            if matches:

                category_found = True

                all_labels.append(
                    label
                )


                for match in matches:

                    matched_text = (
                        match.group(0)
                    )


                    all_matches.append(
                        f"{category}:"
                        f"{label}:"
                        f"{matched_text}"
                    )


        result[
            category
        ] = category_found


        if category_found:

            all_categories.append(
                category
            )


    # --------------------------------------------------------
    # Overall indicator
    # --------------------------------------------------------

    result[
        "has_non_visual"
    ] = (
        len(all_categories) > 0
    )


    result[
        "n_non_visual_categories"
    ] = len(
        all_categories
    )


    result[
        "matched_categories"
    ] = (
        " | ".join(
            sorted(
                set(
                    all_categories
                )
            )
        )
    )


    result[
        "matched_patterns"
    ] = (
        " | ".join(
            sorted(
                set(
                    all_labels
                )
            )
        )
    )


    result[
        "matched_text"
    ] = (
        " | ".join(
            all_matches
        )
    )


    return result


# ============================================================
# 8. Analyze captions
# ============================================================

def analyze_dataframe(
    df,
):

    print(
        f"\nAnalizando: "
        f"{df['source'].iloc[0]}"
    )


    analyses = (

        df[
            "caption"
        ]

        .apply(
            analyze_caption
        )

        .apply(
            pd.Series
        )
    )


    result = pd.concat(
        [
            df.reset_index(
                drop=True
            ),

            analyses.reset_index(
                drop=True
            ),
        ],

        axis=1,
    )


    return result


# 9. Summary helpers

CATEGORIES = list(
    PATTERNS.keys()
)


def create_summary(
    df,
):

    summary_rows = []


    for source, group in df.groupby(
        "source"
    ):

        n = len(
            group
        )


        row = {

            "source":
                source,

            "n_samples":
                n,

            "n_any_nonvisual":
                int(
                    group[
                        "has_non_visual"
                    ].sum()
                ),

            "pct_any_nonvisual":
                (
                    group[
                        "has_non_visual"
                    ].mean()
                    * 100
                ),
        }


        for category in CATEGORIES:

            n_category = int(
                group[
                    category
                ].sum()
            )


            pct_category = (
                group[
                    category
                ].mean()
                * 100
            )


            row[
                f"n_{category}"
            ] = n_category


            row[
                f"pct_{category}"
            ] = pct_category


        summary_rows.append(
            row
        )


    return pd.DataFrame(
        summary_rows
    )


def get_pct(
    summary_df,
    source,
):

    row = (
        summary_df[
            summary_df[
                "source"
            ] == source
        ]
    )

    if len(row) == 0:
        return None

    return float(
        row.iloc[0][
            "pct_any_nonvisual"
        ]
    )


def relative_reduction(
    before,
    after,
):

    if (
        before is None
        or after is None
        or before == 0
    ):
        return None

    return (
        (before - after)
        / before
        * 100
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-references", type=Path)
    parser.add_argument("--cleaned-references", type=Path)
    parser.add_argument("--original-predictions", type=Path)
    parser.add_argument("--clean-paraphrase-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    # 10. Load and analyze the requested sources
    inputs = (
        ("GT_original", args.original_references, load_json_captions),
        ("GT_clean", args.cleaned_references, load_json_captions),
        ("Qwen_original", args.original_predictions, load_qwen_predictions),
        ("Qwen_C&A", args.clean_paraphrase_predictions, load_qwen_predictions),
    )
    selected = [(source, path, loader) for source, path, loader in inputs if path is not None]
    if not selected:
        parser.error("Provide at least one reference or prediction file.")

    analyzed = {}
    for source, path, loader in selected:
        if not path.is_file():
            parser.error(f"Input file was not found: {path}")
        captions = loader(path, source)
        if captions.empty:
            parser.error(f"No captions to analyze in {path}")
        analyzed[source] = analyze_dataframe(captions)

    output_dir = args.output_dir or Path("runs/non_visual_content")
    if args.output_dir is None and len(selected) == 1:
        output_dir /= selected[0][0]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 11. Export per-sample results, positive cases, and summary
    all_results = pd.concat(analyzed.values(), ignore_index=True)
    all_results.to_csv(output_dir / "nonvisual_per_sample_all.csv", index=False)
    all_results.loc[all_results["has_non_visual"]].to_csv(
        output_dir / "nonvisual_positive_cases_review.csv", index=False
    )
    summary_all = create_summary(all_results)
    summary_all.to_csv(output_dir / "summary_nonvisual_full_datasets.csv", index=False)
    print(summary_all.to_string(index=False))

    # 12. Export comparisons when the required sources are available
    comparisons = []
    if {"GT_original", "GT_clean"} <= analyzed.keys():
        gt_original = analyzed["GT_original"]
        gt_clean = analyzed["GT_clean"]
        common_ids = set(gt_original["image_id"]) & set(gt_clean["image_id"])
        gt_results = pd.concat(
            [frame.loc[frame["image_id"].isin(common_ids)] for frame in (gt_original, gt_clean)],
            ignore_index=True,
        )
        gt_results.to_csv(output_dir / "nonvisual_GT_original_vs_clean_common_ids.csv", index=False)
        gt_summary = create_summary(gt_results)
        gt_summary.to_csv(output_dir / "summary_GT_original_vs_clean.csv", index=False)
        before = get_pct(gt_summary, "GT_original") if common_ids else None
        after = get_pct(gt_summary, "GT_clean") if common_ids else None
        comparisons.append(("GT original -> GT clean (common GT IDs)", before, after))

    if {"Qwen_original", "Qwen_C&A"} <= analyzed.keys():
        generation_summary = summary_all.loc[summary_all["source"].ne("GT_clean")]
        generation_summary.to_csv(output_dir / "summary_nonvisual_main_comparison.csv", index=False)
        before = get_pct(generation_summary, "Qwen_original")
        after = get_pct(generation_summary, "Qwen_C&A")
        comparisons.append(("Qwen original -> Qwen C&A (full predictions)", before, after))

    if comparisons:
        reduction_summary = pd.DataFrame(
            {
                "comparison": label,
                "before_pct": before,
                "after_pct": after,
                "relative_reduction_pct": relative_reduction(before, after),
            }
            for label, before, after in comparisons
        )
        reduction_summary.to_csv(output_dir / "nonvisual_reduction_comparison_final.csv", index=False)


if __name__ == "__main__":
    main()
