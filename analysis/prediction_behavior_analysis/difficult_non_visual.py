"""Analyze non-visual references for low-baseline BERTScore samples."""

import argparse
from pathlib import Path

import pandas as pd

from analysis.non_visual_content import (
    CATEGORIES,
    PATTERNS,
    analyze_caption,
    load_json_captions,
    normalize_image_id,
)

IMAGE_COL = "image_id"
OUTCOME_COL = "outcome"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions-csv", type=Path, required=True)
    parser.add_argument("--original-references", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/difficult_non_visual"))
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load difficult samples

    print(
        "\nLoading difficult samples:"
    )

    print(
        args.transitions_csv
    )


    difficult = pd.read_csv(
        args.transitions_csv
    )


    if IMAGE_COL not in difficult.columns:

        raise ValueError(
            f"Missing column '{IMAGE_COL}'"
        )


    if OUTCOME_COL not in difficult.columns:

        raise ValueError(
            f"Missing column '{OUTCOME_COL}'"
        )


    difficult[
        IMAGE_COL
    ] = (
        difficult[
            IMAGE_COL
        ]
        .apply(
            normalize_image_id
        )
    )


    print(
        f"Difficult samples loaded: "
        f"{len(difficult):,}"
    )


    print(
        "\nOriginal outcome distribution:"
    )

    print(
        difficult[
            OUTCOME_COL
        ]
        .value_counts()
    )


    # ============================================================
    # 2. Map sample outcomes
    # ============================================================

    OUTCOME_MAP = {

        "Improved and crossed >0.50":
            "Recovered",

        "Improved but remained <=0.50":
            "Improved_but_low",

        "Not improved":
            "Not_improved"
    }


    difficult[
        "analysis_group"
    ] = (
        difficult[
            OUTCOME_COL
        ]
        .map(
            OUTCOME_MAP
        )
    )


    # ------------------------------------------------------------
    # Check for unmapped outcomes
    # ------------------------------------------------------------

    missing_group = (
        difficult[
            "analysis_group"
        ]
        .isna()
        .sum()
    )


    if missing_group > 0:

        print(
            "\nADVERTENCIA:"
        )

        print(
            f"{missing_group} samples could not be "
            f"mapped to an outcome group."
        )


    # ============================================================
    # 3. Load original references
    # ============================================================

    gt = (
        load_json_captions(args.original_references, "GT_original")
        .drop(columns="source")
        .rename(columns={"caption": "GT_original_caption"})
    )


    # ============================================================
    # 4. Join difficult samples to references
    # ============================================================

    analysis_df = difficult.merge(
        gt,
        on="image_id",
        how="inner"
    )


    print(
        "\n" +
        "=" * 80
    )

    print(
        "JOIN WITH ORIGINAL REFERENCES"
    )

    print(
        "=" * 80
    )


    print(
        f"Samples before the join: "
        f"{len(difficult):,}"
    )


    print(
        f"Samples after the join: "
        f"{len(analysis_df):,}"
    )


    # ============================================================
    # 5. Detect non-visual content in original references
    # ============================================================

    detected = (
        analysis_df[
            "GT_original_caption"
        ]
        .apply(
            analyze_caption
        )
        .apply(
            pd.Series
        )
    )


    analysis_df = pd.concat(
        [
            analysis_df.reset_index(
                drop=True
            ),

            detected.reset_index(
                drop=True
            )
        ],
        axis=1
    )


    # ============================================================
    # 6. Summarize by outcome
    # ============================================================

    GROUP_ORDER = [

        "Recovered",
        "Improved_but_low",
        "Not_improved"

    ]


    summary_rows = []


    for group_name in GROUP_ORDER:

        group = analysis_df[
            analysis_df[
                "analysis_group"
            ] == group_name
        ]


        n = len(
            group
        )


        if n == 0:

            continue


        row = {

            "group":
                group_name,

            "n_samples":
                n,

            "n_any_nonvisual":
                int(
                    group[
                        "has_non_visual"
                    ]
                    .sum()
                ),

            "pct_any_nonvisual":
                (
                    group[
                        "has_non_visual"
                    ]
                    .mean()
                    * 100
                ),

            "mean_nonvisual_categories":
                group[
                    "n_non_visual_categories"
                ]
                .mean()
        }


        for category in CATEGORIES:

            row[
                f"n_{category}"
            ] = int(
                group[
                    category
                ]
                .sum()
            )


            row[
                f"pct_{category}"
            ] = (
                group[
                    category
                ]
                .mean()
                * 100
            )


        summary_rows.append(
            row
        )


    summary = pd.DataFrame(
        summary_rows
    )


    # ============================================================
    # 7. Calculate recovery by non-visual status
    # ============================================================

    analysis_df[
        "is_recovered"
    ] = (
        analysis_df[
            "analysis_group"
        ] == "Recovered"
    )


    association_rows = []


    for nonvisual_status in [
        True,
        False
    ]:

        group = analysis_df[
            analysis_df[
                "has_non_visual"
            ] == nonvisual_status
        ]


        n = len(
            group
        )


        n_recovered = int(
            group[
                "is_recovered"
            ]
            .sum()
        )


        pct_recovered = (
            n_recovered
            / n
            * 100
            if n > 0
            else 0
        )


        association_rows.append(
            {

                "GT_has_non_visual":
                    nonvisual_status,

                "n_samples":
                    n,

                "n_recovered":
                    n_recovered,

                "pct_recovered":
                    pct_recovered
            }
        )


    association = pd.DataFrame(
        association_rows
    )


    # ============================================================
    # 8. Build the recovery contingency table
    # ============================================================

    contingency = pd.crosstab(
        analysis_df[
            "has_non_visual"
        ],
        analysis_df[
            "is_recovered"
        ],
        margins=True
    )


    # ============================================================
    # 9. Summarize length changes by outcome
    #
    # Length differences come from difficult_samples_transitions.csv
    # when the input contains those columns.
    # ============================================================

    length_rows = []


    if "length_change" in analysis_df.columns:

        for group_name in GROUP_ORDER:

            group = analysis_df[
                analysis_df[
                    "analysis_group"
                ] == group_name
            ]


            if len(group) == 0:

                continue


            length_rows.append(
                {

                    "group":
                        group_name,

                    "n_samples":
                        len(group),

                    "mean_length_original":
                        group[
                            "clean_words_original"
                        ]
                        .mean()
                        if "clean_words_original" in group.columns
                        else None,

                    "mean_length_canda":
                        group[
                            "clean_words_canda"
                        ]
                        .mean()
                        if "clean_words_canda" in group.columns
                        else None,

                    "mean_length_change":
                        group[
                            "length_change"
                        ]
                        .mean()
                }
            )


    length_summary = pd.DataFrame(
        length_rows
    )


    # ============================================================
    # 10. Select positive cases for review
    # ============================================================

    positive_cases = analysis_df[
        analysis_df[
            "has_non_visual"
        ]
    ].copy()


    positive_cases = positive_cases.sort_values(
        [
            "analysis_group",
            "n_non_visual_categories"
        ],
        ascending=[
            True,
            False
        ]
    )


    # ============================================================
    # 11. Count patterns by outcome
    # ============================================================

    pattern_rows = []


    for group_name in GROUP_ORDER:

        group = analysis_df[
            analysis_df[
                "analysis_group"
            ] == group_name
        ]


        for category, pattern_list in PATTERNS.items():

            for label, _ in pattern_list:

                count = (
                    group[
                        "matched_patterns"
                    ]
                    .fillna("")
                    .str.split(
                        " | ",
                        regex=False
                    )
                    .apply(
                        lambda x:
                            label in x
                    )
                    .sum()
                )


                pattern_rows.append(
                    {

                        "group":
                            group_name,

                        "category":
                            category,

                        "pattern":
                            label,

                        "n_captions":
                            int(
                                count
                            ),

                        "pct_group":
                            (
                                count
                                / len(group)
                                * 100
                                if len(group) > 0
                                else 0
                            )
                    }
                )


    pattern_frequency = pd.DataFrame(
        pattern_rows
    )


    # ============================================================
    # 12. Describe the methodology
    # ============================================================

    methodology = pd.DataFrame(
        [

            {
                "Parameter":
                    "Population",

                "Definition":
                    "Samples with baseline Qwen Original BERTScore <= 0.50"
            },

            {
                "Parameter":
                    "Recovered",

                "Definition":
                    "Qwen Original BERTScore <= 0.50 and Qwen C&A BERTScore > 0.50"
            },

            {
                "Parameter":
                    "Improved but low",

                "Definition":
                    "Qwen C&A BERTScore improved but remained <= 0.50"
            },

            {
                "Parameter":
                    "Not improved",

                "Definition":
                    "Qwen C&A BERTScore did not exceed Qwen Original for the same image"
            },

            {
                "Parameter":
                    "Non-visual reference",

                "Definition":
                    "Original ground-truth caption"
            },

            {
                "Parameter":
                    "Non-visual detector",

                "Definition":
                    "Same rule-based regular-expression detector used in the previous non-visual content analysis"
            },

            {
                "Parameter":
                    "Categories",

                "Definition":
                    "Patient reference, age, sex/gender, temporal context, figure/document reference, and clinical context"
            },

            {
                "Parameter":
                    "Any non-visual",

                "Definition":
                    "At least one pattern from any non-visual category detected in the original GT"
            }
        ]
    )


    # ============================================================
    # 13. Export CSV tables
    # ============================================================

    analysis_df.to_csv(
        OUTPUT_DIR /
        "difficult_nonvisual_per_sample.csv",
        index=False
    )


    summary.to_csv(
        OUTPUT_DIR /
        "difficult_nonvisual_summary.csv",
        index=False
    )


    association.to_csv(
        OUTPUT_DIR /
        "recovery_by_nonvisual_status.csv",
        index=False
    )


    pattern_frequency.to_csv(
        OUTPUT_DIR /
        "difficult_nonvisual_pattern_frequency.csv",
        index=False
    )


    positive_cases.to_csv(
        OUTPUT_DIR /
        "difficult_nonvisual_positive_cases.csv",
        index=False
    )


    if len(length_summary) > 0:

        length_summary.to_csv(
            OUTPUT_DIR /
            "difficult_nonvisual_length_summary.csv",
            index=False
        )


    # ============================================================
    # 14. Export the workbook
    # ============================================================

    EXCEL_FILE = (
        OUTPUT_DIR /
        "difficult_nonvisual_analysis.xlsx"
    )


    with pd.ExcelWriter(
        EXCEL_FILE,
        engine="openpyxl"
    ) as writer:


        methodology.to_excel(
            writer,
            sheet_name="Methodology",
            index=False
        )


        summary.to_excel(
            writer,
            sheet_name="Group_summary",
            index=False
        )


        association.to_excel(
            writer,
            sheet_name="Recovery_by_nonvisual",
            index=False
        )


        contingency.to_excel(
            writer,
            sheet_name="Contingency"
        )


        pattern_frequency.to_excel(
            writer,
            sheet_name="Pattern_frequency",
            index=False
        )


        positive_cases.to_excel(
            writer,
            sheet_name="Positive_cases",
            index=False
        )


        if len(length_summary) > 0:

            length_summary.to_excel(
                writer,
                sheet_name="Length_summary",
                index=False
            )


        analysis_df.to_excel(
            writer,
            sheet_name="Per_sample",
            index=False
        )


    # ============================================================
    # 15. Display results
    # ============================================================

    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        260
    )


    print(
        "\n" +
        "=" * 100
    )

    print(
        "NON-VISUAL CONTENT BY DIFFICULT SAMPLE OUTCOME"
    )

    print(
        "=" * 100
    )


    print(
        summary.to_string(
            index=False
        )
    )


    print(
        "\n" +
        "=" * 100
    )

    print(
        "RECOVERY RATE BY NON-VISUAL CONTENT STATUS"
    )

    print(
        "=" * 100
    )


    print(
        association.to_string(
            index=False
        )
    )


    print(
        "\n" +
        "=" * 100
    )

    print(
        "CONTINGENCY TABLE"
    )

    print(
        "=" * 100
    )


    print(
        contingency
    )


    if len(length_summary) > 0:

        print(
            "\n" +
            "=" * 100
        )

        print(
            "CAPTION LENGTH BY OUTCOME"
        )

        print(
            "=" * 100
        )

        print(
            length_summary.to_string(
                index=False
            )
        )


    print(
        "\n" +
        "=" * 100
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 100
    )


    print(
        "\nExcel:"
    )

    print(
        EXCEL_FILE
    )


if __name__ == "__main__":
    main()
