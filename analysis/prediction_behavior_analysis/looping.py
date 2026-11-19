"""Detect consecutive n-gram loops in postprocessed Qwen captions."""

import argparse
import re
from pathlib import Path

import pandas as pd


IMAGE_COL = "image_id"

CAPTION_COL = "qwen_caption_clean"

STATUS_COL = "status"


# ============================================================
# Detection settings
# ============================================================

N_VALUES = [
    1,
    2,
    3,
    4,
]


MIN_CONSECUTIVE_REPETITIONS = 2


# ============================================================
# Load predictions
# ============================================================

def load_predictions(path, source):
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, dtype={IMAGE_COL: str}, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype={IMAGE_COL: str}, keep_default_na=False)
    missing = {IMAGE_COL, CAPTION_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    if STATUS_COL in df.columns:
        df = df.loc[df[STATUS_COL].eq("ok")]
    captions = df[CAPTION_COL].astype(str).str.strip()
    result = pd.DataFrame({"source": source, "image_id": df[IMAGE_COL], "caption": captions})
    return result.loc[captions.ne("")].reset_index(drop=True)


def tokenize(
    text,
):

    # --------------------------------------------------------
    # Lowercase text for detection
    # --------------------------------------------------------

    text = str(
        text
    ).lower()


    # --------------------------------------------------------
    # Replace punctuation with spaces.

    # For example:

    # edema
    # edema.

    # are treated as the same token.
    # --------------------------------------------------------

    text = re.sub(
        r"[^\w]+",
        " ",
        text,
    )


    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


    if text == "":

        return []


    return (
        text.split()
    )


# ============================================================
# Find consecutive repeats for one n-gram size
# ============================================================

def find_max_consecutive_loop(
    tokens,
    n,
):

    """
    Find the longest run of consecutive repeats of an n-token block.

    Returns:

        max_run
        repeated_ngram
        start_position
    """


    total_tokens = len(
        tokens
    )


    # Require at least two complete blocks
    if total_tokens < (
        n
        * MIN_CONSECUTIVE_REPETITIONS
    ):

        return (
            1,
            "",
            -1,
        )


    best_run = 1

    best_ngram = ""

    best_start = -1


    # --------------------------------------------------------
    # Try each token as a possible start
    # --------------------------------------------------------

    for i in range(
        0,
        total_tokens - n + 1
    ):

        current_block = tuple(
            tokens[
                i:i+n
            ]
        )


        run = 1


        next_start = (
            i + n
        )


        # ----------------------------------------------------
        # Compare non-overlapping blocks
        # ----------------------------------------------------

        while (
            next_start + n
            <= total_tokens
        ):

            next_block = tuple(
                tokens[
                    next_start:
                    next_start+n
                ]
            )


            if (
                next_block
                == current_block
            ):

                run += 1

                next_start += n

            else:

                break


        # ----------------------------------------------------
        # Track the longest run
        # ----------------------------------------------------

        if run > best_run:

            best_run = run

            best_ngram = (
                " ".join(
                    current_block
                )
            )

            best_start = i


    return (
        best_run,
        best_ngram,
        best_start,
    )


# ============================================================
# Analyze one caption
# ============================================================

def analyze_caption(
    caption,
):

    tokens = tokenize(
        caption
    )


    result = {

        "n_words":
            len(tokens),

    }


    loop_levels = []


    overall_max_run = 1

    overall_loop_ngram = ""

    overall_loop_n = 0


    # ========================================================
    # CHECK N-GRAM SIZES 1 THROUGH 4
    # ========================================================

    for n in N_VALUES:

        (
            max_run,
            loop_ngram,
            loop_start,
        ) = find_max_consecutive_loop(
            tokens,
            n
        )


        has_loop = (
            max_run
            >= MIN_CONSECUTIVE_REPETITIONS
        )


        # ----------------------------------------------------
        # Record the result for this n-gram size
        # ----------------------------------------------------

        result[
            f"has_loop_{n}gram"
        ] = has_loop


        result[
            f"max_run_{n}gram"
        ] = max_run


        result[
            f"loop_ngram_{n}gram"
        ] = loop_ngram


        result[
            f"loop_start_{n}gram"
        ] = loop_start


        if has_loop:

            loop_levels.append(
                n
            )


        # ----------------------------------------------------
        # Track the longest loop in the caption
        # ----------------------------------------------------

        if max_run > overall_max_run:

            overall_max_run = (
                max_run
            )

            overall_loop_ngram = (
                loop_ngram
            )

            overall_loop_n = (
                n
            )


    # ========================================================
    # OVERALL LOOP INDICATOR
    # ========================================================

    result[
        "has_any_loop"
    ] = (
        len(loop_levels) > 0
    )


    # ========================================================
    # LARGEST N-GRAM SIZE WITH A LOOP
    # ========================================================

    if loop_levels:

        result[
            "highest_loop_ngram"
        ] = max(
            loop_levels
        )

    else:

        result[
            "highest_loop_ngram"
        ] = 0


    # ========================================================
    # OVERALL SEVERITY
    # ========================================================

    result[
        "max_consecutive_repetitions"
    ] = (
        overall_max_run
    )


    result[
        "most_severe_loop_ngram"
    ] = (
        overall_loop_ngram
    )


    result[
        "most_severe_loop_n"
    ] = (
        overall_loop_n
    )


    return result


# ============================================================
# Analyze all captions
# ============================================================

def analyze_dataframe(
    df,
):

    print(
        f"\nAnalizando: "
        f"{df['source'].iloc[0]}"
    )


    analysis = (
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

            analysis.reset_index(
                drop=True
            ),
        ],

        axis=1,
    )


    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-predictions", type=Path)
    parser.add_argument("--clean-paraphrase-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    inputs = (
        ("Qwen_original", args.original_predictions),
        ("Qwen_C&A", args.clean_paraphrase_predictions),
    )
    selected = [(source, path) for source, path in inputs if path is not None]
    if not selected:
        parser.error("Provide at least one predictions file.")

    results = []
    for source, path in selected:
        if not path.is_file():
            parser.error(f"Predictions file was not found: {path}")
        captions = load_predictions(path, source)
        if captions.empty:
            parser.error(f"No captions to analyze in {path}")
        results.append(analyze_dataframe(captions))
    all_results = pd.concat(results, ignore_index=True)

    OUTPUT_DIR = args.output_dir or Path("runs/looping")
    if args.output_dir is None and len(selected) == 1:
        OUTPUT_DIR /= selected[0][0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    PER_SAMPLE_FILE = OUTPUT_DIR / "looping_per_sample.csv"
    SUMMARY_FILE = OUTPUT_DIR / "looping_summary.csv"
    EXCLUSIVE_FILE = OUTPUT_DIR / "looping_exclusive_summary.csv"
    FREQUENCY_FILE = OUTPUT_DIR / "looping_frequency_by_ngram.csv"
    SEVERITY_FILE = OUTPUT_DIR / "looping_severity_summary.csv"
    POSITIVE_FILE = OUTPUT_DIR / "looping_positive_cases_review.csv"
    COMPARISON_FILE = OUTPUT_DIR / "looping_reduction.csv"
    EXCEL_FILE = OUTPUT_DIR / "looping_analysis_complete.xlsx"

    # Summarize looping by source.
    summary_rows = []


    for source, group in all_results.groupby(
        "source"
    ):

        n_samples = len(
            group
        )


        loops = (
            group[
                group[
                    "has_any_loop"
                ]
            ]
        )


        row = {

            "source":
                source,

            "n_samples":
                n_samples,

            "n_any_loop":
                int(
                    group[
                        "has_any_loop"
                    ]
                    .sum()
                ),

            "pct_any_loop":
                (
                    group[
                        "has_any_loop"
                    ]
                    .mean()
                    * 100
                ),

            "mean_words":
                float(
                    group[
                        "n_words"
                    ]
                    .mean()
                ),

            "mean_max_run_positive":
                (
                    float(
                        loops[
                            "max_consecutive_repetitions"
                        ]
                        .mean()
                    )

                    if len(loops) > 0

                    else 0.0
                ),

            "max_run_observed":
                int(
                    group[
                        "max_consecutive_repetitions"
                    ]
                    .max()
                ),
        }



        for n in N_VALUES:

            row[
                f"n_loop_{n}gram"
            ] = int(
                group[
                    f"has_loop_{n}gram"
                ]
                .sum()
            )


            row[
                f"pct_loop_{n}gram"
            ] = (
                group[
                    f"has_loop_{n}gram"
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



    # Assign each positive caption to its largest looping n-gram size.
    exclusive_rows = []


    for source, group in all_results.groupby(
        "source"
    ):

        total = len(
            group
        )


        for n in [
            0,
            1,
            2,
            3,
            4,
        ]:

            count = int(
                (
                    group[
                        "highest_loop_ngram"
                    ] == n
                )
                .sum()
            )


            exclusive_rows.append(
                {
                    "source":
                        source,

                    "loop_ngram_size":
                        n,

                    "n_samples":
                        count,

                    "pct_samples":
                        (
                            count
                            / total
                            * 100
                        ),
                }
            )


    exclusive_summary = pd.DataFrame(
        exclusive_rows
    )









    # Count consecutive repetitions by n-gram size.
    frequency_rows = []


    for source, group in all_results.groupby(
        "source"
    ):

        total = len(
            group
        )


        for n in N_VALUES:

            run_column = (
                f"max_run_{n}gram"
            )


            # ----------------------------------------------------
            # Count only runs that meet the looping threshold
            # run >= 2
            # ----------------------------------------------------

            positive_runs = (
                group[
                    group[
                        run_column
                    ]
                    >= MIN_CONSECUTIVE_REPETITIONS
                ][
                    run_column
                ]
            )


            if len(
                positive_runs
            ) == 0:

                continue


            run_counts = (
                positive_runs
                .value_counts()
                .sort_index()
            )


            for run_length, count in run_counts.items():

                frequency_rows.append(
                    {

                        "source":
                            source,

                        "ngram_size":
                            n,

                        "consecutive_repetitions":
                            int(
                                run_length
                            ),

                        "n_captions":
                            int(
                                count
                            ),

                        "pct_total_captions":
                            (
                                count
                                / total
                                * 100
                            ),

                        "pct_within_ngram_loop":
                            (
                                count
                                / len(
                                    positive_runs
                                )
                                * 100
                            ),
                    }
                )


    frequency_summary = pd.DataFrame(
        frequency_rows
    )





    # Group run lengths into four reporting bins.
    compact_frequency_rows = []


    for source, group in all_results.groupby(
        "source"
    ):

        for n in N_VALUES:

            values = (
                group[
                    f"max_run_{n}gram"
                ]
            )


            n_2 = int(
                (
                    values == 2
                )
                .sum()
            )


            n_3 = int(
                (
                    values == 3
                )
                .sum()
            )


            n_4 = int(
                (
                    values == 4
                )
                .sum()
            )


            n_5plus = int(
                (
                    values >= 5
                )
                .sum()
            )


            compact_frequency_rows.append(
                {

                    "source":
                        source,

                    "ngram_size":
                        n,

                    "2_repetitions":
                        n_2,

                    "3_repetitions":
                        n_3,

                    "4_repetitions":
                        n_4,

                    "5plus_repetitions":
                        n_5plus,

                    "total_loop":
                        (
                            n_2
                            + n_3
                            + n_4
                            + n_5plus
                        ),
                }
            )


    compact_frequency = pd.DataFrame(
        compact_frequency_rows
    )



    positive_cases = (
        all_results[
            all_results[
                "has_any_loop"
            ]
        ]
        .copy()
    )


    positive_cases = (
        positive_cases
        .sort_values(
            [
                "max_consecutive_repetitions",
                "highest_loop_ngram",
            ],

            ascending=[
                False,
                False,
            ],
        )
    )



    # Summarize the longest run per positive caption.
    severity_rows = []


    for source, group in positive_cases.groupby(
        "source"
    ):

        total_positive = len(
            group
        )


        counts = (
            group[
                "max_consecutive_repetitions"
            ]
            .value_counts()
            .sort_index()
        )


        for run_length, count in counts.items():

            severity_rows.append(
                {

                    "source":
                        source,

                    "consecutive_repetitions":
                        int(
                            run_length
                        ),

                    "n_cases":
                        int(
                            count
                        ),

                    "pct_loop_cases":
                        (
                            count
                            / total_positive
                            * 100
                        ),
                }
            )


    severity_summary = pd.DataFrame(
        severity_rows
    )



    def get_pct(
        source_name,
    ):

        row = (
            summary[
                summary[
                    "source"
                ] == source_name
            ]
        )


        if len(
            row
        ) == 0:

            return None


        return float(
            row.iloc[0][
                "pct_any_loop"
            ]
        )


    original_pct = get_pct(
        "Qwen_original"
    )


    canda_pct = get_pct(
        "Qwen_C&A"
    )


    if (
        original_pct is not None
        and canda_pct is not None
        and original_pct != 0
    ):

        relative_reduction = (
            (
                original_pct
                - canda_pct
            )
            / original_pct
            * 100
        )

    else:

        relative_reduction = None


    comparison = pd.DataFrame(
        [
            {

                "comparison":
                    "Qwen original -> Qwen C&A",

                "before_pct":
                    original_pct,

                "after_pct":
                    canda_pct,

                "relative_reduction_pct":
                    relative_reduction,
            }
        ]
    )



    methodology = pd.DataFrame(
        [
            {
                "Parameter":
                    "Analysis",

                "Definition":
                    "Consecutive n-gram looping"
            },

            {
                "Parameter":
                    "Models",

                "Definition":
                    ", ".join(all_results["source"].unique())
            },

            {
                "Parameter":
                    "N-gram sizes",

                "Definition":
                    "1, 2, 3, and 4 words"
            },

            {
                "Parameter":
                    "Loop criterion",

                "Definition":
                    "Identical n-gram repeated immediately and consecutively at least twice"
            },

            {
                "Parameter":
                    "Minimum repetitions",

                "Definition":
                    "2"
            },

            {
                "Parameter":
                    "Tokenization",

                "Definition":
                    "Lowercase; punctuation replaced by spaces; whitespace normalized"
            },

            {
                "Parameter":
                    "Overlapping blocks",

                "Definition":
                    "Repeated blocks are compared in non-overlapping segments of size n"
            },

            {
                "Parameter":
                    "Any loop",

                "Definition":
                    "Caption is positive if looping is detected for at least one n-gram size"
            },

            {
                "Parameter":
                    "Exclusive class",

                "Definition":
                    "Positive caption assigned to the largest n-gram size exhibiting looping"
            },
        ]
    )



    all_results.to_csv(
        PER_SAMPLE_FILE,
        index=False,
    )


    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )


    exclusive_summary.to_csv(
        EXCLUSIVE_FILE,
        index=False,
    )


    frequency_summary.to_csv(
        FREQUENCY_FILE,
        index=False,
    )


    severity_summary.to_csv(
        SEVERITY_FILE,
        index=False,
    )


    positive_cases.to_csv(
        POSITIVE_FILE,
        index=False,
    )


    if len(selected) == 2:
        comparison.to_csv(COMPARISON_FILE, index=False)



    with pd.ExcelWriter(
        EXCEL_FILE,
        engine="openpyxl"
    ) as writer:



        methodology.to_excel(
            writer,
            sheet_name=
                "Methodology",
            index=False,
        )



        summary.to_excel(
            writer,
            sheet_name=
                "Summary",
            index=False,
        )



        if len(selected) == 2:
            comparison.to_excel(writer, sheet_name="Reduction", index=False)



        exclusive_summary.to_excel(
            writer,
            sheet_name=
                "Exclusive_by_N",
            index=False,
        )


        #

        compact_frequency.to_excel(
            writer,
            sheet_name=
                "Frequency_compact",
            index=False,
        )



        frequency_summary.to_excel(
            writer,
            sheet_name=
                "Frequency_full",
            index=False,
        )



        severity_summary.to_excel(
            writer,
            sheet_name=
                "Severity",
            index=False,
        )



        positive_cases.to_excel(
            writer,
            sheet_name=
                "Positive_cases",
            index=False,
        )



        all_results.to_excel(
            writer,
            sheet_name=
                "Per_sample",
            index=False,
        )
        for worksheet in writer.sheets.values():
            for row in worksheet.iter_rows(min_row=2):
                for cell in row:
                    if cell.data_type == "f":
                        cell.data_type = "s"

    print(summary.to_string(index=False))
    print(f"Reports saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
