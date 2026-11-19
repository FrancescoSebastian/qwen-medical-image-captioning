"""Compare paired Qwen scores for samples with low baseline BERTScore F1."""

import argparse
from pathlib import Path

import pandas as pd


# 1. Analysis settings

IMAGE_COL = "image_id"

LENGTH_COL = "clean_words"

BERT_COL = "BERTScore_F1_clean"

STATUS_COL = "status"

BERT_THRESHOLD = 0.50

LENGTH_THRESHOLD = 21


# ============================================================
# 2. Load an evaluation table
# ============================================================

def load_data(path, source_name):

    print("\n" + "=" * 70)
    print(f"LOADING: {source_name}")
    print("=" * 70)

    read_table = pd.read_excel if path.suffix.lower() == ".xlsx" else pd.read_csv
    df = read_table(path, dtype={IMAGE_COL: str})

    required = [
        IMAGE_COL,
        LENGTH_COL,
        BERT_COL
    ]

    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"Missing column '{col}' in:\n{path}"
            )


    # --------------------------------------------------------
    # Keep successful inferences
    # --------------------------------------------------------

    if STATUS_COL in df.columns:

        df = df[
            df[STATUS_COL] == "ok"
        ].copy()


    # --------------------------------------------------------
    # Convert metric values to numbers
    # --------------------------------------------------------

    df[LENGTH_COL] = pd.to_numeric(
        df[LENGTH_COL],
        errors="coerce"
    )

    df[BERT_COL] = pd.to_numeric(
        df[BERT_COL],
        errors="coerce"
    )


    df = df.dropna(subset=[IMAGE_COL]).copy()
    df[IMAGE_COL] = (
        df[IMAGE_COL]
        .astype(str)
        .str.strip()
    )


    df = (
        df
        .dropna(
            subset=[
                LENGTH_COL,
                BERT_COL
            ]
        )
        .reset_index(drop=True)
    )


    print(
        f"Eligible samples: {len(df):,}"
    )

    print(
        f"Mean BERTScore: {df[BERT_COL].mean():.6f}"
    )

    return df.loc[df[IMAGE_COL].ne("")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-metrics", type=Path, required=True)
    parser.add_argument("--clean-paraphrase-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sample_level_performance"))
    args = parser.parse_args()
    for path in (args.original_metrics, args.clean_paraphrase_metrics):
        if not path.is_file():
            parser.error(f"Metrics file was not found: {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR = args.output_dir

    # 3. Load per-sample metrics

    original = load_data(
        args.original_metrics,
        "Qwen Original"
    )

    canda = load_data(
        args.clean_paraphrase_metrics,
        "Qwen C&P"
    )


    # ============================================================
    # 4. Pair predictions by image ID
    # ============================================================

    paired = (

        original[
            [
                IMAGE_COL,
                LENGTH_COL,
                BERT_COL
            ]
        ]

        .merge(

            canda[
                [
                    IMAGE_COL,
                    LENGTH_COL,
                    BERT_COL
                ]
            ],

            on=IMAGE_COL,

            how="inner",
            validate="one_to_one",

            suffixes=(
                "_original",
                "_canda"
            )
        )
    )


    print("\n" + "=" * 70)
    print("PAIRED SAMPLES")
    print("=" * 70)

    print(
        f"Total: {len(paired):,}"
    )


    # ============================================================
    # 5. Select low-baseline samples
    #
    # 
    # Select samples using Qwen Original scores.
    # ============================================================

    difficult = paired[

        paired[
            f"{BERT_COL}_original"
        ] <= BERT_THRESHOLD

    ].copy()


    if difficult.empty:
        parser.error("No paired samples have a baseline BERTScore F1 at or below 0.50.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("LOW-BASELINE SUBSET")
    print("=" * 70)

    print(
        f"BERTScore Original <= {BERT_THRESHOLD}"
    )

    print(
        f"Samples: {len(difficult):,}"
    )


    # ============================================================
    # 6. Calculate BERTScore changes
    # ============================================================

    difficult["delta_BERTScore"] = (

        difficult[
            f"{BERT_COL}_canda"
        ]

        -

        difficult[
            f"{BERT_COL}_original"
        ]
    )


    difficult["improved"] = (

        difficult[
            "delta_BERTScore"
        ] > 0
    )


    # ============================================================
    # 7. Detect threshold crossings
    #
    # crossed_threshold:
    #
    # Original <= .50
    # C&A      >  .50
    # ============================================================

    difficult["crossed_threshold"] = (

        difficult[
            f"{BERT_COL}_canda"
        ] > BERT_THRESHOLD
    )


    # ============================================================
    # 8. Classify outcomes
    # ============================================================

    def classify_outcome(row):

        if row["crossed_threshold"]:

            return "Improved and crossed >0.50"

        elif row["improved"]:

            return "Improved but remained <=0.50"

        else:

            return "Not improved"


    difficult["outcome"] = difficult.apply(
        classify_outcome,
        axis=1
    )


    # ============================================================
    # 9. Summarize transitions
    # ============================================================

    n_total = len(difficult)

    n_improved = int(
        difficult["improved"].sum()
    )

    n_crossed = int(
        difficult["crossed_threshold"].sum()
    )

    n_improved_below = int(
        (
            difficult["improved"]
            &
            ~difficult["crossed_threshold"]
        ).sum()
    )

    n_not_improved = (
        n_total
        -
        n_improved
    )


    pct_improved = (
        n_improved /
        n_total *
        100
    )

    pct_crossed = (
        n_crossed /
        n_total *
        100
    )

    pct_crossed_improved = (
        n_crossed /
        n_improved *
        100
        if n_improved > 0
        else 0
    )


    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"Total baseline <=0.50: "
        f"{n_total:,}"
    )

    print(
        f"Improved with C&P: "
        f"{n_improved:,} "
        f"({pct_improved:.2f}%)"
    )

    print(
        f"Improved but remained <=0.50: "
        f"{n_improved_below:,}"
    )

    print(
        f"Crossed from <=0.50 to >0.50: "
        f"{n_crossed:,} "
        f"({pct_crossed:.2f}% of the subset)"
    )

    print(
        f"Among improved samples, crossed >0.50: "
        f"{pct_crossed_improved:.2f}%"
    )

    print(
        f"Did not improve: "
        f"{n_not_improved:,}"
    )


    # ============================================================
    # 10. Measure caption length changes
    # ============================================================

    difficult["length_change"] = (

        difficult[
            f"{LENGTH_COL}_canda"
        ]

        -

        difficult[
            f"{LENGTH_COL}_original"
        ]
    )


    difficult["became_short"] = (

        (
            difficult[
                f"{LENGTH_COL}_original"
            ] > LENGTH_THRESHOLD
        )

        &

        (
            difficult[
                f"{LENGTH_COL}_canda"
            ] <= LENGTH_THRESHOLD
        )
    )


    print("\n" + "=" * 80)
    print("CAPTION LENGTH")
    print("=" * 80)


    print(
        "Mean original caption length:",
        round(
            difficult[
                f"{LENGTH_COL}_original"
            ].mean(),
            3
        )
    )


    print(
        "Mean C&P caption length:",
        round(
            difficult[
                f"{LENGTH_COL}_canda"
            ].mean(),
            3
        )
    )


    print(
        "Mean change:",
        round(
            difficult[
                "length_change"
            ].mean(),
            3
        ),
        "words"
    )


    print(
        "Went from >21 to <=21 words:",
        int(
            difficult[
                "became_short"
            ].sum()
        )
    )


    # ============================================================
    # 11. Build the summary table
    # ============================================================

    summary = pd.DataFrame(
        [
            {
                "category":
                    "Baseline BERTScore <= 0.50",

                "n":
                    n_total,

                "pct":
                    100.0
            },

            {
                "category":
                    "Improved with C&P",

                "n":
                    n_improved,

                "pct":
                    pct_improved
            },

            {
                "category":
                    "Improved but remained <= 0.50",

                "n":
                    n_improved_below,

                "pct":
                    n_improved_below /
                    n_total *
                    100
            },

            {
                "category":
                    "Crossed to > 0.50",

                "n":
                    n_crossed,

                "pct":
                    pct_crossed
            },

            {
                "category":
                    "Not improved",

                "n":
                    n_not_improved,

                "pct":
                    n_not_improved /
                    n_total *
                    100
            }
        ]
    )


    # ============================================================
    # 12. Export CSV reports
    # ============================================================

    difficult.to_csv(
        OUTPUT_DIR /
        "difficult_samples_transitions.csv",
        index=False
    )


    summary.to_csv(
        OUTPUT_DIR /
        "difficult_samples_summary.csv",
        index=False
    )


    # ============================================================
    # 13. Create the two-panel figure
    # ============================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True
    )


    # ============================================================
    # 14. Draw one panel
    # ============================================================

    def plot_panel(
        ax,
        x_col,
        y_col,
        title
    ):

        # --------------------------------------------------------
        # Did not improve
        # --------------------------------------------------------

        no_improvement = difficult[
            ~difficult["improved"]
        ]


        # --------------------------------------------------------
        # Improved
        # --------------------------------------------------------

        improvement = difficult[
            difficult["improved"]
        ]


        # --------------------------------------------------------
        # Background: samples without improvement
        # --------------------------------------------------------

        ax.scatter(

            no_improvement[x_col],
            no_improvement[y_col],

            s=24,

            alpha=0.8,

            color="#4C78A8", #azul

            edgecolors="none",

            label="Not improved"
        )


        # --------------------------------------------------------
        # Highlight improvements
        # --------------------------------------------------------

        ax.scatter(

            improvement[x_col],
            improvement[y_col],

            s=28,

            alpha=0.8,

            color="#C44E52", #rojo

            edgecolors="none",

            label="Improved with C&P"
        )


        # --------------------------------------------------------
        # Thresholds
        # --------------------------------------------------------

        ax.axhline(
            BERT_THRESHOLD,
            linestyle="--",
            linewidth=1.3,
            color="black"
        )


        ax.axvline(
            LENGTH_THRESHOLD,
            linestyle="--",
            linewidth=1.3,
            color="black"
        )


        ax.set_title(
            title,
            fontsize=13
        )


        ax.set_xlabel(
            "Prediction caption length (words)"
        )


        ax.grid(
            alpha=0.12
        )


    # ============================================================
    # 15. Plot Qwen Original
    # ============================================================

    plot_panel(

        axes[0],

        f"{LENGTH_COL}_original",

        f"{BERT_COL}_original",

        "(a) Qwen-O"
    )


    # ============================================================
    # 16. Plot Qwen C&P
    # ============================================================

    plot_panel(

        axes[1],

        f"{LENGTH_COL}_canda",

        f"{BERT_COL}_canda",

        "(b) Qwen-C&P"
    )


    # ============================================================
    # 17. Label the y-axis
    # ============================================================

    axes[0].set_ylabel(
        "BERTScore F1"
    )


    # ============================================================
    # 18. Set shared axis limits
    # ============================================================

    all_lengths = pd.concat(
        [
            difficult[
                f"{LENGTH_COL}_original"
            ],

            difficult[
                f"{LENGTH_COL}_canda"
            ]
        ]
    )


    x_max = max(
        30,
        all_lengths.quantile(0.99) + 3
    )


    axes[0].set_xlim(
        0,
        x_max
    )


    # Keep the 0.50 threshold visible.

    y_min = min(
        difficult[
            f"{BERT_COL}_original"
        ].min(),

        difficult[
            f"{BERT_COL}_canda"
        ].min()
    )


    y_max = max(
        difficult[
            f"{BERT_COL}_original"
        ].max(),

        difficult[
            f"{BERT_COL}_canda"
        ].max()
    )


    axes[0].set_ylim(
        max(0, y_min - 0.04),
        min(1, y_max + 0.04)
    )


    # ============================================================
    # 19. Annotate the Qwen C&P panel
    #
    # 
    # ============================================================

    axes[1].text(

        0.97,
        0.96,

        (
            f"Qwen-O BS≤0.50: {n_total}\n"
            f"Improved with Qwen-C&P: {n_improved} ({pct_improved:.1f}%)\n"
            f"Reached BS>0.50: {n_crossed} ({pct_crossed:.1f}%)"
        ),

        transform=
            axes[1].transAxes,

        horizontalalignment=
            "right",

        verticalalignment=
            "top",

        fontsize=9,

        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="gray"
        )
    )


    # ============================================================
    # 20. Add the shared legend
    # ============================================================

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )


    fig.legend(

        handles,
        labels,

        loc="lower center",

        ncol=2,

        frameon=False,

        bbox_to_anchor=(
            0.5,
            -0.01
        )
    )


    # ============================================================
    # 21. Adjust the layout
    # ============================================================

    plt.tight_layout(
        rect=[
            0,
            0.07,
            1,
            1
        ]
    )


    # ============================================================
    # 22. Export the figure
    # ============================================================

    PNG_FILE = (
        OUTPUT_DIR /
        "difficult_samples_before_after.png"
    )


    PDF_FILE = (
        OUTPUT_DIR /
        "difficult_samples_before_after.pdf"
    )


    plt.savefig(
        PNG_FILE,
        dpi=400,
        bbox_inches="tight"
    )


    plt.savefig(
        PDF_FILE,
        bbox_inches="tight"
    )


    plt.close()


    # ============================================================
    # 23. Export the Excel workbook
    # ============================================================

    EXCEL_FILE = (
        OUTPUT_DIR /
        "difficult_samples_analysis.xlsx"
    )


    with pd.ExcelWriter(
        EXCEL_FILE,
        engine="openpyxl"
    ) as writer:

        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        difficult.to_excel(
            writer,
            sheet_name="Per_sample",
            index=False
        )


    # ============================================================
    # 24. Show output paths
    # ============================================================

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print("\nFigure:")
    print(PNG_FILE)

    print("\nPDF:")
    print(PDF_FILE)

    print("\nExcel:")
    print(EXCEL_FILE)


if __name__ == "__main__":
    main()
