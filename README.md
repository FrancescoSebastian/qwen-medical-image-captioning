# Textual Preprocessing for Qwen-Based Medical Image Captioning

## Dataset access

This repository contains code and prompts. ImageCLEFmedical 2024 images,
reference captions, and datasets derived from them are not distributed here.
Obtain the dataset through the official ImageCLEFmedical access process.
Keep your local copy under `data/` or outside the repository and pass its
location to the scripts. Generated files under `runs/` are also excluded
from version control.

## Caption preprocessing

### Cleaning

Run the cleaning stage from the repository root. The input JSON maps each
image ID to its original caption or caption list. Use the training JSON
to construct training data. The same command can clean references for the
non-visual content analysis if given the test JSON and a separate output directory.

```bash
python -m pip install accelerate
python -m preprocessing.clean_captions \
  --input-json /path/to/train_2024_captions.json \
  --output-dir runs/preprocessing/train
```

Cleaning uses DeepSeek-R1-Distill-Llama-8B and the original
[cleaning prompt](preprocessing/cleaning_prompt.txt). The model writes
resumable blocks of 1,000 captions by default. Rerun with the same arguments
to continue. To use completed blocks from the original scripts without
loading the model, point to their directory and add `--reuse-blocks-only`.
The block files must cover the input JSON in its original order with the
same block size.

The script merges the blocks, writes rejection and word-count reports, and
exports `paraphrasing_input.json`. This file maps image IDs to one-element
caption lists. Empty captions, `NA` responses, and captions longer than
300 words are excluded, while one- and two-word captions are kept.
For cleaned test references, use this same output as the cleaned reference
JSON in the non-visual content analysis.

### Paraphrasing

```bash
python -m preprocessing.paraphrase_captions \
  --input-json runs/preprocessing/train/paraphrasing_input.json \
  --output-dir runs/preprocessing/paraphrasing
```

The stage uses the original [paraphrasing prompt](preprocessing/paraphrasing_prompt.txt).
Only cleaned captions of at least three words are sent to the model.
The final `paraphrased_captions.json` retains nonempty answers enclosed in
`<answer>` and `</answer>` with at most 300 words. The combined responses
and rejected IDs are saved separately. Use `--reuse-blocks-only` to
process existing `p_subset_*.json` blocks from the same eligible inputs
and block size without loading DeepSeek.

### Build the Qwen-C&P training CSV

```bash
python -m preprocessing.build_training_dataset \
  --cleaned-json runs/preprocessing/train/paraphrasing_input.json \
  --paraphrased-json runs/preprocessing/paraphrasing/paraphrased_captions.json \
  --output-csv runs/preprocessing/Train_2024_master_augmented_filtered.csv
```

The CSV contains one `ID,Caption` row per retained cleaned or paraphrased
caption. Short cleaned captions are included. Images with both variants
appear twice, even when the paraphrasing stage retained the original
wording. The script prints the counts for both stages and the total.

## Qwen LoRA fine-tuning

The training script uses OpenMed/Qwen3.5-2B-MedVL. It requires Python 3.11 or later, a CUDA-capable GPU, and an environment with PyTorch, Transformers, PEFT, pandas, Pillow, safetensors, and tqdm.

The ImageCLEFmedical 2024 data directory must contain the `train/` and
`valid/` image directories and the validation file `valid_2024_captions.csv`.

Pass the training CSV explicitly with `--train-csv`. For Qwen-O, use the
CSV with the original training captions (`train_original.csv` in the
example below). For Qwen-C&P, use
`runs/preprocessing/Train_2024_master_augmented_filtered.csv`, generated
by the preprocessing pipeline above.

Both the training and validation CSVs must contain the `ID` and `Caption`
columns. The loader keeps separate rows when an image ID appears with
multiple captions.

Run these commands from the repository root and replace the example paths with your own.

### Original captions

```bash
python qwen/train_qwen_lora.py \
  --variant original \
  --data-root /path/to/ImageCLEF2024 \
  --train-csv /path/to/train_original.csv
```

### Cleaned and paraphrased captions

```bash
python qwen/train_qwen_lora.py \
  --variant clean-paraphrase \
  --data-root /path/to/ImageCLEF2024 \
  --train-csv runs/preprocessing/Train_2024_master_augmented_filtered.csv
```

Pass an already prepared CSV for the selected variant. Both variants use the same validation CSV and hyperparameters. Their default output directories are `runs/original/` and `runs/clean-paraphrase/`. Use `--output-dir` to specify a different directory.

### Resuming a run

Repeat the command with the same variant, CSVs, and output directory. The script checks that the CSV contents match the run and loads the latest usable checkpoint. For a run created with the previous script, pass `--output-dir /path/to/the/run --resume-legacy` and use the exact CSVs from that run.

In Spyder, enter the command-line arguments in the script's run configuration.

## Qwen test inference

The test data directory must contain `test/` with the images and
`test_2024_captions.csv` with the `ID` and `Caption` columns. Use the same
test set for both adapters.

```bash
python qwen/infer_qwen_lora.py --variant original --data-root /path/to/ImageCLEF2024
python qwen/infer_qwen_lora.py --variant clean-paraphrase --data-root /path/to/ImageCLEF2024
```

Each command loads `runs/<variant>/best_adapter` by default. Use
`--adapter-dir /path/to/best_adapter` for a run stored elsewhere or pass the
directory of a specific `checkpoint-*`. The selected directory determines
which weights are loaded.

The script writes `predictions_test.csv` and `predictions_test.json` next to
the adapter directory, inside `inference_<adapter-directory-name>_test/`.
Each record contains `image_id`, `reference_caption`, `qwen_caption`, and
`status`. Use `--output-dir` to choose another destination for a new inference
run when predictions already exist.

## Prediction post-processing and evaluation

Install the evaluation dependencies in the Qwen environment and download the
WordNet data used by METEOR.

```bash
python -m pip install bert-score rouge-score nltk openpyxl
python -m nltk.downloader wordnet
```

```bash
python qwen/evaluate_qwen.py \
  --original-csv runs/original/inference_best_adapter_test/predictions_test.csv \
  --clean-paraphrase-csv runs/clean-paraphrase/inference_best_adapter_test/predictions_test.csv \
  --output-dir runs/qwen_evaluation
```

Pass either input on its own to evaluate one variant. The script reads the
reference captions embedded in the inference CSV. It preserves the raw
predictions in the per-sample workbook and trims generated chat continuations
starting at the first newline-prefixed `assistant` or `user` marker or
`<think>`.

The Excel reports list BERTScore F1 (with `microsoft/deberta-xlarge-mnli`),
ROUGE-L F1, BLEU-1, and METEOR in the order used in the article. The summary reports the
mean per-sample scores and the evaluated and excluded row counts. Failed
inferences, missing IDs or captions, and predictions that become empty
after cleaning are excluded. The script warns if the two variants are scored
on different image-reference pairs.

## Non-visual content analysis

### Detect non-visual content

Run the detector from the repository root before preparing the manual review workbook.

```bash
python -m analysis.non_visual_content \
  --original-references /path/to/test_2024_captions.json \
  --cleaned-references /path/to/test_clean_final.json \
  --original-predictions runs/qwen_evaluation/original_per_sample_metrics.xlsx \
  --clean-paraphrase-predictions runs/qwen_evaluation/clean-paraphrase_per_sample_metrics.xlsx
```

Each input is optional. Pass one file to analyze just that source. For example:

```bash
python -m analysis.non_visual_content \
  --original-predictions runs/qwen_evaluation/original_per_sample_metrics.xlsx
```

Comparisons are generated when their input pairs are available. For Table 3,
provide the original references and both prediction files. The cleaned
references are only needed for the reference-cleaning comparison.

The reference JSON files map image IDs to captions or lists of captions. The
prediction inputs may be Excel workbooks or CSV files containing `image_id`
and `qwen_caption_clean`. The default output directory is
`runs/non_visual_content/`, with a source-specific subdirectory for single-file runs. The output includes `nonvisual_per_sample_all.csv`,
category counts and rates, and `nonvisual_positive_cases_review.csv` for the
manual review script. For the comparison in Table 3, use the
`GT_original`, `Qwen_original`, and `Qwen_C&A` rows of the all-source summary.

### Prepare the manual review workbook

The manual review script takes the CSV of cases previously flagged by the
non-visual content detector. The CSV must include `source`, `image_id`, and
`matched_categories`. Any additional columns are copied to the workbook.

```bash
python analysis/prepare_non_visual_review.py \
  --input-csv runs/non_visual_content/nonvisual_positive_cases_review.csv \
  --output-xlsx runs/non_visual_content/manual_review_nonvisual.xlsx
```

The workbook includes all flagged Qwen C&P predictions, up to 20 flagged
Qwen original predictions per category before deduplication by image ID,
and up to 50 flagged cleaned reference captions. Sampling uses seed 42.
Each sheet includes `manual_classification` and `manual_comment` columns.

## Prediction behavior analysis

### Sample-level performance

Install Matplotlib with `python -m pip install matplotlib` to generate the figures.

Generate the paired BERTScore F1 transition analysis from the per-sample
evaluation files. Inputs can be CSV or Excel and must contain `image_id`,
`BERTScore_F1_clean`, and `clean_words`.

```bash
python -m analysis.prediction_behavior_analysis.sample_level_performance \
  --original-metrics runs/qwen_evaluation/original_per_sample_metrics.xlsx \
  --clean-paraphrase-metrics runs/qwen_evaluation/clean-paraphrase_per_sample_metrics.xlsx
```

The script pairs predictions by image ID and selects rows with Qwen-O
BERTScore F1 at or below 0.50. It writes the two-panel figure as PNG and PDF,
a summary workbook, and `difficult_samples_transitions.csv` to
`runs/sample_level_performance/` by default.

#### Non-visual content in low-baseline samples

The difficult-sample analysis applies the same detector to original reference
captions for samples with baseline BERTScore F1 at or below 0.50. It takes an
existing `difficult_samples_transitions.csv` with `image_id` and `outcome`.
The expected outcomes are `Improved and crossed >0.50`,
`Improved but remained <=0.50`, and `Not improved`.

```bash
python -m analysis.prediction_behavior_analysis.difficult_non_visual \
  --transitions-csv runs/sample_level_performance/difficult_samples_transitions.csv \
  --original-references /path/to/test_2024_captions.json
```

The default output directory is `runs/difficult_non_visual/`. It contains
CSV tables and an Excel workbook with group summaries, recovery rates by
non-visual content status, and per-sample results.

### Looping

Run the detector on one or both sets of postprocessed Qwen predictions.
Each input may be a CSV or an Excel workbook with `image_id` and
`qwen_caption_clean`.

```bash
python -m analysis.prediction_behavior_analysis.looping \
  --original-predictions runs/qwen_evaluation/original_per_sample_metrics.xlsx \
  --clean-paraphrase-predictions runs/qwen_evaluation/clean-paraphrase_per_sample_metrics.xlsx
```

A caption shows looping when the same 1- to 4-token block repeats immediately
at least twice. Detection lowercases the text and replaces punctuation with
spaces. The CSVs and Excel workbook report per-caption detections, frequency,
severity, and positive cases. Relative reduction is reported when both
variants are supplied. The default output directory is `runs/looping/`,
with a source-specific subdirectory when analyzing one variant.

### Modality consistency

Use the original reference JSON and one or both postprocessed prediction
files (CSV or Excel with `image_id` and `qwen_caption_clean`).

```bash
python -m analysis.prediction_behavior_analysis.modality_consistency \
  --original-references /path/to/test_2024_captions.json \
  --original-predictions runs/qwen_evaluation/original_per_sample_metrics.xlsx \
  --clean-paraphrase-predictions runs/qwen_evaluation/clean-paraphrase_per_sample_metrics.xlsx
```

For the paper's paired comparison, provide both variants. The strict subset
includes shared image IDs with a detected modality in the reference and
both predictions. Consistency requires at least one shared modality. A
second summary includes all paired samples with a detectable reference
modality and counts matches, mismatches, and omissions. If only one
variant is provided, the strict subset uses the reference and that variant.
The default report directory is `runs/modality_consistency/`, with a
source-specific subdirectory for single-variant runs.

### Conversational leakage

Analyze one or both raw inference CSVs before caption post-processing.
The script counts each eligible prediction once if it contains
a newline followed by `assistant` or `user`, or contains `<think>`.
The denominator includes nonmissing captions from successful inferences.

```bash
python -m analysis.prediction_behavior_analysis.conversational_leakage \
  --original-predictions runs/original/inference_best_adapter_test/predictions_test.csv \
  --clean-paraphrase-predictions runs/clean-paraphrase/inference_best_adapter_test/predictions_test.csv \
  --output-csv runs/conversational_leakage/summary.csv
```

Omit either prediction flag to analyze a single variant. The summary is
printed to the terminal and saved to CSV when `--output-csv` is supplied.

## Citation

If you use this code or the preprocessing prompts in your research, please cite the accompanying manuscript.

### Plain text

Rascón-Cervantes, S., Ramirez-Alonso, G., Lopez-Santillan, R., Lopez-Flores, D. R., & López-Monroy, A. P. (2026). *Textual Preprocessing for Qwen-Based Medical Image Captioning* [Manuscript under review].

### BibTeX (LaTeX)

```bibtex
@unpublished{rasconcervantes2026textual,
  author = {Rasc{\'o}n-Cervantes, Sebasti{\'a}n and Ramirez-Alonso, Graciela and
            Lopez-Santillan, Roberto and Lopez-Flores, David R. and
            L{\'o}pez-Monroy, Adri{\'a}n Pastor},
  title  = {Textual Preprocessing for {Qwen}-Based Medical Image Captioning},
  year   = {2026},
  note   = {Manuscript under review}
}
```
