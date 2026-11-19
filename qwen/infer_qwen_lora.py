"""Generate ImageCLEFmedical 2024 test captions with a trained Qwen LoRA adapter."""

import argparse
import json
from pathlib import Path

# ============================================================
# 1. CONFIGURATION AND PATH VALIDATION
# ============================================================

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=("original", "clean-paraphrase"), required=True)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument(
    "--adapter-dir",
    type=Path,
    help="LoRA adapter directory (default: runs/<variant>/best_adapter).",
)
parser.add_argument("--output-dir", type=Path, help="Directory for CSV and JSON predictions.")
args = parser.parse_args()

MODEL_ID = "OpenMed/Qwen3.5-2B-MedVL"
TEST_CSV = args.data_root / "test_2024_captions.csv"
TEST_IMAGES_DIR = args.data_root / "test"
IMAGE_COL = "ID"
CAPTION_COL = "Caption"

CHECKPOINT_DIR = args.adapter_dir or Path("runs") / args.variant / "best_adapter"
OUTPUT_DIR = args.output_dir or CHECKPOINT_DIR.parent / f"inference_{CHECKPOINT_DIR.name}_test"
OUTPUT_CSV = OUTPUT_DIR / "predictions_test.csv"
OUTPUT_JSON = OUTPUT_DIR / "predictions_test.json"

if not (CHECKPOINT_DIR / "adapter_config.json").is_file():
    parser.error(f"Missing adapter_config.json in {CHECKPOINT_DIR}")
if not any((CHECKPOINT_DIR / name).is_file()
           for name in ("adapter_model.safetensors", "adapter_model.bin")):
    parser.error(f"LoRA weights were not found in {CHECKPOINT_DIR}")
if not TEST_CSV.is_file():
    parser.error(f"Test CSV was not found: {TEST_CSV}")
if not TEST_IMAGES_DIR.is_dir():
    parser.error(f"Test image directory was not found: {TEST_IMAGES_DIR}")
if OUTPUT_CSV.exists() or OUTPUT_JSON.exists():
    parser.error(f"Predictions already exist in {OUTPUT_DIR}. Choose another --output-dir.")

# Load model dependencies after parsing arguments so --help works without them.
import pandas as pd
from PIL import Image, ImageFile

import torch
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

from peft import PeftModel


ImageFile.LOAD_TRUNCATED_IMAGES = True


DEVICE = torch.device("cuda:0")


# ============================================================
# 2. GENERATION SETTINGS
# ============================================================

MAX_NEW_TOKENS = 80


PROMPT = (
    "Generate a concise radiology caption that directly describes the "
    "principal findings visible in the medical image. "
    "When evident, also mention the imaging modality and anatomical region. "
    "Do not explain the imaging modality, discuss its typical use, "
    "or provide general medical information. "
    "Output only the caption."
)


# ============================================================
# 3. UTILITIES
# ============================================================

def get_image_path(
    image_id,
    images_dir,
):

    image_id = str(
        image_id
    ).strip()

    direct = (
        images_dir /
        image_id
    )

    if direct.exists():
        return direct

    for extension in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    ]:

        candidate = (
            images_dir /
            f"{image_id}{extension}"
        )

        if candidate.exists():
            return candidate

    return None


def load_image(
    image_path,
):

    with Image.open(
        image_path
    ) as img:

        return (
            img
            .convert("RGB")
            .copy()
        )


# ============================================================
# 4. GPU
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is unavailable."
    )


print(
    "\nGPU:",
    torch.cuda.get_device_name(0)
)


# ============================================================
# 5. PROCESSOR
# ============================================================

print(
    "\nLoading the processor..."
)


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID
    )
)


# Use the same image processor settings as training.

processor.image_processor.size = {

    "shortest_edge":
        256 * 256,

    "longest_edge":
        1024 * 1024,
}


print(
    "Image processor settings:"
)

print(
    processor.image_processor.size
)


# ============================================================
# 6. BASE MODEL
# ============================================================

print(
    "\nLoading the base model..."
)


base_model = (
    AutoModelForImageTextToText
    .from_pretrained(

        MODEL_ID,

        torch_dtype=
            torch.bfloat16,

        device_map=None,
    )
)


base_model.to(
    DEVICE
)


# ============================================================
# 7. LoRA
# ============================================================

print(
    "\nLoading the LoRA adapter:"
)

print(
    CHECKPOINT_DIR
)


model = (
    PeftModel
    .from_pretrained(

        base_model,

        CHECKPOINT_DIR,

        is_trainable=False,
    )
)


model.eval()


if hasattr(
    model.config,
    "use_cache"
):

    model.config.use_cache = True


# ============================================================
# 8. TEST
# ============================================================

df = pd.read_csv(
    TEST_CSV
)


df = (

    df

    .dropna(
        subset=[
            IMAGE_COL,
            CAPTION_COL,
        ]
    )

    .reset_index(
        drop=True
    )
)


if df.empty:
    raise ValueError(f"No test rows with both an ID and a caption in {TEST_CSV}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Adapter variant: {args.variant}")

print(
    f"\nTest samples: "
    f"{len(df):,}"
)


# ============================================================
# 9. INFERENCE
# ============================================================

results = []


progress_bar = tqdm(
    df.iterrows(),
    total=len(df),
    desc="Inference",
)


with torch.inference_mode():

    for idx, row in progress_bar:

        image_id = str(
            row[IMAGE_COL]
        ).strip()

        reference_caption = str(
            row[CAPTION_COL]
        ).strip()


        image_path = get_image_path(

            image_id,

            TEST_IMAGES_DIR,
        )


        # ----------------------------------------------------
        # MISSING IMAGE
        # ----------------------------------------------------

        if image_path is None:

            results.append({

                "image_id":
                    image_id,

                "reference_caption":
                    reference_caption,

                "qwen_caption":
                    "",

                "status":
                    "image_not_found",

            })

            continue


        try:

            image = load_image(
                image_path
            )


            # ------------------------------------------------
            # QWEN INPUT MESSAGE
            # ------------------------------------------------

            messages = [

                {

                    "role": "user",

                    "content": [

                        {
                            "type": "image"
                        },

                        {
                            "type": "text",
                            "text": PROMPT
                        },

                    ],
                }

            ]


            text = (
                processor
                .apply_chat_template(

                    messages,

                    tokenize=False,

                    add_generation_prompt=True,
                )
            )


            # ------------------------------------------------
            # PROCESSOR
            # ------------------------------------------------

            inputs = processor(

                text=[text],

                images=[image],

                padding=True,

                return_tensors="pt",
            )


            inputs = {

                key:
                    value.to(
                        DEVICE
                    )

                if torch.is_tensor(value)

                else value

                for key, value
                in inputs.items()
            }


            input_length = (
                inputs[
                    "input_ids"
                ]
                .shape[1]
            )


            # ------------------------------------------------
            # GENERATION
            # ------------------------------------------------

            with torch.autocast(

                device_type="cuda",

                dtype=torch.bfloat16,

            ):

                generated_ids = (
                    model.generate(

                        **inputs,

                        max_new_tokens=
                            MAX_NEW_TOKENS,

                        do_sample=False,

                    )
                )


            # ------------------------------------------------
            # REMOVE INPUT TOKENS
            # ------------------------------------------------

            generated_tokens = (

                generated_ids[
                    :,
                    input_length:
                ]
            )


            prediction = (
                processor
                .batch_decode(

                    generated_tokens,

                    skip_special_tokens=True,

                )[0]

                .strip()
            )


            # ------------------------------------------------
            # PREDICTION RECORD
            # ------------------------------------------------

            results.append({

                "image_id":
                    image_id,

                "reference_caption":
                    reference_caption,

                "qwen_caption":
                    prediction,

                "status":
                    "ok",

            })


        except Exception as e:

            results.append({

                "image_id":
                    image_id,

                "reference_caption":
                    reference_caption,

                "qwen_caption":
                    "",

                "status":
                    (
                        f"error: "
                        f"{type(e).__name__}: "
                        f"{e}"
                    ),

            })


        # ----------------------------------------------------
        # SAVE INTERMEDIATE RESULTS
        # every 100 samples
        # ----------------------------------------------------

        if (
            len(results) % 100
            == 0
        ):

            pd.DataFrame(
                results
            ).to_csv(

                OUTPUT_CSV,

                index=False,
            )


# ============================================================
# 10. SAVE FINAL CSV
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(

    OUTPUT_CSV,

    index=False,
)


# ============================================================
# 11. SAVE FINAL JSON
# ============================================================

with open(

    OUTPUT_JSON,

    "w",

    encoding="utf-8",

) as f:

    json.dump(

        results,

        f,

        indent=2,

        ensure_ascii=False,
    )


# ============================================================
# 12. SUMMARY
# ============================================================

num_ok = (
    results_df[
        "status"
    ]
    .eq("ok")
    .sum()
)


num_errors = (
    len(results_df)
    - num_ok
)


print(
    "\n" +
    "=" * 70
)

print(
    "INFERENCE COMPLETE"
)

print(
    "=" * 70
)


print(
    f"\nTotal: "
    f"{len(results_df):,}"
)

print(
    f"Successful: "
    f"{num_ok:,}"
)

print(
    f"Failed: "
    f"{num_errors:,}"
)


print(
    "\nCSV:"
)

print(
    OUTPUT_CSV
)


print(
    "\nJSON:"
)

print(
    OUTPUT_JSON
)
