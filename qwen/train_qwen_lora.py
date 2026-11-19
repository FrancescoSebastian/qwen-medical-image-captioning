# ============================================================
# Qwen3.5-2B-MedVL + LoRA
# FULL TRAINING - MEDICAL IMAGE CAPTIONING
# ============================================================
#
# Strategy:
#   Vision encoder        -> frozen
#   Base LLM       -> frozen
#   LoRA on the LLM        -> trainable
#
# Full training dataset.
#
# Features:
#   - multimodal batching
#   - BF16
#   - gradient checkpointing
#   - gradient accumulation
#   - cosine scheduler
#   - resumable checkpoints
#   - optimizer/scheduler state
#   - qualitative validation each epoch
#   - loss logging
#
# ============================================================


# ============================================================
# 0. ENVIRONMENT VARIABLES
# ============================================================

import argparse
import hashlib
import os
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Fine-tune Qwen3.5-2B-MedVL on ImageCLEFmedical 2024 captions."
)
parser.add_argument("--variant", choices=("original", "clean-paraphrase"), required=True)
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--train-csv", required=True, type=Path)
parser.add_argument("--output-dir", type=Path)
parser.add_argument(
    "--resume-legacy",
    action="store_true",
    help="Use an existing run directory created before dataset tracking was added.",
)
args = parser.parse_args()

output_dir = args.output_dir or Path("runs") / args.variant
if args.resume_legacy and args.output_dir is None:
    parser.error("--resume-legacy requires an explicit --output-dir.")


os.environ["CC"] = "/usr/bin/gcc"
os.environ["CXX"] = "/usr/bin/g++"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ============================================================
# 1. IMPORTS
# ============================================================

import gc
import json
import math
import random
import shutil

import pandas as pd

from PIL import Image, ImageFile

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
)

import time
from datetime import timedelta

execution_start = time.perf_counter()

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 2. RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. GENERAL CONFIGURATION
# ============================================================

MODEL_ID = "OpenMed/Qwen3.5-2B-MedVL"


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

DATA_ROOT = args.data_root
TRAIN_IMAGES_DIR = DATA_ROOT / "train"
VALID_IMAGES_DIR = DATA_ROOT / "valid"
VALID_CSV = DATA_ROOT / "valid_2024_captions.csv"
TRAIN_CSV = args.train_csv


# ------------------------------------------------------------
# CSV COLUMNS
# ------------------------------------------------------------

IMAGE_COL = "ID"
CAPTION_COL = "Caption"


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = output_dir


# ============================================================
# 4. HYPERPARAMETERS
# ============================================================

NUM_EPOCHS = 3


# ------------------------------------------------------------
# BATCH
# ------------------------------------------------------------

BATCH_SIZE = 2#1

GRADIENT_ACCUMULATION_STEPS = 8

# Effective batch size:
#
# 2 x 8 = 16


# ------------------------------------------------------------
# OPTIMIZATION
# ------------------------------------------------------------

LEARNING_RATE = 5e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.05

MAX_GRAD_NORM = 1.0


# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05


LORA_SUFFIXES = (

    "q_proj",

    "k_proj",

    "v_proj",

    "o_proj",

    "gate_proj",

    "up_proj",

    "down_proj",
)


# ------------------------------------------------------------
# CHECKPOINTS
# ------------------------------------------------------------

SAVE_EVERY_OPTIMIZER_STEPS = 200


# After an interruption:
#
# True = automatically find the latest usable checkpoint.
#
AUTO_RESUME = True


# ------------------------------------------------------------
# QUALITATIVE VALIDATION
# ------------------------------------------------------------

NUM_VALID_EXAMPLES = 10

VALIDATION_SAMPLES = 1000

VALID_BATCH_SIZE = 1

BEST_ADAPTER_DIR = (
    OUTPUT_DIR /
    "best_adapter"
)

EPOCH_METRICS_CSV = (
    OUTPUT_DIR /
    "epoch_metrics.csv"
)

MAX_NEW_TOKENS = 64


# ------------------------------------------------------------
# DATALOADER
# ------------------------------------------------------------

NUM_WORKERS = 0

PIN_MEMORY =False


# ============================================================
# 5. PROMPT
# ============================================================

PROMPT = (
    "Generate a concise radiology caption that directly describes the "
    "principal findings visible in the medical image. "
    "When evident, also mention the imaging modality and anatomical region. "
    "Do not explain the imaging modality, discuss its typical use, "
    "or provide general medical information. "
    "Output only the caption."
)


# ============================================================
# 6. UTILITIES
# ============================================================

def check_run_data():
    """Keep automatic resumption tied to the same caption data."""
    run_data = {"variant": args.variant}
    for split, csv_path in (("train", TRAIN_CSV), ("valid", VALID_CSV)):
        with csv_path.open("rb") as csv_file:
            run_data[f"{split}_csv_sha256"] = hashlib.file_digest(
                csv_file, "sha256"
            ).hexdigest()

    manifest = OUTPUT_DIR / "caption_data.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != run_data:
            raise ValueError(
                f"{OUTPUT_DIR} belongs to a different dataset variant or CSV. "
                "Use the original inputs or choose another --output-dir."
            )
        return

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not args.resume_legacy:
        raise ValueError(
            f"{OUTPUT_DIR} is nonempty and has no caption_data.json. "
            "Choose a new --output-dir or use --resume-legacy with the original inputs."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as manifest_file:
        json.dump(run_data, manifest_file, indent=2)
        manifest_file.write("\n")


# ------------------------------------------------------------

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


# ------------------------------------------------------------

def load_image(
    image_path,
):

    with Image.open(
        image_path
    ) as img:

        image = (
            img
            .convert("RGB")
            .copy()
        )


    return image


# ============================================================
# 7. DATASET
# ============================================================

class MedicalCaptionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        images_dir,
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.images_dir = images_dir


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[idx]


        image_id = str(
            row[IMAGE_COL]
        ).strip()


        caption = str(
            row[CAPTION_COL]
        ).strip()


        image_path = get_image_path(

            image_id,

            self.images_dir,
        )


        return {

            "image_id":
                image_id,

            "caption":
                caption,

            "image_path":
                image_path,
        }


# ============================================================
# 8. MULTIMODAL COLLATOR
# ============================================================

class QwenCaptionCollator:

    def __init__(
        self,
        processor,
    ):

        self.processor = processor


    def __call__(
        self,
        batch,
    ):

        images = []

        captions = []

        image_ids = []


        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        for sample in batch:

            image_path = sample[
                "image_path"
            ]


            if image_path is None:

                continue


            try:

                image = load_image(
                    image_path
                )


            except Exception as e:

                print(
                    f"\n[IMAGEN OMITIDA] "
                    f"{sample['image_id']}"
                )

                print(
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                continue


            images.append(
                image
            )

            captions.append(
                sample["caption"]
            )

            image_ids.append(
                sample["image_id"]
            )


        # Skip the batch if every sample failed
        if len(images) == 0:

            return None


        # ----------------------------------------------------
        # Build conversations
        # ----------------------------------------------------

        full_texts = []

        prompt_texts = []


        for caption in captions:


            # =================================================
            # USER + ASSISTANT
            # =================================================

            full_messages = [

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
                },

                {
                    "role": "assistant",

                    "content": [

                        {
                            "type": "text",
                            "text": caption
                        }

                    ],
                },
            ]


            # =================================================
            # USER ONLY
            # =================================================

            prompt_messages = [

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


            full_text = (
                self.processor
                .apply_chat_template(

                    full_messages,

                    tokenize=False,

                    add_generation_prompt=False,
                )
            )


            prompt_text = (
                self.processor
                .apply_chat_template(

                    prompt_messages,

                    tokenize=False,

                    add_generation_prompt=True,
                )
            )


            full_texts.append(
                full_text
            )

            prompt_texts.append(
                prompt_text
            )


        # ----------------------------------------------------
        # Process the full batch
        # ----------------------------------------------------

        full_inputs = self.processor(

            text=full_texts,

            images=images,

            padding=True,

            return_tensors="pt",
        )


        # ----------------------------------------------------
        # Process prompts to determine their lengths
        # ----------------------------------------------------

        prompt_inputs = self.processor(

            text=prompt_texts,

            images=images,

            padding=True,

            return_tensors="pt",
        )


        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        labels = (
            full_inputs["input_ids"]
            .clone()
        )


        # Ignore padding
        labels[
            full_inputs["attention_mask"] == 0
        ] = -100


        # ----------------------------------------------------
        # Mask each prompt separately
        # ----------------------------------------------------

        prompt_lengths = (

            prompt_inputs[
                "attention_mask"
            ]

            .sum(dim=1)
        )


        for i, prompt_length in enumerate(
            prompt_lengths
        ):

            prompt_length = int(
                prompt_length.item()
            )


            labels[
                i,
                :prompt_length
            ] = -100


        full_inputs[
            "labels"
        ] = labels


        # ----------------------------------------------------
        # Remove unused forward inputs
        # ----------------------------------------------------

        full_inputs.pop(
            "token_type_ids",
            None
        )


        full_inputs[
            "image_ids"
        ] = image_ids


        return full_inputs


# ============================================================
# 9. MOVE BATCH TO GPU
# ============================================================

def move_to_device(
    batch,
    device,
):

    result = {}


    for key, value in batch.items():


        if key == "image_ids":

            result[key] = value

            continue


        if torch.is_tensor(
            value
        ):

            result[key] = (
                value
                .to(
                    device,
                    non_blocking=True,
                )
            )


        else:

            result[key] = value


    return result


# ============================================================
# 10. FIND LORA TARGET MODULES
# ============================================================

def find_language_lora_modules(
    model,
):

    target_modules = []

    excluded_visual = []


    vision_keywords = (

        "visual",

        "vision",

        "image",
    )


    for name, module in (
        model.named_modules()
    ):


        if not isinstance(
            module,
            torch.nn.Linear
        ):

            continue


        if not name.endswith(
            LORA_SUFFIXES
        ):

            continue


        lower_name = (
            name.lower()
        )


        # ----------------------------------------------------
        # Exclude the vision encoder from LoRA
        # ----------------------------------------------------

        if any(

            keyword in lower_name

            for keyword
            in vision_keywords

        ):

            excluded_visual.append(
                name
            )

            continue


        target_modules.append(
            name
        )


    return (
        target_modules,
        excluded_visual,
    )


# ============================================================
# 11. FIND THE LATEST CHECKPOINT
# ============================================================

def find_latest_checkpoint():
    """Return the latest readable checkpoint."""
    checkpoints = sorted(
        (p for p in OUTPUT_DIR.glob("checkpoint-*")
         if p.is_dir() and p.name.split("-")[-1].isdigit()),
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )
    for checkpoint in checkpoints:
        state_file = checkpoint / "training_state.pt"
        adapter_file = checkpoint / "adapter_model.safetensors"
        adapter_bin = checkpoint / "adapter_model.bin"
        if not (state_file.is_file() and
                (adapter_file.is_file() or adapter_bin.is_file()) and
                (checkpoint / "adapter_config.json").is_file()):
            print(f"[INCOMPLETE CHECKPOINT] {checkpoint.name}")
            continue
        try:
            # Load checkpoints only from trusted sources.
            state = torch.load(state_file, map_location="cpu", weights_only=False)
            required = ("epoch_idx", "batch_idx", "global_micro_step",
                        "optimizer_step", "optimizer", "scheduler")
            if any(key not in state for key in required):
                raise ValueError("Missing training state fields")
            if int(state["optimizer_step"]) != int(checkpoint.name.split("-")[-1]):
                raise ValueError("The optimizer step does not match the directory name")
            del state
            if adapter_file.is_file():
                from safetensors import safe_open
                with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
                    if not list(f.keys()):
                        raise ValueError("Empty adapter")
            print(f"[VERIFIED CHECKPOINT] {checkpoint.name}")
            return checkpoint
        except Exception as exc:
            print(f"[UNUSABLE CHECKPOINT] {checkpoint.name}: {exc}")
    if checkpoints:
        raise RuntimeError("No usable checkpoint was found. "
                           "Stopped to preserve the existing run.")
    return None


# ============================================================
# 12. ATOMIC CHECKPOINT SAVING 
# ============================================================

def save_checkpoint(
    model, processor, optimizer, scheduler, epoch_idx, batch_idx,
    global_micro_step, optimizer_step,
):
    checkpoint_dir = OUTPUT_DIR / f"checkpoint-{optimizer_step:07d}"
    temporary_dir = OUTPUT_DIR / f".checkpoint-{optimizer_step:07d}.tmp"

    # Preserve existing checkpoints.
    if checkpoint_dir.exists():
        print(f"[CHECKPOINT ALREADY EXISTS] {checkpoint_dir}")
        return
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    try:
        model.save_pretrained(temporary_dir)
        processor.save_pretrained(temporary_dir)
        training_state = {
            "epoch_idx": epoch_idx,
            "batch_idx": batch_idx,
            "global_micro_step": global_micro_step,
            "optimizer_step": optimizer_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }
        torch.save(training_state, temporary_dir / "training_state.pt")
        if not (temporary_dir / "adapter_config.json").is_file():
            raise RuntimeError("adapter_config.json is missing")
        if not ((temporary_dir / "adapter_model.safetensors").is_file() or
                (temporary_dir / "adapter_model.bin").is_file()):
            raise RuntimeError("LoRA weights are missing")
        # Rename atomically on the same filesystem once all files are ready.
        temporary_dir.rename(checkpoint_dir)
        print(f"\nCheckpoint saved: {checkpoint_dir}\n")
    except Exception:
        print(f"[CHECKPOINT SAVE ERROR] Temporary files: {temporary_dir}")
        raise


# ============================================================
# 13. GENERATE CAPTION
# ============================================================

def generate_caption(
    model,
    processor,
    image,
    device,
):

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


    inputs = processor(

        text=[text],

        images=[image],

        return_tensors="pt",
    )


    inputs.pop(
        "token_type_ids",
        None
    )


    inputs = {

        key:
            value.to(device)

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


    with torch.inference_mode():

        outputs = model.generate(

            **inputs,

            max_new_tokens=
                MAX_NEW_TOKENS,

            do_sample=False,
        )


    generated = (

        outputs[
            0,
            input_length:
        ]
    )


    caption = processor.decode(

        generated,

        skip_special_tokens=True,

        clean_up_tokenization_spaces=False,
    ).strip()


    return caption

# ============================================================
# VALIDATION LOSS
# ============================================================

def calculate_validation_loss(
    model,
    valid_loader,
    device,
):

    """
    Compute mean loss on the validation subset.

    Run without backpropagation or weight updates.
    """

    model.eval()

    total_loss = 0.0
    valid_batches = 0

    progress_bar = tqdm(
        valid_loader,
        desc="Validation loss",
        leave=False,
    )

    with torch.inference_mode():

        for batch in progress_bar:

            if batch is None:
                continue

            # Keep image IDs for debugging only
            image_ids = batch.pop(
                "image_ids",
                None,
            )

            batch = move_to_device(
                batch,
                device,
            )

            try:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    outputs = model(
                        **batch
                    )

                    loss = outputs.loss

                if not torch.isfinite(loss):

                    print(
                        "\nNon-finite validation loss."
                    )

                    if image_ids is not None:
                        print(f"IDs: {image_ids}")

                    continue

                loss_value = (
                    loss
                    .detach()
                    .float()
                    .item()
                )

                total_loss += loss_value
                valid_batches += 1

                progress_bar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    avg=f"{total_loss / valid_batches:.4f}",
                )

                del batch, outputs, loss

            except Exception as e:

                if isinstance(e, (RuntimeError, torch.cuda.OutOfMemoryError)):
                    # Stop on CUDA runtime errors because the context may be invalid.
                    raise

                print(
                    "\n[VALIDATION LOSS ERROR]"
                )

                if image_ids is not None:
                    print(f"IDs: {image_ids}")

                print(
                    f"{type(e).__name__}: {e}"
                )

                continue

    model.train()

    if valid_batches == 0:

        return float("nan")

    return total_loss / valid_batches

# ============================================================
# 14. QUALITATIVE VALIDATION
# ============================================================

def qualitative_validation(
    model,
    processor,
    valid_df,
    epoch_number,
    device,
):

    print(
        "\n" +
        "=" * 75
    )

    print(
        f"QUALITATIVE VALIDATION "
        f"- EPOCH {epoch_number}"
    )

    print(
        "=" * 75
    )


    model.eval()


    results = []


    num_examples = min(

        NUM_VALID_EXAMPLES,

        len(valid_df),
    )


    for idx in range(
        num_examples
    ):

        row = (
            valid_df
            .iloc[idx]
        )


        image_id = str(
            row[IMAGE_COL]
        ).strip()


        reference = str(
            row[CAPTION_COL]
        ).strip()


        image_path = get_image_path(

            image_id,

            VALID_IMAGES_DIR,
        )


        if image_path is None:

            continue


        try:

            image = load_image(
                image_path
            )


            prediction = generate_caption(

                model,

                processor,

                image,

                device,
            )


            print(
                "\n" +
                "-" * 75
            )

            print(
                f"IMAGE:\n"
                f"{image_id}"
            )

            print(
                f"\nREFERENCE:\n"
                f"{reference}"
            )

            print(
                f"\nPREDICTION:\n"
                f"{prediction}"
            )


            results.append(

                {

                    "image_id":
                        image_id,

                    "reference_caption":
                        reference,

                    "qwen_lora_caption":
                        prediction,
                }

            )


        except Exception as e:

            print(
                f"\n[VALID ERROR] "
                f"{image_id}: {e}"
            )


    output_file = (

        OUTPUT_DIR /

        f"validation_epoch_"
        f"{epoch_number:02d}.json"
    )


    with open(

        output_file,

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            results,

            f,

            ensure_ascii=False,

            indent=2,
        )


    model.train()


# ============================================================
# 15. HARDWARE
# ============================================================

print(
    "=" * 75
)

print(
    "Qwen3.5-2B-MedVL + LLM LoRA"
)

print(
    "FULL MEDICAL IMAGE CAPTIONING TRAINING"
)

print(
    "=" * 75
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is unavailable."
    )


DEVICE = torch.device(
    "cuda:0"
)


print(
    f"\nPyTorch: "
    f"{torch.__version__}"
)

print(
    f"CUDA: "
    f"{torch.version.cuda}"
)

print(
    f"GPU: "
    f"{torch.cuda.get_device_name(0)}"
)


# RTX 5090
torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# 16. DATA
# ============================================================

print(f"Caption variant: {args.variant}")
print(f"Training CSV: {TRAIN_CSV}")
print(f"Validation CSV: {VALID_CSV}")
print(f"Output directory: {OUTPUT_DIR}")


train_df = pd.read_csv(
    TRAIN_CSV
)


valid_df = pd.read_csv(
    VALID_CSV
)


# ------------------------------------------------------------
# Drop rows with missing IDs or captions
# ------------------------------------------------------------

train_df = (

    train_df

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


valid_df = (

    valid_df

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


print(
    f"\nTrain: "
    f"{len(train_df):,}"
)

print(
    f"Valid: "
    f"{len(valid_df):,}"
)


check_run_data()


# ============================================================
# 17. PROCESSOR
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
processor.image_processor.size = {
    "shortest_edge": 256 * 256,
    "longest_edge": 1024 * 1024,
}

print("\nVision configuration:")
print(processor.image_processor.size)
# ============================================================
# 18. BASE MODEL
# ============================================================

print(
    "\nLoading the model..."
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


if hasattr(
    base_model.config,
    "use_cache"
):

    base_model.config.use_cache = False


# ============================================================
# 19. FREEZE BASE MODEL PARAMETERS
# ============================================================

for parameter in (
    base_model.parameters()
):

    parameter.requires_grad = False


# ============================================================
# 20. RESUME OR INITIALIZE LORA
# ============================================================

resume_checkpoint = None


if AUTO_RESUME:

    resume_checkpoint = (
        find_latest_checkpoint()
    )


# ------------------------------------------------------------
# RESUME
# ------------------------------------------------------------

if resume_checkpoint is not None:


    print(
        f"\nResuming from:\n"
        f"{resume_checkpoint}"
    )


    model = (
        PeftModel
        .from_pretrained(

            base_model,

            resume_checkpoint,

            is_trainable=True,
        )
    )


# ------------------------------------------------------------
# NEW TRAINING RUN
# ------------------------------------------------------------

else:


    print(
        "\nCreating a new LoRA adapter..."
    )


    (
        lora_targets,
        excluded_visual,
    ) = find_language_lora_modules(
        base_model
    )


    print(
        f"\nLLM targets: "
        f"{len(lora_targets)}"
    )

    print(
        f"Excluded vision targets: "
        f"{len(excluded_visual)}"
    )


    print(
        "\nExample LLM targets:"
    )


    for name in (
        lora_targets[:15]
    ):

        print(
            f"  {name}"
        )


    print(
        "\nExample excluded vision targets:"
    )


    for name in (
        excluded_visual[:10]
    ):

        print(
            f"  {name}"
        )


    if len(
        lora_targets
    ) == 0:

        raise RuntimeError(
            "No LoRA targets were found."
        )


    lora_config = LoraConfig(

        r=LORA_R,

        lora_alpha=
            LORA_ALPHA,

        lora_dropout=
            LORA_DROPOUT,

        target_modules=
            lora_targets,

        bias="none",

        task_type=
            TaskType.CAUSAL_LM,
    )


    model = get_peft_model(

        base_model,

        lora_config,
    )


# ============================================================
# 21. GRADIENT CHECKPOINTING
# ============================================================

if hasattr(
    model,
    "gradient_checkpointing_enable"
):

    model.gradient_checkpointing_enable()


if hasattr(
    model,
    "enable_input_require_grads"
):

    model.enable_input_require_grads()


# ============================================================
# 22. TRAINABLE PARAMETERS
# ============================================================

print(
    "\n" +
    "=" * 75
)

print(
    "TRAINABLE PARAMETERS"
)

print(
    "=" * 75
)


model.print_trainable_parameters()


trainable_params = sum(

    p.numel()

    for p
    in model.parameters()

    if p.requires_grad
)


total_params = sum(

    p.numel()

    for p
    in model.parameters()
)


print(
    f"\nTrainable: "
    f"{trainable_params:,}"
)

print(
    f"Total: "
    f"{total_params:,}"
)

print(
    f"Trainable %: "
    f"{100 * trainable_params / total_params:.4f}%"
)


# ============================================================
# 23. DATASET
# ============================================================

train_dataset = MedicalCaptionDataset(

    train_df,

    TRAIN_IMAGES_DIR,
)


collator = QwenCaptionCollator(
    processor
)

# ============================================================
# FIXED VALIDATION SUBSET
# ============================================================

num_validation_samples = min(
    VALIDATION_SAMPLES,
    len(valid_df),
)

validation_loss_df = (
    valid_df
    .sample(
        n=num_validation_samples,
        random_state=SEED,
    )
    .reset_index(drop=True)
)

validation_dataset = MedicalCaptionDataset(
    validation_loss_df,
    VALID_IMAGES_DIR,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    collate_fn=collator,
    drop_last=False,
)

print(
    f"\nValidation examples: "
    f"{len(validation_dataset):,}"
)

# ============================================================
# 24. OPTIMIZER
# ============================================================

optimizer = AdamW(

    [

        p

        for p
        in model.parameters()

        if p.requires_grad

    ],

    lr=
        LEARNING_RATE,

    weight_decay=
        WEIGHT_DECAY,
)


# ============================================================
# 25. STEP COUNTS
# ============================================================

batches_per_epoch = math.ceil(

    len(train_dataset) /
    BATCH_SIZE
)


optimizer_steps_per_epoch = math.ceil(

    batches_per_epoch /
    GRADIENT_ACCUMULATION_STEPS
)


total_optimizer_steps = (

    optimizer_steps_per_epoch *
    NUM_EPOCHS
)


warmup_steps = max(

    1,

    int(
        total_optimizer_steps *
        WARMUP_RATIO
    )
)


scheduler = get_cosine_schedule_with_warmup(

    optimizer,

    num_warmup_steps=
        warmup_steps,

    num_training_steps=
        total_optimizer_steps,
)


# ============================================================
# 26. INITIAL TRAINING STATE
# ============================================================

start_epoch = 0

resume_batch_idx = -1

global_micro_step = 0

optimizer_step = 0


# ============================================================
# 27. RESTORE TRAINING STATE
# ============================================================

if resume_checkpoint is not None:


    state_path = (

        resume_checkpoint /
        "training_state.pt"
    )


    state = torch.load(

        state_path,

        map_location="cpu",

        weights_only=False,
    )


    optimizer.load_state_dict(
        state["optimizer"]
    )


    scheduler.load_state_dict(
        state["scheduler"]
    )


    start_epoch = (
        state["epoch_idx"]
    )


    resume_batch_idx = (
        state["batch_idx"]
    )


    global_micro_step = (
        state["global_micro_step"]
    )


    optimizer_step = (
        state["optimizer_step"]
    )


    if (
        "torch_rng_state"
        in state
    ):

        torch.set_rng_state(
            state[
                "torch_rng_state"
            ]
        )


    if (
        "cuda_rng_state"
        in state
    ):

        torch.cuda.set_rng_state_all(
            state[
                "cuda_rng_state"
            ]
        )


    print(
        f"\nResume epoch: "
        f"{start_epoch + 1}"
    )

    print(
        f"Resume batch: "
        f"{resume_batch_idx + 1}"
    )

    print(
        f"Optimizer step: "
        f"{optimizer_step}"
    )


# ============================================================
# 28. CONFIGURATION
# ============================================================

print(
    "\n" +
    "=" * 75
)

print(
    "TRAINING CONFIGURATION"
)

print(
    "=" * 75
)


print(
    f"\nTrain examples: "
    f"{len(train_dataset):,}"
)

print(
    f"Epochs: "
    f"{NUM_EPOCHS}"
)

print(
    f"Physical batch: "
    f"{BATCH_SIZE}"
)

print(
    f"Gradient accumulation: "
    f"{GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"Effective batch: "
    f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"Batches/epoch: "
    f"{batches_per_epoch:,}"
)

print(
    f"Optimizer steps/epoch: "
    f"{optimizer_steps_per_epoch:,}"
)

print(
    f"Total optimizer steps: "
    f"{total_optimizer_steps:,}"
)

print(
    f"Warmup steps: "
    f"{warmup_steps:,}"
)

print(
    f"LR: "
    f"{LEARNING_RATE}"
)

print(
    f"LoRA r: "
    f"{LORA_R}"
)

print(
    f"LoRA alpha: "
    f"{LORA_ALPHA}"
)


# ============================================================
# 29. LOSS LOG
# ============================================================

LOSS_CSV = (

    OUTPUT_DIR /
    "training_loss.csv"
)


if LOSS_CSV.exists():
    try:
        loss_records = (
            pd.read_csv(LOSS_CSV)
            .to_dict("records")
        )

        print(
            f"\nRestored loss history: "
            f"{len(loss_records):,} records"
        )

    except Exception as e:
        print(
            f"\nCould not restore training_loss.csv: {e}"
        )
        loss_records = []

else:
    loss_records = []

if EPOCH_METRICS_CSV.exists():

    try:
        epoch_metrics = (
            pd.read_csv(EPOCH_METRICS_CSV)
            .to_dict("records")
        )

        if len(epoch_metrics) > 0:

            best_row = min(
                epoch_metrics,
                key=lambda x: x["validation_loss"]
            )

            best_validation_loss = float(
                best_row["validation_loss"]
            )

            best_epoch = int(
                best_row["epoch"]
            )

        else:
            best_validation_loss = float("inf")
            best_epoch = None

    except Exception as e:

        print(
            f"\nCould not restore epoch_metrics.csv: {e}"
        )

        epoch_metrics = []
        best_validation_loss = float("inf")
        best_epoch = None

else:

    epoch_metrics = []
    best_validation_loss = float("inf")
    best_epoch = None
# ============================================================
# 29B. PERFORMANCE MEASUREMENTS FOR THE CURRENT RUN
# ============================================================
# After resuming, timings cover the current process only.
PERFORMANCE_JSON = OUTPUT_DIR / "training_performance_latest.json"
PERFORMANCE_EPOCH_CSV = OUTPUT_DIR / "training_performance_epochs.csv"
PERFORMANCE_LOG_EVERY_STEPS = 100
performance_started_at = time.perf_counter()
performance_initial_optimizer_step = optimizer_step
performance_initial_micro_step = global_micro_step
performance_resume_from = str(resume_checkpoint) if resume_checkpoint else None
performance_samples = 0
performance_batches = 0
performance_optimizer_steps = 0
performance_batch_compute_seconds = 0.0
performance_epoch_records = []
torch.cuda.reset_peak_memory_stats(DEVICE)

def write_performance_snapshot(status="running"):
    # Synchronize for meaningful elapsed wall time and CUDA peak stats.
    torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter() - performance_started_at
    payload = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(DEVICE),
        "status": status,
        "resume_checkpoint": performance_resume_from,
        "measurement_scope": "current_process_only; excludes steps before resume",
        "train_dataset_size": len(train_dataset),
        "epochs_configured": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "initial_optimizer_step": performance_initial_optimizer_step,
        "current_optimizer_step": optimizer_step,
        "optimizer_steps_this_run": performance_optimizer_steps,
        "microsteps_this_run": global_micro_step - performance_initial_micro_step,
        "samples_processed_this_run": performance_samples,
        "valid_batches_this_run": performance_batches,
        "elapsed_seconds_since_training_start": round(elapsed, 4),
        "training_wall_samples_per_second": round(performance_samples / elapsed, 4) if elapsed else None,
        "training_wall_optimizer_steps_per_second": round(performance_optimizer_steps / elapsed, 4) if elapsed else None,
        "mean_batch_compute_seconds": round(performance_batch_compute_seconds / performance_batches, 4) if performance_batches else None,
        "batch_compute_seconds_total": round(performance_batch_compute_seconds, 4),
        "peak_allocated_gpu_gb": round(torch.cuda.max_memory_allocated(DEVICE) / 1024**3, 4),
        "peak_reserved_gpu_gb": round(torch.cuda.max_memory_reserved(DEVICE) / 1024**3, 4),
        "epoch_segments": performance_epoch_records,
        "notes": "Wall throughput includes checkpoint saves and completed validation; batch compute time excludes DataLoader, CSV writes and checkpoint saves. CUDA memory is PyTorch process memory, not total system VRAM. Epoch segments may be partial after resume."
    }
    temp = PERFORMANCE_JSON.with_suffix(".json.tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temp, PERFORMANCE_JSON)
    return payload

# ============================================================
# 30. TRAIN
# ============================================================

model.train()


optimizer.zero_grad(
    set_to_none=True
)


print(
    "\n" +
    "=" * 75
)

print(
    "STARTING TRAINING"
)

print(
    "=" * 75
)


for epoch_idx in range(

    start_epoch,

    NUM_EPOCHS,
):


    # --------------------------------------------------------
    # Reproducible shuffle for each epoch
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Reproducible shuffle for each epoch
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        SEED + epoch_idx
    )
    # Reproduce the random draws made by the DataLoader
    # before the original epoch shuffle.
    _ = torch.empty(
        (),
        dtype=torch.int64
    ).random_(
        generator=generator
    ).item()

    # --------------------------------------------------------
    # Reconstruct the original shuffled order for the epoch
    # --------------------------------------------------------

    epoch_indices = torch.randperm(
        len(train_dataset),
        generator=generator,
    ).tolist()


    # --------------------------------------------------------
    # When resuming within the same epoch,
    # skip the samples already processed
    # --------------------------------------------------------

    batch_offset = 0

    if (
        epoch_idx == start_epoch
        and
        resume_batch_idx >= 0
    ):

        batch_offset = (
            resume_batch_idx + 1
        )

        start_sample = (
            batch_offset
            * BATCH_SIZE
        )

        epoch_indices = epoch_indices[
            start_sample:
        ]

        print(
            f"\nResuming after batch "
            f"{resume_batch_idx:,}"
        )

        print(
            f"Starting batch offset: "
            f"{batch_offset:,}"
        )

        print(
            f"Samples remaining in the epoch: "
            f"{len(epoch_indices):,}"
        )


    # --------------------------------------------------------
    # Build a subset from the remaining samples in the epoch
    # --------------------------------------------------------

    epoch_dataset = Subset(
        train_dataset,
        epoch_indices,
    )


    train_loader = DataLoader(

        epoch_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        collate_fn=collator,

        drop_last=False,
    )


    epoch_segment_start = time.perf_counter()
    epoch_segment_samples_start = performance_samples
    epoch_segment_steps_start = performance_optimizer_steps

    epoch_loss = 0.0

    epoch_valid_batches = 0


    progress_bar = tqdm(
    enumerate(train_loader),
    total=len(train_loader),
    desc=(
        f"Epoch {epoch_idx + 1}/{NUM_EPOCHS} "
        f"(from batch {batch_offset:,})"
    ),
    )


    for (
        batch_idx,
        batch,
    ) in progress_bar:


        original_batch_idx = (
            batch_idx + batch_offset
        )
        # Measure batch compute time, excluding data loading.
        torch.cuda.synchronize(DEVICE)
        batch_compute_start = time.perf_counter()

        # ----------------------------------------------------
        # Image IDs for debugging only
        # ----------------------------------------------------

        image_ids = batch.pop(
            "image_ids"
        )


        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        try:
            batch = move_to_device(
                batch,
                DEVICE,
            )


            # =================================================
            # FORWARD BF16
            # =================================================

            with torch.autocast(

                device_type="cuda",

                dtype=
                    torch.bfloat16,
            ):


                outputs = model(
                    **batch
                )


                loss = (
                    outputs.loss
                )


            # =================================================
            # CHECK LOSS
            # =================================================

            if not torch.isfinite(
                loss
            ):


                print(
                    f"\nNon-finite loss."
                    )

                print(
                    f"IDs: "
                    f"{image_ids}"
                )


                optimizer.zero_grad(
                    set_to_none=True
                )


                continue


            # =================================================
            # BACKWARD
            # =================================================

            scaled_loss = (

                loss /

                GRADIENT_ACCUMULATION_STEPS
            )


            scaled_loss.backward()


            global_micro_step += 1


            loss_value = (

                loss
                .detach()
                .float()
                .item()
            )


            epoch_loss += (
                loss_value
            )


            epoch_valid_batches += 1
            performance_samples += len(image_ids)
            performance_batches += 1

            # =================================================
            # OPTIMIZER STEP
            # =================================================

            should_step = (

                global_micro_step %

                GRADIENT_ACCUMULATION_STEPS

                == 0
            )


            is_last_batch = (

                batch_idx
                == len(train_loader) - 1
            )


            if (
                should_step
                or
                is_last_batch
            ):


                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    MAX_GRAD_NORM,
                )


                optimizer.step()


                scheduler.step()


                optimizer.zero_grad(
                    set_to_none=True
                )


                optimizer_step += 1
                performance_optimizer_steps += 1


                # =============================================
                # LOG
                # =============================================

                loss_records.append(

                    {

                        "epoch":
                            epoch_idx + 1,

                        "batch_idx":
                            original_batch_idx,

                        "optimizer_step":
                            optimizer_step,

                        "loss":
                            loss_value,

                        "learning_rate":
                            scheduler
                            .get_last_lr()[0],
                    }
                )


                pd.DataFrame(
                    loss_records
                ).to_csv(

                    LOSS_CSV,

                    index=False,
                )


                # =============================================
                # CHECKPOINT
                # =============================================

                if (

                    optimizer_step %

                    SAVE_EVERY_OPTIMIZER_STEPS

                    == 0

                ):


                    save_checkpoint(

                        model=
                            model,

                        processor=
                            processor,

                        optimizer=
                            optimizer,

                        scheduler=
                            scheduler,

                        epoch_idx=
                            epoch_idx,

                        batch_idx=
                            original_batch_idx,

                        global_micro_step=
                            global_micro_step,

                        optimizer_step=
                            optimizer_step,
                    )


            torch.cuda.synchronize(DEVICE)
            performance_batch_compute_seconds += time.perf_counter() - batch_compute_start
            if (performance_optimizer_steps > 0 and should_step and
                    performance_optimizer_steps % PERFORMANCE_LOG_EVERY_STEPS == 0):
                write_performance_snapshot("running")

            # =================================================
            # PROGRESS
            # =================================================

            average_loss = (

                epoch_loss /

                epoch_valid_batches
            )


            progress_bar.set_postfix(

                loss=
                    f"{loss_value:.4f}",

                avg=
                    f"{average_loss:.4f}",

                opt=
                    optimizer_step,

                lr=
                    f"{scheduler.get_last_lr()[0]:.2e}",

                vram=
                    f"{torch.cuda.memory_allocated()/1024**3:.1f}G",
            )


            # =================================================
            # CLEAN
            # =================================================

            del (
                batch,
                outputs,
                loss,
                scaled_loss,
            )


        except torch.cuda.OutOfMemoryError:


            print(
                "\n\nCUDA OUT OF MEMORY"
            )

            print(
                f"Batch IDs: "
                f"{image_ids}"
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            gc.collect()

            torch.cuda.empty_cache()


            raise


        except Exception as e:


            print(
                f"\n[TRAIN ERROR]"
            )

            print(
                f"IDs: "
                f"{image_ids}"
            )

            print(
                f"{type(e).__name__}: "
                f"{e}"
            )


            # Avoid CUDA calls after a failure because the context may be unusable.
            # Stop and resume from the last verified checkpoint.
            raise


    torch.cuda.synchronize(DEVICE)
    epoch_training_seconds = time.perf_counter() - epoch_segment_start
    epoch_segment_samples = performance_samples - epoch_segment_samples_start
    epoch_segment_steps = performance_optimizer_steps - epoch_segment_steps_start
    performance_epoch_records.append({
        "epoch": epoch_idx + 1,
        "partial_epoch_after_resume": bool(epoch_idx == start_epoch and batch_offset > 0),
        "training_seconds_excluding_validation": round(epoch_training_seconds, 4),
        "samples_processed": epoch_segment_samples,
        "optimizer_steps": epoch_segment_steps,
        "samples_per_second": round(epoch_segment_samples / epoch_training_seconds, 4) if epoch_training_seconds else None,
        "optimizer_steps_per_second": round(epoch_segment_steps / epoch_training_seconds, 4) if epoch_training_seconds else None,
    })
    pd.DataFrame(performance_epoch_records).to_csv(PERFORMANCE_EPOCH_CSV, index=False)
    write_performance_snapshot("epoch_training_completed")

    # ========================================================
    # END OF EPOCH
    # ========================================================

    print(
        "\n" +
        "=" * 75
    )

    print(
        f"EPOCH "
        f"{epoch_idx + 1} COMPLETE"
    )

    print(
        "=" * 75
    )


    epoch_average = (

        epoch_loss /

        max(
            epoch_valid_batches,
            1
        )
    )
    print(
        "\nComputing validation loss..."
    )
    
    validation_loss = calculate_validation_loss(
        model=model,
        valid_loader=validation_loader,
        device=DEVICE,
    )
    
    print(
        f"\nValidation loss: "
        f"{validation_loss:.6f}"
    )

    print(
        f"\nMean training loss: "
        f"{epoch_average:.6f}"
    )


    print(
        f"Allocated VRAM: "
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
    )


    print(
        f"Peak reserved VRAM: "
        f"{torch.cuda.max_memory_reserved()/1024**3:.2f} GB"
    )

    # --------------------------------------------------------
    # SAVE EPOCH METRICS
    # --------------------------------------------------------
    
    epoch_metrics.append(
        {
            "epoch": epoch_idx + 1,
            "train_loss": epoch_average,
            "validation_loss": validation_loss,
            "optimizer_step": optimizer_step,
        }
    )
    
    pd.DataFrame(
        epoch_metrics
    ).to_csv(
        EPOCH_METRICS_CSV,
        index=False,
    )
        
        # --------------------------------------------------------
    # SAVE THE BEST ADAPTER
    # --------------------------------------------------------

    if (
        math.isfinite(validation_loss)
        and
        validation_loss < best_validation_loss
    ):

        best_validation_loss = validation_loss
        best_epoch = epoch_idx + 1

        if BEST_ADAPTER_DIR.exists():
            shutil.rmtree(BEST_ADAPTER_DIR)

        BEST_ADAPTER_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(
            BEST_ADAPTER_DIR
        )

        processor.save_pretrained(
            BEST_ADAPTER_DIR
        )

        best_info = {
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "optimizer_step": optimizer_step,
        }

        with open(
            BEST_ADAPTER_DIR / "best_model_info.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                best_info,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\nSaved a new best adapter:"
        )

        print(
            BEST_ADAPTER_DIR
        )

        print(
            f"Epoch: {best_epoch}"
        )

        print(
            f"Validation loss: "
            f"{best_validation_loss:.6f}"
        )


    # --------------------------------------------------------
    # CHECKPOINT END OF EPOCH
    # --------------------------------------------------------

    save_checkpoint(

        model=model,

        processor=processor,

        optimizer=optimizer,

        scheduler=scheduler,

        epoch_idx=epoch_idx,

        batch_idx=(
            batch_offset
            + len(train_loader)
            - 1
        ),

        global_micro_step=global_micro_step,

        optimizer_step=optimizer_step,
    )


    # --------------------------------------------------------
    # QUALITATIVE VALIDATION
    # --------------------------------------------------------

    try:

        if hasattr(
            model.config,
            "use_cache"
        ):

            model.config.use_cache = True


        qualitative_validation(

            model=model,

            processor=processor,

            valid_df=valid_df,

            epoch_number=epoch_idx + 1,

            device=DEVICE,
        )


        if hasattr(
            model.config,
            "use_cache"
        ):

            model.config.use_cache = False


    except Exception as e:

        print(
            "\nQualitative validation error:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )


    write_performance_snapshot("epoch_completed")

    # Clear the batch offset after the resumed epoch
    resume_batch_idx = -1


# ============================================================
# 31. SAVE THE FINAL ADAPTER
# ============================================================

FINAL_DIR = (

    OUTPUT_DIR /
    "final_adapter"
)


FINAL_DIR.mkdir(

    parents=True,

    exist_ok=True,
)


model.save_pretrained(
    FINAL_DIR
)


processor.save_pretrained(
    FINAL_DIR
)


# ============================================================
# 32. SUMMARY
# ============================================================

print(
    "\n" +
    "=" * 75
)

print(
    "TRAINING COMPLETE"
)

print(
    "=" * 75
)


print(
    f"\nFinal adapter:\n"
    f"{FINAL_DIR}"
)

print(
    f"\nLoss:\n"
    f"{LOSS_CSV}"
)

print(
    f"\nCurrent allocated VRAM: "
    f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
)

print(
    f"Peak reserved VRAM: "
    f"{torch.cuda.max_memory_reserved()/1024**3:.2f} GB"
)

print(
    f"\nEpoch metrics:\n"
    f"{EPOCH_METRICS_CSV}"
)

print(
    f"\nBest adapter:\n"
    f"{BEST_ADAPTER_DIR}"
)

if best_epoch is not None:

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

final_performance = write_performance_snapshot("completed")
print(f"\nPerformance report: {PERFORMANCE_JSON}")
print(f"Epoch metrics: {PERFORMANCE_EPOCH_CSV}")
print(f"Samples per second (current run): {final_performance['training_wall_samples_per_second']}")
print(f"Peak allocated VRAM: {final_performance['peak_allocated_gpu_gb']} GB")

total_runtime = time.perf_counter() - execution_start

print(
    "\nTotal runtime:",
    timedelta(seconds=round(total_runtime))
)