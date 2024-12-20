import os
from typing import Tuple
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from src.utils.path_utils import get_project_root

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

PROJECT_ROOT = get_project_root()
RAW_DIR = PROJECT_ROOT / "data/raw/factify"
PROCESSED_DIR = PROJECT_ROOT / "data/preprocessed"

BATCH_SIZE = 20  # Number of rows to process per batch


def generate_caption(image_path: str) -> str:
    """Generates a caption for an image given its path."""
    try:
        image = Image.open(f"{PROJECT_ROOT}/{image_path}").convert("RGB")
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""


def process_image_row(row: pd.Series) -> Tuple[str, str, str, str]:
    """Processes a single row to generate captions and enriched columns."""
    claim_image_caption = generate_caption(row["claim_image"])
    evidence_image_caption = generate_caption(row["evidence_image"])

    claim_enriched = f"{row['claim']}. {claim_image_caption}"
    evidence_enriched = f"{row['evidence']}. {evidence_image_caption}"

    return (
        claim_image_caption,
        evidence_image_caption,
        claim_enriched,
        evidence_enriched,
    )


def get_last_processed_index(df: pd.DataFrame) -> int:
    """
    Find the last processed row index by searching backwards from the end
    until finding a row where evidence_image_caption is not NA.
    Returns -1 if no processed rows are found.
    """
    for idx in range(len(df) - 1, -1, -1):
        if pd.notna(df.loc[idx, "evidence_image_caption"]):
            return idx
    return -1


def process_csv(input_csv: str, output_csv: str) -> None:
    """Processes the CSV in chunks and writes results incrementally with efficient checkpointing."""
    # Load input DataFrame
    input_df = pd.read_csv(input_csv)

    # Initialize or load output DataFrame
    if os.path.exists(output_csv):
        output_df = pd.read_csv(output_csv)
        if len(output_df) != len(input_df):
            print(
                "Mismatch in input and output CSV lengths. Reinitializing output CSV..."
            )
    else:
        output_df = input_df.copy()
        for col in [
            "claim_image_caption",
            "evidence_image_caption",
            "claim_enriched",
            "evidence_enriched",
        ]:
            output_df[col] = pd.NA

    # Find the last processed index
    last_processed_idx = get_last_processed_index(output_df)
    print(f"Resuming from index {last_processed_idx + 1}")

    # Process remaining rows in batches
    total_rows = len(input_df)
    with tqdm(total=total_rows, initial=last_processed_idx + 1) as pbar:
        for idx in range(last_processed_idx + 1, total_rows, BATCH_SIZE):
            batch_end = min(idx + BATCH_SIZE, total_rows)

            # Process each row in the batch
            for row_idx in range(idx, batch_end):
                row = input_df.iloc[row_idx]

                # Skip if already processed
                if pd.notna(output_df.at[row_idx, "evidence_image_caption"]):
                    continue

                # Process the row
                claim_cap, evidence_cap, claim_enr, evidence_enr = process_image_row(
                    row
                )

                # Update the output DataFrame
                output_df.loc[row_idx, "claim_image_caption"] = claim_cap
                output_df.loc[row_idx, "evidence_image_caption"] = evidence_cap
                output_df.loc[row_idx, "claim_enriched"] = claim_enr
                output_df.loc[row_idx, "evidence_enriched"] = evidence_enr

                pbar.update(1)

            # Save after each batch
            output_df.to_csv(output_csv, index=False)
            print(f"Saved progress at index {batch_end}")


if __name__ == "__main__":
    for name in ["train", "test"]:
        input_csv = f"{PROCESSED_DIR}/{name}.csv"
        output_csv = f"{PROCESSED_DIR}/{name}_enriched.csv"

        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input CSV file does not exist: {input_csv}")

        process_csv(input_csv, output_csv)
        print(f"Processing complete. Output saved to {output_csv}")
