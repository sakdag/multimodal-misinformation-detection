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


def initialize_csv(input_csv: str, output_csv: str) -> None:
    """Initializes the output CSV with additional columns if not already created."""
    if not os.path.exists(output_csv):
        df = pd.read_csv(input_csv)

        # Add new columns
        df["claim_image_caption"] = ""
        df["evidence_image_caption"] = ""
        df["claim_enriched"] = ""
        df["evidence_enriched"] = ""

        # Save the initialized CSV
        df.to_csv(output_csv, index=False)


def process_csv(input_csv: str, output_csv: str) -> None:
    """Processes the CSV in chunks and writes results incrementally."""
    # Initialize the output CSV if not already created
    initialize_csv(input_csv, output_csv)

    # Load the output CSV to check progress
    output_df = pd.read_csv(output_csv)
    input_df = pd.read_csv(input_csv)

    # Ensure alignment between input and output CSVs
    if len(output_df) != len(input_df):
        raise ValueError("Mismatch between input and output CSV row counts.")

    for batch_start in tqdm(
        range(0, len(input_df), BATCH_SIZE), desc="Processing rows"
    ):
        batch_end = min(batch_start + BATCH_SIZE, len(input_df))
        batch = input_df.iloc[batch_start:batch_end]

        results = []
        for _, row in batch.iterrows():
            if pd.isna(output_df.at[batch_start + _, "claim_image_caption"]):
                results.append(process_image_row(row))
            else:
                results.append(
                    (
                        output_df.at[batch_start + _, "claim_image_caption"],
                        output_df.at[batch_start + _, "evidence_image_caption"],
                        output_df.at[batch_start + _, "claim_enriched"],
                        output_df.at[batch_start + _, "evidence_enriched"],
                    )
                )

        # Update the output dataframe with results
        for idx, (claim_cap, evidence_cap, claim_enr, evidence_enr) in enumerate(
            results
        ):
            output_idx = batch_start + idx
            output_df.at[output_idx, "claim_image_caption"] = claim_cap
            output_df.at[output_idx, "evidence_image_caption"] = evidence_cap
            output_df.at[output_idx, "claim_enriched"] = claim_enr
            output_df.at[output_idx, "evidence_enriched"] = evidence_enr

        # Save progress to CSV after processing each batch
        output_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = f"{PROCESSED_DIR}/train.csv"
    output_csv = f"{PROCESSED_DIR}/train_enriched.csv"

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file does not exist: {input_csv}")

    process_csv(input_csv, output_csv)
    print(f"Processing complete. Output saved to {output_csv}")
