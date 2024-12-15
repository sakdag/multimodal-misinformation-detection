import os
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor

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


def generate_caption(image_path):
    """Generates a caption for an image given its path."""
    try:
        image = Image.open(f"{PROJECT_ROOT}/{image_path}").convert("RGB")
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""


def generate_conditional_caption(image_path, text):
    """Generates a conditional caption for an image given its path and a prompt."""
    try:
        image = Image.open(f"{PROJECT_ROOT}/{image_path}").convert("RGB")
        inputs = processor(image, text, return_tensors="pt")
        output = model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error processing image {image_path} with prompt '{text}': {e}")
        return ""


def process_image_row(row):
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


def process_csv(input_csv, output_csv):
    """Processes the CSV to add image captions and enriched columns."""
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Ensure the required columns are present
    required_columns = ["claim", "claim_image", "evidence", "evidence_image"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Generate captions and enriched columns
    claim_captions = []
    evidence_captions = []
    claim_conditional_captions = []
    evidence_conditional_captions = []
    claim_enriched = []
    evidence_enriched = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        claim_image_caption = generate_caption(row["claim_image"])
        evidence_image_caption = generate_caption(row["evidence_image"])

        # claim_conditional_caption = generate_conditional_caption(
        #     row["claim_image"], "a photography of"
        # )
        # evidence_conditional_caption = generate_conditional_caption(
        #     row["evidence_image"], "a photography of"
        # )

        claim_captions.append(claim_image_caption)
        evidence_captions.append(evidence_image_caption)

        # claim_conditional_captions.append(claim_conditional_caption)
        # evidence_conditional_captions.append(evidence_conditional_caption)

        claim_enriched.append(f"{row['claim']}. {claim_image_caption}")
        evidence_enriched.append(f"{row['evidence']}. {evidence_image_caption}")

    # Add the new columns to the dataframe
    df["claim_image_caption"] = claim_captions
    df["evidence_image_caption"] = evidence_captions
    # df["claim_image_caption_conditional"] = claim_conditional_captions
    # df["evidence_image_caption_conditional"] = evidence_conditional_captions
    df["claim_enriched"] = claim_enriched
    df["evidence_enriched"] = evidence_enriched

    # Save the updated dataframe to the output CSV
    df.to_csv(output_csv, index=False)


def process_csv_parallel(input_csv, output_csv):
    """Processes the CSV to add image captions and enriched columns."""
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Ensure the required columns are present
    required_columns = ["claim", "claim_image", "evidence", "evidence_image"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Prepare lists for captions and enriched text
    claim_captions = []
    evidence_captions = []
    claim_enriched = []
    evidence_enriched = []

    # Use ThreadPoolExecutor to parallelize the row processing
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, row in df.iterrows():
            futures.append(executor.submit(process_image_row, row))

        # Monitor progress using tqdm
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing rows"
        ):
            (
                claim_image_caption,
                evidence_image_caption,
                claim_enriched_text,
                evidence_enriched_text,
            ) = future.result()

            # Append results to lists
            claim_captions.append(claim_image_caption)
            evidence_captions.append(evidence_image_caption)
            claim_enriched.append(claim_enriched_text)
            evidence_enriched.append(evidence_enriched_text)

    # Add the new columns to the dataframe
    df["claim_image_caption"] = claim_captions
    df["evidence_image_caption"] = evidence_captions
    df["claim_enriched"] = claim_enriched
    df["evidence_enriched"] = evidence_enriched

    # Save the updated dataframe to the output CSV
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = f"{PROCESSED_DIR}/test.csv"
    output_csv = f"{PROCESSED_DIR}/test_enriched.csv"

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file does not exist: {input_csv}")

    process_csv(input_csv, output_csv)
    print(f"Processing complete. Output saved to {output_csv}")
