import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import os
from src.utils.path_utils import get_project_root
from src.utils.data_utils import PREPROCESSED_DIR
from typing import List, Optional
from dataclasses import dataclass
import random

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

PROJECT_ROOT = get_project_root()


@dataclass
class Evidence:
    evidence_id: str
    evidence: Optional[str]
    evidence_image: Optional[Image.Image]
    evidence_image_caption: Optional[str]
    classification: Optional[str] = None


CLASSIFICATION_CATEGORIES = ["support", "refute", "not_enough_information"]


def generate_caption(image: Image.Image) -> str:
    """Generates a caption for a given image."""
    try:
        with st.spinner("Generating caption..."):
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs)
            return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return ""


def enrich_text_with_caption(text: str, image_caption: str) -> str:
    """Appends the image caption to the given text."""
    if image_caption:
        return f"{text}. {image_caption}"
    return text


def retrieve_evidence_by_text(csv_path: str, num_rows: int = 3) -> List[Evidence]:
    """Retrieves random evidence rows from a CSV file."""
    evidences = []
    try:
        df = pd.read_csv(csv_path)
        sampled_rows = df.sample(n=num_rows)
        for _, row in sampled_rows.iterrows():
            evidence_text = row.get("evidence")
            evidence_image_caption = row.get("evidence_image_caption")
            evidence_id = row.get("id")

            evidence_image_path = row.get("evidence_image")
            evidence_image = None
            if pd.notna(evidence_image_path):
                full_image_path = os.path.join(PROJECT_ROOT, evidence_image_path)
                try:
                    evidence_image = Image.open(full_image_path).convert("RGB")
                except Exception as e:
                    st.error(f"Failed to load image {evidence_image_path}: {e}")

            evidences.append(
                Evidence(
                    evidence=evidence_text,
                    evidence_image=evidence_image,
                    evidence_image_caption=evidence_image_caption,
                    evidence_id=evidence_id,
                )
            )
    except Exception as e:
        st.error(f"Error loading CSV: {e}")

    return evidences


def retrieve_evidence_by_image(csv_path: str, num_rows: int = 3) -> List[Evidence]:
    """Retrieves random evidence rows from a CSV file."""
    evidences = []
    try:
        df = pd.read_csv(csv_path)
        sampled_rows = df.sample(n=num_rows)
        for _, row in sampled_rows.iterrows():
            evidence_text = row.get("evidence")
            evidence_image_caption = row.get("evidence_image_caption")
            evidence_id = row.get("id")

            evidence_image_path = row.get("evidence_image")
            evidence_image = None
            if pd.notna(evidence_image_path):
                full_image_path = os.path.join(PROJECT_ROOT, evidence_image_path)
                try:
                    evidence_image = Image.open(full_image_path).convert("RGB")
                except Exception as e:
                    st.error(f"Failed to load image {evidence_image_path}: {e}")

            evidences.append(
                Evidence(
                    evidence=evidence_text,
                    evidence_image=evidence_image,
                    evidence_image_caption=evidence_image_caption,
                    evidence_id=evidence_id,
                )
            )
    except Exception as e:
        st.error(f"Error loading CSV: {e}")

    return evidences


def classify_evidence(evidences: List[Evidence], claim: str) -> List[Evidence]:
    """Assigns a random classification to each evidence."""
    for evidence in evidences:
        evidence.classification = random.choice(CLASSIFICATION_CATEGORIES)
    return evidences


def determine_final_classification(evidences: List[Evidence]) -> str:
    """Determines the final classification based on evidence classifications."""
    votes = {category: 0 for category in CLASSIFICATION_CATEGORIES}
    for evidence in evidences:
        if evidence.classification:
            votes[evidence.classification] += 1

    max_votes = max(votes.values())
    top_categories = [k for k, v in votes.items() if v == max_votes]

    if len(top_categories) > 1:
        return "not_enough_information"
    return top_categories[0]


def display_evidence_tab(evidences: List[Evidence], tab_label: str):
    """Displays evidence in a tabbed format."""
    with st.container():
        for index, evidence in enumerate(evidences):
            with st.container():
                st.subheader(f"Evidence {index + 1}")
                if evidence.evidence_image:
                    st.image(
                        evidence.evidence_image,
                        caption="Evidence Image",
                        use_container_width=True,
                    )
                st.text_area(
                    "Evidence Caption",
                    value=evidence.evidence_image_caption or "No caption available.",
                    height=100,
                    key=f"caption_{tab_label}_{index}",
                    disabled=True,
                )
                st.text_area(
                    "Evidence Text",
                    value=evidence.evidence or "No text available.",
                    height=100,
                    key=f"text_{tab_label}_{index}",
                    disabled=True,
                )
                if evidence.classification:
                    st.write(f"**Classification:** {evidence.classification}")


def main():
    st.title("Text and Image Enrichment App")

    st.write(
        "Upload an image and/or enter text to enrich the text with an image caption."
    )

    # File uploader for images (only one image allowed)
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    uploaded_image = st.file_uploader(
        "Upload an image (1 max)", type=["jpg", "jpeg", "png"], key="image_uploader"
    )

    if uploaded_image:
        st.session_state.uploaded_image = uploaded_image

    # Display the uploaded image immediately
    if st.session_state.uploaded_image:
        try:
            image = Image.open(st.session_state.uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to display the image: {e}")

    # Text input field (limit to 4096 characters)
    input_text = st.text_area("Enter text (max 4096 characters)", "", max_chars=4096)

    # Reset button
    if st.button("Reset"):
        st.session_state.uploaded_image = None
        st.experimental_rerun()

    # Generate Enriched Text button
    if st.button("Generate Enriched Text"):
        if not st.session_state.uploaded_image and not input_text:
            st.warning("Please upload an image or enter text.")
            return

        # Generate caption if an image is uploaded
        image_caption = ""
        if st.session_state.uploaded_image:
            image_caption = generate_caption(image)
            st.write("**Generated Image Caption:**", image_caption)

        # Enrich text with the generated caption
        with st.spinner("Enriching text..."):
            enriched_text = enrich_text_with_caption(input_text, image_caption)

        # Display the enriched text
        st.write("**Enriched Text:**")
        st.write(enriched_text)

        # Load dataset and fetch evidence rows
        dataset = "test_enriched"
        csv_path = os.path.join(PREPROCESSED_DIR, f"{dataset}.csv")

        # Evidence retrieved using text
        text_evidences = retrieve_evidence_by_text(csv_path)
        text_evidences = classify_evidence(text_evidences, enriched_text)

        # Evidence retrieved using image
        image_evidences = retrieve_evidence_by_image(csv_path)
        image_evidences = classify_evidence(image_evidences, enriched_text)

        # Interactive evidence display using tabs
        if text_evidences or image_evidences:
            st.write("## Retrieved Evidences")
            tabs = st.tabs(["Text Evidences", "Image Evidences"])

            # Display text evidences
            with tabs[0]:
                st.write("### Text Evidences")
                display_evidence_tab(text_evidences, "text")

            # Display image evidences
            with tabs[1]:
                st.write("### Image Evidences")
                display_evidence_tab(image_evidences, "image")

            # Final classification result
            all_evidences = text_evidences + image_evidences
            final_classification = determine_final_classification(all_evidences)
            st.write("## Final Classification")
            for idx, evidence in enumerate(text_evidences):
                st.write(
                    f"Text evidence {idx + 1} classification result: {evidence.classification}"
                )
            for idx, evidence in enumerate(image_evidences):
                st.write(
                    f"Image evidence {idx + 1} classification result: {evidence.classification}"
                )
            st.write(f"**Result:** {final_classification}")


if __name__ == "__main__":
    main()
