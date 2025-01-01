import random
import time

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import os

from src.evidence.im2im_retrieval import ImageCorpus
from src.evidence.text2text_retrieval import SemanticSimilarity
from src.utils.path_utils import get_project_root
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

PROJECT_ROOT = get_project_root()


@dataclass
class Evidence:
    evidence_id: str
    dataset: str
    text: Optional[str]
    image: Optional[Image.Image]
    caption: Optional[str]
    image_path: Optional[str]
    classification_result: Optional[Tuple[str, str, str, str]] = None


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


@st.cache_data
def get_train_df():
    data_dir = os.path.join(PROJECT_ROOT, "data", "preprocessed")
    train_csv_path = os.path.join(data_dir, "train_enriched.csv")
    return pd.read_csv(train_csv_path)


@st.cache_data
def get_test_df():
    data_dir = os.path.join(PROJECT_ROOT, "data", "preprocessed")
    train_csv_path = os.path.join(data_dir, "test_enriched.csv")
    return pd.read_csv(train_csv_path)


@st.cache_data
def get_semantic_similarity(
    train_embeddings_file: str,
    test_embeddings_file: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    return SemanticSimilarity(
        train_embeddings_file=train_embeddings_file,
        test_embeddings_file=test_embeddings_file,
        train_df=train_df,
        test_df=test_df,
    )


def retrieve_evidences_by_text(
    query: str,
    top_k: int = 5,
) -> List[Evidence]:
    """
    Retrieves evidence rows from preloaded embeddings and CSV data using semantic similarity.

    Args:
        query (str): The query text to perform the search.
        top_k (int): Number of top results to retrieve.

    Returns:
        List[Evidence]: A list of retrieved evidence objects.
    """
    train_embeddings_file = os.path.join(PROJECT_ROOT, "train_embeddings.h5")
    test_embeddings_file = os.path.join(PROJECT_ROOT, "test_embeddings.h5")
    similarity = get_semantic_similarity(
        train_embeddings_file=train_embeddings_file,
        test_embeddings_file=test_embeddings_file,
        train_df=get_train_df(),
        test_df=get_test_df(),
    )
    evidences = []
    try:
        # Perform semantic search across both train and test datasets
        results = similarity.search(query=query, top_k=top_k)

        # Retrieve evidence rows based on the search results
        for evidence_id, score in results:
            # Determine whether the ID belongs to train or test set
            if evidence_id.startswith("train_"):
                df = similarity.train_csv
            elif evidence_id.startswith("test_"):
                df = similarity.test_csv
            else:
                continue  # Skip invalid IDs

            # Extract the row by ID
            row = df[df["id"] == int(evidence_id.split("_")[1])].iloc[0]
            evidence_text = row.get("evidence_enriched")
            evidence_image_caption = row.get("evidence_image_caption")
            evidence_image_path = row.get("evidence_image")
            evidence_image = None
            full_image_path = None

            # Load the image if a valid path is provided
            if pd.notna(evidence_image_path):
                full_image_path = os.path.join(PROJECT_ROOT, evidence_image_path)
                try:
                    evidence_image = Image.open(full_image_path).convert("RGB")
                except Exception as e:
                    st.error(f"Failed to load image {evidence_image_path}: {e}")

            evidence_id_number = evidence_id.split("_")[1]
            evidence_dataset = evidence_id.split("_")[0]

            # Create an Evidence object
            evidences.append(
                Evidence(
                    text=evidence_text,
                    image=evidence_image,
                    caption=evidence_image_caption,
                    evidence_id=evidence_id_number,
                    dataset=evidence_dataset,
                    image_path=full_image_path,
                )
            )
    except Exception as e:
        st.error(f"Error performing semantic search: {e}")

    return evidences


@st.cache_data
def get_image_corpus(image_features):
    return ImageCorpus(image_features)


def retrieve_evidences_by_image(
    image_path: str,
    top_k: int = 5,
) -> List[Evidence]:
    """
    Retrieves evidence rows from preloaded embeddings and CSV data using semantic similarity.

    Args:
        query (str): The query text to perform the search.
        top_k (int): Number of top results to retrieve.

    Returns:
        List[Evidence]: A list of retrieved evidence objects.
    """
    image_features = os.path.join(PROJECT_ROOT, "evidence_features.pkl")
    image_corpus = get_image_corpus(image_features)
    evidences = []
    try:
        # Perform semantic search across both train and test datasets
        results = image_corpus.retrieve_similar_images(image_path, top_k=top_k)

        # Retrieve evidence rows based on the search results
        for evidence_path, score in results:
            evidence_id = evidence_path.split("/")[-1]
            evidence_id_number = evidence_id.split("_")[0]
            # Determine whether the ID belongs to train or test set
            if "train" in evidence_path:
                df = get_train_df()
            elif "test" in evidence_path:
                df = get_test_df()
            else:
                continue  # Skip invalid IDs

            # Extract the row by ID
            row = df[df["id"] == int(evidence_id_number)].iloc[0]
            evidence_text = row.get("evidence_enriched")
            evidence_image_caption = row.get("evidence_image_caption")
            evidence_image_path = row.get("evidence_image")
            evidence_image = None
            full_image_path = None

            # Load the image if a valid path is provided
            if pd.notna(evidence_image_path):
                full_image_path = os.path.join(PROJECT_ROOT, evidence_image_path)
                try:
                    evidence_image = Image.open(full_image_path).convert("RGB")
                except Exception as e:
                    st.error(f"Failed to load image {evidence_image_path}: {e}")

            evidence_dataset = evidence_id.split("_")[1].split(".")[0]

            # Create an Evidence object
            evidences.append(
                Evidence(
                    text=evidence_text,
                    image=evidence_image,
                    caption=evidence_image_caption,
                    dataset=evidence_dataset,
                    evidence_id=evidence_id_number,
                    image_path=full_image_path,
                )
            )
    except Exception as e:
        st.error(f"Error performing semantic search: {e}")

    return evidences


def classify_evidence(
    claim_text: str, claim_image_path: str, evidence_text: str, evidence_image_path: str
) -> Tuple[str, str, str, str]:
    """Assigns a random classification to each evidence."""
    time.sleep(1)
    return (
        random.choice(CLASSIFICATION_CATEGORIES),
        random.choice(CLASSIFICATION_CATEGORIES),
        random.choice(CLASSIFICATION_CATEGORIES),
        random.choice(CLASSIFICATION_CATEGORIES),
    )


def display_evidence_tab(evidences: List[Evidence], tab_label: str):
    """Displays evidence in a tabbed format."""
    with st.container():
        for index, evidence in enumerate(evidences):
            with st.container():
                st.subheader(f"Evidence {index + 1}")
                st.write(f"Evidence Dataset: {evidence.dataset}")
                st.write(f"Evidence ID: {evidence.evidence_id}")
                if evidence.image:
                    st.image(
                        evidence.image,
                        caption="Evidence Image",
                        use_container_width=True,
                    )
                st.text_area(
                    "Evidence Caption",
                    value=evidence.caption or "No caption available.",
                    height=100,
                    key=f"caption_{tab_label}_{index}",
                    disabled=True,
                )
                st.text_area(
                    "Evidence Text",
                    value=evidence.text or "No text available.",
                    height=100,
                    key=f"text_{tab_label}_{index}",
                    disabled=True,
                )
                if evidence.classification_result:
                    st.write("**Classification:**")
                    st.write(f"**text|text:** {evidence.classification_result[0]}")
                    st.write(f"**text|image:** {evidence.classification_result[1]}")
                    st.write(f"**image|text:** {evidence.classification_result[2]}")
                    st.write(f"**image|image:** {evidence.classification_result[3]}")


def main():
    st.title("Multimodal Evidence-Based Misinformation Classification")

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

    # Generate Enriched Text button
    if st.button("Verify Claim"):
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

        # Evidence retrieved using text
        text_evidences = retrieve_evidences_by_text(enriched_text)
        for text_evidence in text_evidences:
            a, b, c, d = classify_evidence(
                claim_text=enriched_text,
                claim_image_path=st.session_state.uploaded_image,
                evidence_text=text_evidence.text,
                evidence_image_path=text_evidence.image_path,
            )
            text_evidence.classification_result = a, b, c, d

        # Evidence retrieved using image
        image_evidences = retrieve_evidences_by_image(st.session_state.uploaded_image)
        for image_evidence in image_evidences:
            a, b, c, d = classify_evidence(
                claim_text=enriched_text,
                claim_image_path=st.session_state.uploaded_image,
                evidence_text=image_evidence.text,
                evidence_image_path=image_evidence.image_path,
            )
            image_evidence.classification_result = a, b, c, d

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


if __name__ == "__main__":
    main()
