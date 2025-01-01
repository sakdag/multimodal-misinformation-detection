import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import os

from evaluate import MisinformationPredictor
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

            # Create an Evidence object
            evidences.append(
                Evidence(
                    text=evidence_text,
                    image=evidence_image,
                    caption=evidence_image_caption,
                    dataset=evidence_path.split("/")[-2],
                    evidence_id=evidence_id_number,
                    image_path=full_image_path,
                )
            )
    except Exception as e:
        st.error(f"Error performing semantic search: {e}")

    return evidences


@st.cache_resource
def get_predictor():
    return MisinformationPredictor(model_path="ckpts/model.pt", device="cpu")


def classify_evidence(
    claim_text: str, claim_image_path: str, evidence_text: str, evidence_image_path: str
) -> Tuple[str, str, str, str]:
    """Assigns a random classification to each evidence."""
    predictor = get_predictor()
    predictions = predictor.evaluate(
        claim_text, claim_image_path, evidence_text, evidence_image_path
    )
    if predictions:
        return (
            predictions.get("text_text", "not_enough_information"),
            predictions.get("text_image", "not_enough_information"),
            predictions.get("image_text", "not_enough_information"),
            predictions.get("image_image", "not_enough_information"),
        )
    else:
        return (
            "not_enough_information",
            "not_enough_information",
            "not_enough_information",
            "not_enough_information",
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
    st.write("Upload claims that have image and/or text content to verify.")

    # File uploader for images
    uploaded_image = st.file_uploader(
        "Upload an image (1 max)", type=["jpg", "jpeg", "png"], key="image_uploader"
    )

    if uploaded_image:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to display the image: {e}")

    # Text input field
    input_text = st.text_area("Enter text (max 4096 characters)", "", max_chars=4096)

    # Sliders for top_k values
    col1, col2 = st.columns(2)
    with col1:
        top_k_text = st.slider(
            "Top-k Text Evidences", min_value=1, max_value=5, value=2, key="top_k_text"
        )
    with col2:
        top_k_image = st.slider(
            "Top-k Image Evidences",
            min_value=1,
            max_value=5,
            value=2,
            key="top_k_image",
        )

    # Generate Enriched Text button
    if st.button("Verify Claim"):
        if not uploaded_image and not input_text:
            st.warning("Please upload an image or enter text.")
            return

        progress = st.progress(0)

        # Step 1: Generate caption
        progress.progress(10)
        st.write("### Step 1: Generating caption...")
        image_caption = ""
        if uploaded_image:
            image_caption = generate_caption(image)
            st.write("**Generated Image Caption:**", image_caption)

        # Step 2: Enrich text
        progress.progress(40)
        st.write("### Step 2: Enriching text...")
        enriched_text = enrich_text_with_caption(input_text, image_caption)
        st.write("**Enriched Text:**")
        st.write(enriched_text)

        # Step 3: Retrieve evidences by text
        progress.progress(50)
        st.write("### Step 3: Retrieving evidences by text...")
        if input_text:
            text_evidences = retrieve_evidences_by_text(enriched_text, top_k=top_k_text)
            st.write(f"Retrieved {len(text_evidences)} text evidences.")
        else:
            text_evidences = None
            st.write("Text modality is missing from the input claim!")

        # Step 4: Retrieve evidences by image
        progress.progress(70)
        st.write("### Step 4: Retrieving evidences by image...")
        if uploaded_image:
            image_evidences = retrieve_evidences_by_image(
                uploaded_image, top_k=top_k_image
            )
            st.write(f"Retrieved {len(image_evidences)} image evidences.")
        else:
            image_evidences = None
            st.write("Image modality is missing from the input claim!")

        # Step 5: Classify evidences
        progress.progress(90)
        st.write("### Step 5: Verifying claim with retrieved evidences...")
        for evidence in (text_evidences or []) + (image_evidences or []):
            a, b, c, d = classify_evidence(
                claim_text=enriched_text,
                claim_image_path=uploaded_image,
                evidence_text=evidence.text,
                evidence_image_path=evidence.image_path,
            )
            evidence.classification_result = a, b, c, d

        # Step 6: Display evidences
        progress.progress(100)
        if text_evidences or image_evidences:
            st.write("## Results")
            tabs = st.tabs(["Text Evidences", "Image Evidences"])

            with tabs[0]:
                if text_evidences:
                    st.write("### Text Evidences")
                    display_evidence_tab(text_evidences, "text")
                else:
                    st.write("Text modality is missing from the input claim!")

            with tabs[1]:
                if image_evidences:
                    st.write("### Image Evidences")
                    display_evidence_tab(image_evidences, "image")
                else:
                    st.write("Image modality is missing from the input claim!")


if __name__ == "__main__":
    main()
