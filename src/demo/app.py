import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)


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


# Streamlit app
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


if __name__ == "__main__":
    main()
