import os
import shutil

from src.utils.path_utils import get_project_root


def separate_evidence_images(base_dir):
    """
    Separates evidence images from the train directory and copies them into a new 'evidence_corpus' folder.

    Args:
        base_dir (str): The base directory containing the 'train' folder.
    """
    # Define paths
    datasets = ["train", "test"]
    evidence_corpus_dir = os.path.join(base_dir, "evidence_corpus")

    # Create the evidence_corpus directory if it doesn't exist
    os.makedirs(evidence_corpus_dir, exist_ok=True)

    # Loop through the train directory and copy evidence images
    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        for filename in os.listdir(dataset_dir):
            if filename.split("_")[-1].split(".")[0] == "evidence":
                new_filename = f"{dataset}_{filename}"
                source_path = os.path.join(dataset_dir, filename)
                target_path = os.path.join(evidence_corpus_dir, new_filename)

                shutil.copy(source_path, target_path)

    print("All evidence images in the train set have been copied.")


import pickle

# File path for the evidence features pickle
pickle_file_path = "evidence_features.pkl"


# Function to update the keys in the pickle
def update_pickle_keys(pickle_file_path, output_pickle_path=None):
    # Open and load the existing pickle
    with open(pickle_file_path, "rb") as f:
        feature_dict = pickle.load(f)

    updated_dict = {}

    # Update each key
    for old_path, features in feature_dict.items():
        # Extract the filename (e.g., test_0_evidence.jpg)
        filename = os.path.basename(old_path)

        # Determine if it's a test or train image based on the filename
        if filename.startswith("test"):
            new_relative_path = os.path.join(
                "data",
                "raw",
                "factify",
                "extracted",
                "images",
                "test",
                filename.split("_", 1)[1],
            )
        elif filename.startswith("train"):
            new_relative_path = os.path.join(
                "data",
                "raw",
                "factify",
                "extracted",
                "images",
                "train",
                filename.split("_", 1)[1],
            )
        else:
            raise ValueError(f"Unexpected filename format: {filename}")

        # Add the updated key and its value to the new dictionary
        updated_dict[new_relative_path] = features

    # Save the updated dictionary back to a pickle file
    output_path = output_pickle_path if output_pickle_path else pickle_file_path
    with open(output_path, "wb") as f:
        pickle.dump(updated_dict, f)

    print(f"Updated pickle saved at: {output_path}")


# Example usage
if __name__ == "__main__":
    pickle_file_path = "/evidence_features.pkl"
    project_root = get_project_root()
    # Run the function
    base_dir = os.path.join(
        project_root, "data", "raw", "factify", "extracted", "images"
    )
    separate_evidence_images(base_dir)

    # out_pkl_path = "C:\\Users\\defne\\Desktop\\2024-2025FallSemester\\Applied NLP\\multimodal-misinformation-detection\\data\\raw\\factify\\extracted\\images"
    # update_pickle_keys(pickle_file_path, output_pickle_path=out_pkl_path)
