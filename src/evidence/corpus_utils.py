import os
import shutil

def separate_evidence_images(base_dir):
    """
    Separates evidence images from the train directory and copies them into a new 'evidence_corpus' folder.

    Args:
        base_dir (str): The base directory containing the 'train' folder.
    """
    # Define paths
    datasets = ['train', 'test']
    evidence_corpus_dir = os.path.join(base_dir, 'evidence_corpus')

    # Create the evidence_corpus directory if it doesn't exist
    os.makedirs(evidence_corpus_dir, exist_ok=True)

    # Loop through the train directory and copy evidence images
    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        for filename in os.listdir(dataset_dir):
            if filename.split('_')[-1].split('.')[0] == 'evidence':
                new_filename = f"{dataset}_{filename}"
                source_path = os.path.join(dataset_dir, filename)
                target_path = os.path.join(evidence_corpus_dir, new_filename)

                shutil.copy(source_path, target_path)

    print("All evidence images in the train set have been copied.")

# Example usage
if __name__ == "__main__":
    base_dir = "C:\\Users\\defne\\Desktop\\2024-2025FallSemester\\Applied NLP\\multimodal-misinformation-detection\\data\\raw\\factify\\extracted\\images"  # Replace with your base directory path
    separate_evidence_images(base_dir)
