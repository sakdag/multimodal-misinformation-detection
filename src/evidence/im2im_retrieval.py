import os.path
from torchvision.models import resnet50
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
import torch
import pickle
import matplotlib.pyplot as plt

class ImageSimilarity:

    def __init__(self):
        self.model = resnet50(weights='DEFAULT')
        self.model = nn.Sequential(*list(self.model.children())[:-1])  #Ignoring the last classification layer
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def extract_features(self, image_stream):
        image = Image.open(image_stream).convert('RGB')
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            features = self.model(image)
            features = features.flatten()
        return features

    def similarity(self, features1, features2):
        # Calculating cosine similarity
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos(features1.unsqueeze(0), features2.unsqueeze(0))
        return similarity.item()

class ImageCorpus:
    def __init__(self, feature_corpus_path):
        self.feature_corpus_path = feature_corpus_path
        self.feature_dict = self.load_features()
        self.feature_extractor = ImageSimilarity()

    def load_features(self):
        if os.path.exists(self.feature_corpus_path):
            try:
                with open(self.feature_corpus_path, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                print("Warning: Pickle file is empty or corrupted. Initializing empty feature dict.")
        return {}

    def save_features(self):
        with open(self.feature_corpus_path, 'wb') as f:
            pickle.dump(self.feature_dict, f)

    def add_image(self, image_path):
        features = self.feature_extractor.extract_features(image_path)
        self.feature_dict[image_path] = features
        self.save_features()

    def create_feature_corpus(self, image_dir):

        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                features = self.feature_extractor.extract_features(image_path)
                self.feature_dict[image_path] = features

        self.save_features()

    def retrieve_similar_images(self, query_image_path, top_k=50):
        query_features = self.feature_extractor.extract_features(query_image_path)
        similarity_scores = {}

        for image_name, corpus_feature in self.feature_dict.items():
            similarity = self.feature_extractor.similarity(query_features, corpus_feature)
            similarity_scores[image_name] = similarity

        retrieved_images = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

        # Filter out identical images (based on scores)
        unique_scores = set()
        filtered_images = []

        for image_path, score in retrieved_images:
            if score not in unique_scores:  # Check if this score is already added
                unique_scores.add(score)
                filtered_images.append((image_path, score))

            if len(filtered_images) == top_k:  # Stop once we have top_k unique images
                break

        return filtered_images

def visualize_retrieved_images(query_image_path, top_retrievals):
    # Load query image
    query_image = Image.open(query_image_path).convert("RGB")

    # Load retrieved images and their scores
    retrieved_images = [(Image.open(img_path).convert("RGB"), score) for img_path, score in top_retrievals]

    # Set up the grid for visualization
    total_retrieved = len(retrieved_images)
    rows = 2 + (total_retrieved - 1) // 5  # 1 row for query + rows for 5 images per row
    cols = 5

    # Set figure size
    plt.figure(figsize=(20, rows * 4))

    # Plot query image at the top row (centered in row of 5)
    plt.subplot(rows, cols, (cols // 2) + 1)  # Center in the first row
    plt.imshow(query_image)
    plt.title("Query Image", fontsize=12)
    plt.axis('off')

    # Plot retrieved images
    for idx, (img, score) in enumerate(retrieved_images):
        plt.subplot(rows, cols, cols + idx + 1)  # Start plotting after the query image
        plt.imshow(img)
        plt.title(f"Rank: {idx+1}\nScore: {score:.4f}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    image_feature = "C:\\Users\\defne\\Desktop\\2024-2025FallSemester\\Applied NLP\\multimodal-misinformation-detection\\data\\raw\\factify\\extracted\\images\\evidence_features.pkl"
    image_dir = "C:\\Users\\defne\\Desktop\\2024-2025FallSemester\\Applied NLP\\multimodal-misinformation-detection\\data\\raw\\factify\\extracted\\images\\evidence_corpus"  # Replace with your base directory path
    query_image_path = "C:\\Users\\defne\\Desktop\\2024-2025FallSemester\\Applied NLP\\multimodal-misinformation-detection\\data\\raw\\factify\\extracted\\images\\test\\1_claim.jpg"

    image_corpus = ImageCorpus(image_feature)
    top_retrievals = image_corpus.retrieve_similar_images(query_image_path, top_k=5)
    visualize_retrieved_images(query_image_path, top_retrievals)
