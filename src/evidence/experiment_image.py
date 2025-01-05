from src.utils.path_utils import get_project_root
import pickle
import os

from im2im_retrieval import ImageCorpus

def calculate_topk_accuracy_image_retrieval(image_corpus, query_images, k_values=[1, 2, 5, 10]):

    num_hits_at_k = {k: 0 for k in k_values} #accuracy
    top_k = max(k_values)

    for query_image in query_images:
        query_image_path = os.path.join(get_project_root(),
                                        "data",
                                        "raw",
                                        "factify",
                                        "extracted",
                                        "images",
                                        "test",
                                        query_image)

        query_features = image_corpus.feature_extractor.extract_features(query_image_path)
        similarity_scores = {}

        for image_name, corpus_feature in image_corpus.feature_dict.items():
            similarity = image_corpus.feature_extractor.similarity(
                query_features, corpus_feature
            )
            similarity_scores[image_name] = similarity

        retrieved_images = sorted(
            similarity_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Filter out identical images (based on scores)
        unique_scores = set()
        filtered_images = []

        query_id = int(query_image.split('_')[0])

        for image_path, score in retrieved_images:
            image_id = int(image_path.split('\\')[-1].split('_')[0])
            if (score not in unique_scores) or (image_path.split('\\')[-2] == 'test' and query_id == image_id):  # Check if this score is already added
                unique_scores.add(score)
                filtered_images.append((image_path, score))

            if (
                    len(filtered_images) == top_k
            ):  # Stop once we have top_k unique images
                break

        relevant_evidence_id = str(query_id) + '_evidence'

        top_k_evidence_ids = [filtered_images[i][0].split('\\')[-1].split('.')[0] for i in range(top_k)]
        for k in k_values:
            if relevant_evidence_id in top_k_evidence_ids[:k]:
                print(f'Relevant evidence id: {relevant_evidence_id}')
                num_hits_at_k[k] += 1

    for k in num_hits_at_k:
        num_hits_at_k[k] /= len(query_images)

    return num_hits_at_k

def save_results_to_file(results, file_path):
    """Save results to a text file."""
    with open(file_path, 'w') as f:
        f.write("Top-k Accuracy Image Retrieval Results:\n")
        for k, accuracy in results.items():
            f.write(f"{k}: {accuracy}\n")

if __name__ == '__main__':

    project_root = get_project_root()
    image_feature = os.path.join(project_root, "evidence_features.pkl")
    image_dir = os.path.join(
        project_root, "data", "raw", "factify", "extracted", "images", "evidence_corpus"
    )  # Replace with your base directory path

    test_image_path = os.path.join(
        project_root,
        "data",
        "raw",
        "factify",
        "extracted",
        "images",
        "test",
    )

    test_images = [image for image in os.listdir(test_image_path) if image.split('_')[1] == 'claim.jpg']

    image_corpus = ImageCorpus(image_feature)

    results = calculate_topk_accuracy_image_retrieval(image_corpus, test_images)
    print(results)

    # Save the results to a text file
    output_file_path = os.path.join(project_root, "image_retrieval_topk_accuracy_results.txt")
    save_results_to_file(results, output_file_path)


