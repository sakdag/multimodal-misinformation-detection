# The effect of text enrichment on the retrieval

# top k accuracy

from text2text_retrieval import SemanticSimilarity
import os
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from src.utils.path_utils import get_project_root

def calculate_topk_accuracy_text_retrieval(similarity,
                            k_values=[1, 2, 5, 10]):

    num_hits_at_k = {k: 0 for k in k_values}  # accuracy

    queries = similarity.test_csv["claim"]
    top_k = max(k_values)

    for query_id, query in enumerate(queries):

        question_embedding = similarity.bi_encoder.encode(query, convert_to_tensor=True, device=similarity.device)
        question_embedding = question_embedding.to(dtype=torch.float16)
        # question_embedding = question_embedding

        hits_train = util.semantic_search(
            question_embedding, similarity.train_embeddings, top_k=top_k * 10
        )
        hits_train = hits_train[0]  # Get the hits for the first query
        # print(f"len(hits_train) = {len(hits_train)}")
        hits_test = util.semantic_search(
            question_embedding, similarity.test_embeddings, top_k=top_k * 10
        )
        hits_test = hits_test[0]
        # print(f"len(hits_test): {len(hits_test)}")

        ##### Re-Ranking #####
        # Now, score all retrieved passages with the cross_encoder
        cross_inp_train = [
            [query, similarity.train_csv["evidence"][hit["corpus_id"]]]
            for hit in hits_train
        ]
        cross_scores_train = similarity.cross_encoder.predict(cross_inp_train)

        cross_inp_test = [
            [query, similarity.test_csv["evidence"][hit["corpus_id"]]]
            for hit in hits_test
        ]
        cross_scores_test = similarity.cross_encoder.predict(cross_inp_test)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores_train)):
            hits_train[idx]["cross-score"] = cross_scores_train[idx]

        for idx in range(len(cross_scores_test)):
            hits_test[idx]["cross-score"] = cross_scores_test[idx]

        hits_train_cross_encoder = sorted(
            hits_train, key=lambda x: x.get("cross-score"), reverse=True
        )
        hits_train_cross_encoder = hits_train_cross_encoder
        hits_test_cross_encoder = sorted(
            hits_test, key=lambda x: x.get("cross-score"), reverse=True
        )
        hits_test_cross_encoder = hits_test_cross_encoder

        results = [
                      (similarity.train_ids[hit["corpus_id"]].decode("utf-8"), hit.get("cross-score"))
                      for hit in hits_train_cross_encoder
                  ] + [
                      (similarity.test_ids[hit["corpus_id"]].decode("utf-8"), hit.get("cross-score"))
                      for hit in hits_test_cross_encoder
                  ]

        # ##### Filter out duplicates based on scores #####
        unique_scores = set()
        filtered_results = []

        # print(results)
        for id_, score in sorted(results, key=lambda x: x[1], reverse=True):
            if (score not in unique_scores) or (id_.split('_')[0] == 'test' and query_id == int(id_.split("_")[1])):
                unique_scores.add(score)
                filtered_results.append((id_, score))

            if (
                    len(filtered_results) == top_k
            ):  # Stop when top_k unique scores are reached
                break

        relevant_evidence_id = 'test_' + str(query_id)

        try:
            top_k_retrieved_ids = [filtered_results[i][0] for i in range(top_k)]
            for k in k_values:
                if relevant_evidence_id in top_k_retrieved_ids[:k]:
                    num_hits_at_k[k] += 1
        except IndexError as e:
            top_k_retrieved_ids = [filtered_results[i][0] for i in range(len(filtered_results))]
            for k in k_values:
                if relevant_evidence_id in top_k_retrieved_ids[:min(k, len(filtered_results))]:
                    num_hits_at_k[k] += 1


    for k in num_hits_at_k:
        num_hits_at_k[k] /= len(queries)

    return num_hits_at_k

def save_results_to_file(results, file_path):
    """Save results to a text file."""
    with open(file_path, "w") as f:
        f.write("Top-k Accuracy Results:\n")
        for k, accuracy in results.items():
            f.write(f"{k}: {accuracy}\n")

if __name__ == '__main__':
    project_root = get_project_root()
    data_dir = os.path.join(project_root, "data", "preprocessed")

    train_csv_path = os.path.join(data_dir, "train_enriched.csv")
    test_csv_path = os.path.join(data_dir, "test_enriched.csv")
    train_embeddings_file = os.path.join(project_root, "train_embeddings_notenriched.h5")
    test_embeddings_file = os.path.join(project_root, "test_embeddings_notenriched.h5")

    similarity = SemanticSimilarity(
        train_embeddings_file=train_embeddings_file,
        test_embeddings_file=test_embeddings_file,
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
    )

    results = calculate_topk_accuracy_text_retrieval(similarity, k_values=[1, 2, 5, 10])
    print(results)

    # Save the results to a text file
    output_file_path = os.path.join(project_root, "topk_accuracy_results_notenriched.txt")
    save_results_to_file(results, output_file_path)
