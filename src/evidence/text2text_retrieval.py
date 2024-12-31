import h5py
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os
import torch
import pandas as pd

from src.utils.path_utils import get_project_root


class SemanticSimilarity:
    def __init__(
        self,
        train_embeddings_file,
        test_embeddings_file,
        train_csv_path,
        test_csv_path,
        no_rerank=False,
    ):
        # We use the Bi-Encoder to encode all passages
        self.bi_encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        self.bi_encoder.max_seq_length = 512  # Truncate long passages to 256 tokens

        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.train_embeddings, self.train_ids = self._load_embeddings(
            train_embeddings_file
        )
        self.test_embeddings, self.test_ids = self._load_embeddings(
            test_embeddings_file
        )

        # Load corresponding CSV files for enriched evidence
        self.train_csv = pd.read_csv(train_csv_path)
        self.test_csv = pd.read_csv(test_csv_path)

    def _load_embeddings(self, h5_file_path):
        """
        Load embeddings and IDs from the HDF5 file
        """
        with h5py.File(h5_file_path, "r") as h5_file:
            embeddings = torch.tensor(h5_file["embeddings"][:], dtype=torch.float16)
            ids = list(h5_file["ids"][:])  # Retrieve the IDs as a list of strings

        return embeddings, ids

    def search(self, query, top_k):
        ##### Sematic Search #####
        # Encode the query using the bi-encoder and find potentially relevant passages
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        question_embedding = question_embedding.to(dtype=torch.float16)
        # question_embedding = question_embedding

        hits_train = util.semantic_search(
            question_embedding, self.train_embeddings, top_k=top_k * 5
        )
        hits_train = hits_train[0]  # Get the hits for the first query
        print(f"len(hits_train) = {len(hits_train)}")
        hits_test = util.semantic_search(
            question_embedding, self.test_embeddings, top_k=top_k * 5
        )
        hits_test = hits_test[0]
        print(f"len(hits_test): {len(hits_test)}")

        ##### Re-Ranking #####
        # Now, score all retrieved passages with the cross_encoder
        cross_inp_train = [
            [query, self.train_csv["evidence_enriched"][hit["corpus_id"]]]
            for hit in hits_train
        ]
        cross_scores_train = self.cross_encoder.predict(cross_inp_train)

        cross_inp_test = [
            [query, self.test_csv["evidence_enriched"][hit["corpus_id"]]]
            for hit in hits_test
        ]
        cross_scores_test = self.cross_encoder.predict(cross_inp_test)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores_train)):
            hits_train[idx]["cross-score"] = cross_scores_train[idx]

        for idx in range(len(cross_scores_test)):
            hits_test[idx]["cross-score"] = cross_scores_test[idx]

        hits_train_cross_encoder = sorted(
            hits_train, key=lambda x: x.get("cross-score"), reverse=True
        )
        hits_train_cross_encoder = hits_train_cross_encoder[: top_k * 5]
        hits_test_cross_encoder = sorted(
            hits_test, key=lambda x: x.get("cross-score"), reverse=True
        )
        hits_test_cross_encoder = hits_test_cross_encoder[: top_k * 5]

        results = [
            (self.train_ids[hit["corpus_id"]].decode("utf-8"), hit.get("cross-score"))
            for hit in hits_train_cross_encoder
        ] + [
            (self.test_ids[hit["corpus_id"]].decode("utf-8"), hit.get("cross-score"))
            for hit in hits_test_cross_encoder
        ]

        ##### Filter out duplicates based on scores #####
        unique_scores = set()
        filtered_results = []

        print(results)
        for id_, score in sorted(results, key=lambda x: x[1], reverse=True):
            if score not in unique_scores:
                unique_scores.add(score)
                filtered_results.append((id_, score))

            if (
                len(filtered_results) == top_k
            ):  # Stop when top_k unique scores are reached
                break

        return filtered_results


class TextCorpus:
    def __init__(self, data_dir, split):
        self.bi_encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        self.split = split  # train evidences or test evidences
        self.data_dir = data_dir  # .csv file for enriched train and test is contained.

    def encode_corpus(self):
        """
        Encode the corpus (evidence_enriched column for both train and test) and store the embeddings.
        """
        file_path = os.path.join(self.data_dir, f"{self.split}_enriched.csv")
        df = pd.read_csv(file_path)

        # Extract the enriched evidence column and ids
        evidence_enriched = df["evidence_enriched"].tolist()
        ids = df["id"].tolist()  # Assuming the 'id' column is in the CSV

        # Encode the evidence using the bi-encoder
        embeddings = self.bi_encoder.encode(evidence_enriched, convert_to_tensor=True)

        # Define HDF5 file path
        h5_file_path = os.path.join(get_project_root(), f"{self.split}_embeddings.h5")

        with h5py.File(h5_file_path, "w") as h5_file:
            h5_file.create_dataset(
                "embeddings", data=embeddings.numpy(), dtype="float16"
            )

            h5_file.create_dataset(
                "ids",
                data=[f"{self.split}_{id}" for id in ids],
                dtype=h5py.string_dtype(),
            )

        print(f"Embeddings saved to {h5_file_path}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    project_root = get_project_root()
    data_dir = os.path.join(project_root, "data", "preprocessed")

    # query = train_enriched['evidence_enriched'][0]
    # train_embeddings = os.path.join(get_project_root(), 'train_evidence_embeddings.pkl')
    # test_embeddings = os.path.join(get_project_root(), 'test_evidence_embeddings.pkl')

    # semantic = SemanticSimilarity(train_embeddings, test_embeddings)
    # semantic.search(query, top_k=10)

    # evidence = TextCorpus(data_dir, 'train')

    # Define file paths
    train_csv_path = os.path.join(data_dir, "train_enriched.csv")
    test_csv_path = os.path.join(data_dir, "test_enriched.csv")
    train_embeddings_file = os.path.join(project_root, "train_embeddings.h5")
    test_embeddings_file = os.path.join(project_root, "test_embeddings.h5")

    # Initialize the SemanticSimilarity class
    similarity = SemanticSimilarity(
        train_embeddings_file=train_embeddings_file,
        test_embeddings_file=test_embeddings_file,
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
        no_rerank=False,  # Set to True if reranking is not needed
    )

    # Load the first query from train_enriched.csv
    train_df = pd.read_csv(train_csv_path)
    first_query = train_df["claim_enriched"].iloc[2]  # Get the first query

    # Define the number of top-k results to retrieve
    top_k = 5

    # Perform the semantic search
    results = similarity.search(query=first_query, top_k=top_k)
    finish_time = time.time() - start_time
    # Display the results

    print(results)
    print(f"Finish time: {finish_time}")
