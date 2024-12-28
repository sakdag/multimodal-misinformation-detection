import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch
import pandas as pd
import pickle

import src.utils.path_utils
from src.utils.path_utils import get_project_root



class SemanticSimilarity:
    def __init__(self,train_embeddings_file, test_embeddings_file,  no_rerank=False):
        # We use the Bi-Encoder to encode all passages
        self.bi_encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.bi_encoder.max_seq_length = 512  # Truncate long passages to 256 tokens

        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.no_rerank = no_rerank
        print(f"self.no_rerank: {self.no_rerank}")

        self.train_csv = pd.read_csv(os.path.join(get_project_root(), "data", "preprocessed", "train_enriched.csv"))
        self.train_embeddings = torch.tensor(list(self._load_embeddings(train_embeddings_file).values()))
        # Combine all embeddings into one corpus with keys
        self.test_csv = pd.read_csv(os.path.join(get_project_root(), "data", "preprocessed", "test_enriched.csv"))
        self.test_embeddings = torch.tensor(list(self._load_embeddings(test_embeddings_file).values()))


    def _load_embeddings(self, file):
        """
        Load corpus and embeddings.
        """

        if os.path.exists(file):
            try:
                with open(file, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                print("Warning: Pickle file is empty or corrupted. Initializing empty feature dict.")
        return {}

    def extract_embeddings(self, query):
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        return question_embedding

    def search(self, query, top_k):
        if self.no_rerank:
            bi_encoder_top_k = top_k
        else:
            bi_encoder_top_k = 1000
        ##### Sematic Search #####
        # Encode the query using the bi-encoder and find potentially relevant passages
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        # question_embedding = question_embedding

        hits_train = util.semantic_search(question_embedding, self.train_embeddings, top_k=bi_encoder_top_k)
        hits_train = hits_train[0]  # Get the hits for the first query

        hits_test = util.semantic_search(question_embedding, self.test_embeddings, top_k=bi_encoder_top_k)
        hits_test = hits_test[0]

        if self.no_rerank:
            return
        else:
            ##### Re-Ranking #####
            # Now, score all retrieved passages with the cross_encoder
            cross_inp_train = [[query, self.train_csv[hit['corpus_id']]['evidence_enriched']] for hit in hits_train]
            cross_scores_train = self.cross_encoder.predict(cross_inp_train)

            cross_inp_test = [[query, self.test_csv[hit['corpus_id']]['evidence_enriched']] for hit in hits_test]
            cross_scores_test = self.cross_encoder.predict(cross_inp_test)

            # Sort results by the cross-encoder scores
            for idx in range(len(cross_scores_train)):
                hits_train[idx]['cross-score'] = cross_scores_train[idx]

            hits_after_cross_encoder = sorted(hits_train, key=lambda x: x['cross-score'], reverse=True)

            return hits_after_cross_encoder[:top_k]

class TextCorpus:
    def __init__(self, data_dir, split):
        self.bi_encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.split = split #train evidences or test evidences
        self.data_dir = data_dir #.csv file for enriched train and test is contained.
        self.embeddings_file = os.path.join(get_project_root(), f"{self.split}_evidence_embeddings.pkl")
        self.embedding_dict = self._load_embeddings()

    def _load_embeddings(self):
        """
        Load corpus and embeddings.
        """

        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                print("Warning: Pickle file is empty or corrupted. Initializing empty feature dict.")
        return {}

    def encode_corpus(self):
        """
        Encode the corpus (evidence_enriched column for both train and test) and store the embeddings.
        """
        # Process for both train and test

        file_path = os.path.join(self.data_dir, f'{self.split}_enriched.csv')
        df = pd.read_csv(file_path)

        # Open the embeddings file in append mode
        with open(self.embeddings_file, 'wb') as f:
            for index, row in df.iterrows():
                # Extract the enriched evidence and id
                evidence_enriched = row['evidence_enriched']
                idx = row['id']  # Assuming the 'id' column is in the CSV

                # Encode the evidence using the bi-encoder
                embedding = self.bi_encoder.encode(evidence_enriched, convert_to_tensor=False)

                # Create a key like 'train_0', 'test_1', etc.
                key = f"{self.split}_{idx}"
                self.embedding_dict[key] = embedding

                # Save the updated dictionary to the file
                pickle.dump(self.embedding_dict, f)

                # Clear the dictionary to avoid memory buildup
                self.embedding_dict = {}

        print(f"Embeddings saved incrementally to {self.embeddings_file}")


    def get_embedding(self, id):
        """
        Retrieve the embedding for a specific train/test_id.
        """
        key = f"{self.split}_{id}"
        return self.embedding_dict.get(key)


if __name__ == '__main__':
    project_root = get_project_root()
    data_dir = os.path.join(project_root, "data", "preprocessed")

    # Process train data
    # train_manager = TextCorpus(data_dir, "train")
    # train_manager.encode_corpus()

    # Process test data
    # test_manager = TextCorpus(data_dir, "test")
    # test_manager.encode_corpus()
    train_enriched = pd.read_csv(os.path.join(data_dir, "train_enriched.csv"))
    query = train_enriched['evidence_enriched'][0]
    train_embeddings = os.path.join(get_project_root(), 'train_evidence_embeddings.pkl')
    test_embeddings = os.path.join(get_project_root(), 'test_evidence_embeddings.pkl')

    semantic = SemanticSimilarity(train_embeddings, test_embeddings)
    semantic.search(query, top_k=10)






