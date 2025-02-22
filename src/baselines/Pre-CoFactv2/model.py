import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadAttention, PositionwiseFeedForward


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class FakeNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.text_embedding = nn.Sequential(
            nn.Linear(config["text_dim"], config["hidden_dim"]),
            # Mish()
            nn.ReLU(),
        )
        self.document_text_embedding = nn.Sequential(
            nn.Linear(config["text_dim"], config["hidden_dim"]),
            # Mish()
            nn.ReLU(),
        )

        self.image_embedding = nn.Sequential(
            nn.Linear(config["image_dim"], config["hidden_dim"]),
            # Mish()
            nn.ReLU(),
        )
        self.document_image_embedding = nn.Sequential(
            nn.Linear(config["image_dim"], config["hidden_dim"]),
            # Mish()
            nn.ReLU(),
        )

        self.claim_document_text_attention = MultiHeadAttention(
            config["head"],
            config["hidden_dim"],
            config["hidden_dim"],
            config["hidden_dim"],
            dropout=config["dropout"],
        )
        self.claim_document_text_pos_ffn = PositionwiseFeedForward(
            config["hidden_dim"], config["hidden_dim"] * 2, dropout=config["dropout"]
        )

        self.claim_document_image_attention = MultiHeadAttention(
            config["head"],
            config["hidden_dim"],
            config["hidden_dim"],
            config["hidden_dim"],
            dropout=config["dropout"],
        )
        self.claim_document_image_pos_ffn = PositionwiseFeedForward(
            config["hidden_dim"], config["hidden_dim"] * 2, dropout=config["dropout"]
        )

        self.text_image_attention = MultiHeadAttention(
            config["head"],
            config["hidden_dim"],
            config["hidden_dim"],
            config["hidden_dim"],
            dropout=config["dropout"],
        )
        self.text_image_pos_ffn = PositionwiseFeedForward(
            config["hidden_dim"], config["hidden_dim"] * 2, dropout=config["dropout"]
        )
        self.image_text_attention = MultiHeadAttention(
            config["head"],
            config["hidden_dim"],
            config["hidden_dim"],
            config["hidden_dim"],
            dropout=config["dropout"],
        )
        self.image_text_pos_ffn = PositionwiseFeedForward(
            config["hidden_dim"], config["hidden_dim"] * 2, dropout=config["dropout"]
        )

        self.claim_document_text_image_attention = MultiHeadAttention(
            config["head"],
            config["hidden_dim"],
            config["hidden_dim"],
            config["hidden_dim"],
            dropout=config["dropout"],
        )
        self.claim_document_text_image_pos_ffn = PositionwiseFeedForward(
            config["hidden_dim"], config["hidden_dim"] * 2, dropout=config["dropout"]
        )
        self.claim_document_image_text_attention = MultiHeadAttention(
            config["head"],
            config["hidden_dim"],
            config["hidden_dim"],
            config["hidden_dim"],
            dropout=config["dropout"],
        )
        self.claim_document_image_text_pos_ffn = PositionwiseFeedForward(
            config["hidden_dim"], config["hidden_dim"] * 2, dropout=config["dropout"]
        )

        self.attention_fusion = nn.Sequential(
            nn.Linear(config["hidden_dim"] * 16, config["hidden_dim"]),
            nn.ReLU(),
        )

        self.feature_embedding = nn.Sequential(nn.Linear(32, 16), nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Linear(16 + config["hidden_dim"], 128), nn.ReLU(), nn.Linear(128, 5)
        )

    def forward(
        self, claim_text, claim_image, document_text, document_image, add_feature
    ):
        # transform to embeddings
        claim_text_embedding = self.text_embedding(claim_text)
        claim_image_embedding = self.image_embedding(claim_image)
        document_text_embedding = self.document_text_embedding(document_text)
        document_image_embedding = self.document_image_embedding(document_image)

        # claim-document attention
        claim_document_text, _ = self.claim_document_text_attention(
            claim_text_embedding, document_text_embedding, document_text_embedding
        )
        claim_document_text = self.claim_document_text_pos_ffn(claim_document_text)
        document_claim_text, _ = self.claim_document_text_attention(
            document_text_embedding, claim_text_embedding, claim_text_embedding
        )
        document_claim_text = self.claim_document_text_pos_ffn(document_claim_text)

        claim_document_image, _ = self.claim_document_image_attention(
            claim_image_embedding, document_image_embedding, document_image_embedding
        )
        claim_document_image = self.claim_document_image_pos_ffn(claim_document_image)
        document_claim_image, _ = self.claim_document_image_attention(
            document_image_embedding, claim_image_embedding, claim_image_embedding
        )
        document_claim_image = self.claim_document_image_pos_ffn(document_claim_image)

        # text-image co-attention
        claim_text_image, _ = self.text_image_attention(
            claim_text_embedding, claim_image_embedding, claim_image_embedding
        )
        claim_text_image = self.text_image_pos_ffn(claim_text_image)
        claim_image_text, _ = self.image_text_attention(
            claim_image_embedding, claim_text_embedding, claim_text_embedding
        )
        claim_image_text = self.image_text_pos_ffn(claim_image_text)

        document_text_image, _ = self.text_image_attention(
            document_text_embedding, document_image_embedding, document_image_embedding
        )
        document_text_image = self.text_image_pos_ffn(document_text_image)
        document_image_text, _ = self.image_text_attention(
            document_image_embedding, document_text_embedding, document_text_embedding
        )
        document_image_text = self.image_text_pos_ffn(document_image_text)

        claim_text_document_image, _ = self.text_image_attention(
            claim_text_embedding, document_image_embedding, document_image_embedding
        )
        claim_text_document_image = self.text_image_pos_ffn(claim_text_document_image)
        claim_image_document_text, _ = self.image_text_attention(
            claim_image_embedding, document_text_embedding, document_text_embedding
        )
        claim_image_document_text = self.image_text_pos_ffn(claim_image_document_text)

        document_image_claim_text, _ = self.claim_document_image_text_attention(
            document_image_embedding, claim_text_embedding, claim_text_embedding
        )
        document_image_claim_text = self.claim_document_text_image_pos_ffn(
            document_image_claim_text
        )
        document_text_claim_image, _ = self.claim_document_text_image_attention(
            document_text_embedding, claim_image_embedding, claim_image_embedding
        )
        document_text_claim_image = self.claim_document_image_text_pos_ffn(
            document_text_claim_image
        )

        # aggregate word and image embedding to sentence embedding
        claim_document_text = torch.mean(claim_document_text, dim=1)
        document_claim_text = torch.mean(document_claim_text, dim=1)
        claim_document_image = torch.mean(claim_document_image, dim=1)
        document_claim_image = torch.mean(document_claim_image, dim=1)
        claim_text_embedding = torch.mean(claim_text_embedding, dim=1)
        document_text_embedding = torch.mean(document_text_embedding, dim=1)
        claim_image_embedding = torch.mean(claim_image_embedding, dim=1)
        document_image_embedding = torch.mean(document_image_embedding, dim=1)
        claim_text_document_image = torch.mean(claim_text_document_image, dim=1)
        claim_image_document_text = torch.mean(claim_image_document_text, dim=1)
        document_image_claim_text = torch.mean(document_image_claim_text, dim=1)
        document_text_claim_image = torch.mean(document_text_claim_image, dim=1)

        claim_text_image = torch.mean(claim_text_image, dim=1)
        claim_image_text = torch.mean(claim_image_text, dim=1)
        document_text_image = torch.mean(document_text_image, dim=1)
        document_image_text = torch.mean(document_image_text, dim=1)

        concat_text_image_embeddings = torch.cat(
            (
                claim_text_embedding,
                claim_image_embedding,
                document_text_embedding,
                document_image_embedding,
                claim_document_text,
                document_claim_text,
                claim_document_image,
                document_claim_image,
                claim_text_image,
                claim_image_text,
                document_text_image,
                document_image_text,
                claim_text_document_image,
                claim_image_document_text,
                document_image_claim_text,
                document_text_claim_image,
            ),
            dim=-1,
        )

        text_image_embeddings = self.attention_fusion(concat_text_image_embeddings)
        feature_embeddings = self.feature_embedding(add_feature)

        concat_embeddings = torch.cat(
            (text_image_embeddings, feature_embeddings), dim=-1
        )

        predicted_output = self.classifier(concat_embeddings)

        return predicted_output, concat_embeddings
