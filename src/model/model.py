import torch
import torch.nn as nn
from .layers import MLP, MultiHeadAttention


class MultiViewClaimRepresentation(nn.Module):
    """
    Multi-view claim representation module with transformer-like architecture
    for self-attention and cross-attention in text and image modalities.
    """
    def __init__(self, text_input_dim=384, image_input_dim=1024, embed_dim=512, num_heads=8, dropout=0.1, mlp_ratio=4.0, fused_attn=False):
        super().__init__()
        self.text_input_dim = text_input_dim
        self.image_input_dim = image_input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.text_proj = nn.Linear(text_input_dim, embed_dim)
        self.image_proj = nn.Linear(image_input_dim, embed_dim)
        
        # Text projections for attention
        self.text_WQ = nn.Linear(embed_dim, embed_dim)
        self.text_WK = nn.Linear(embed_dim, embed_dim)
        self.text_WV = nn.Linear(embed_dim, embed_dim)
        
        # Image projections for attention
        self.image_WQ = nn.Linear(embed_dim, embed_dim)
        self.image_WK = nn.Linear(embed_dim, embed_dim)
        self.image_WV = nn.Linear(embed_dim, embed_dim)

        # Output projections
        self.text_self_attn_out = nn.Linear(embed_dim, embed_dim)
        self.image_self_attn_out = nn.Linear(embed_dim, embed_dim)
        self.text_cross_attn_out = nn.Linear(embed_dim, embed_dim)
        self.image_cross_attn_out = nn.Linear(embed_dim, embed_dim)

        # Layer norms
        self.text_self_ln1 = nn.LayerNorm(embed_dim)
        self.text_self_ln2 = nn.LayerNorm(embed_dim)
        self.image_self_ln1 = nn.LayerNorm(embed_dim)
        self.image_self_ln2 = nn.LayerNorm(embed_dim)
        self.text_cross_ln1 = nn.LayerNorm(embed_dim)
        self.text_cross_ln2 = nn.LayerNorm(embed_dim)
        self.image_cross_ln1 = nn.LayerNorm(embed_dim)
        self.image_cross_ln2 = nn.LayerNorm(embed_dim)

        # MLPs
        self.text_mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.image_mlp = MLP(embed_dim, mlp_ratio, dropout)

        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout, fused_attn)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, X_t=None, X_i=None):
        """
        Args:
            X_t (Tensor): Text embeddings of shape (B, L_t, D)
            X_i (Tensor): Image embeddings of shape (B, L_i, D)

        Returns:
            (H_t_fused, H_i_fused):
                H_t_fused: Text representations with self- and co-attention
                H_i_fused: Image representations with self- and co-attention
        """
        # Project inputs to embedding dimension first
        if X_t is not None:
            X_t = self.text_proj(X_t)
        if X_i is not None:
            X_i = self.image_proj(X_i)
        
        # Pre-compute Q,K,V for both modalities if present
        text_Q = self.text_WQ(X_t) if X_t is not None else None
        text_K = self.text_WK(X_t) if X_t is not None else None
        text_V = self.text_WV(X_t) if X_t is not None else None

        image_Q = self.image_WQ(X_i) if X_i is not None else None
        image_K = self.image_WK(X_i) if X_i is not None else None
        image_V = self.image_WV(X_i) if X_i is not None else None

        # Unimodal text case
        if X_t is not None and X_i is None:
            # Self attention without MLP
            H_t = X_t + self.attention(text_Q, text_K, text_V, self.text_self_attn_out)
            H_t = self.text_self_ln1(H_t)
            # Apply MLP after self attention
            H_t = H_t + self.text_mlp(H_t)
            H_t = self.text_self_ln2(H_t)
            return H_t, None

        # Unimodal image case
        if X_i is not None and X_t is None:
            # Self attention without MLP
            H_i = X_i + self.attention(image_Q, image_K, image_V, self.image_self_attn_out)
            H_i = self.image_self_ln1(H_i)
            # Apply MLP after self attention
            H_i = H_i + self.image_mlp(H_i)
            H_i = self.image_self_ln2(H_i)
            return None, H_i

        # Multimodal case
        # Text processing
        H_t = X_t + self.attention(text_Q, text_K, text_V, self.text_self_attn_out)  # Self attention
        H_t = self.text_self_ln1(H_t)
        C_t = H_t + self.attention(H_t, text_K, text_V, self.text_cross_attn_out)  # Cross attention
        C_t = self.text_cross_ln1(C_t)
        # Apply MLP after combined attention
        C_t = C_t + self.text_mlp(C_t)
        C_t = self.text_cross_ln2(C_t)

        # Image processing
        H_i = X_i + self.attention(image_Q, image_K, image_V, self.image_self_attn_out)  # Self attention
        H_i = self.image_self_ln1(H_i)
        C_i = H_i + self.attention(H_i, image_K, image_V, self.image_cross_attn_out)  # Cross attention
        C_i = self.image_cross_ln1(C_i)
        # Apply MLP after combined attention
        C_i = C_i + self.image_mlp(C_i)
        C_i = self.image_cross_ln2(C_i)

        return C_t, C_i


class CrossAttentionEvidenceConditioning(nn.Module):
    """
    Cross-attention module to condition claim representations
    on textual and visual evidence.
    """
    def __init__(self, text_input_dim=384, image_input_dim=1024, embed_dim=768, num_heads=8, dropout=0.1, mlp_ratio=4.0, fused_attn=False):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.fused_attn = fused_attn

        # Query projections
        self.text_WQ = nn.Linear(embed_dim, embed_dim)
        self.image_WQ = nn.Linear(embed_dim, embed_dim)

        # Text evidence projections
        self.text_evidence_key = nn.Linear(text_input_dim, embed_dim)
        self.text_evidence_value = nn.Linear(text_input_dim, embed_dim)

        # Image evidence projections
        self.image_evidence_key = nn.Linear(image_input_dim, embed_dim)
        self.image_evidence_value = nn.Linear(image_input_dim, embed_dim)

        # Separate output projections for each attention path
        self.text_text_out = nn.Linear(embed_dim, embed_dim)
        self.text_image_out = nn.Linear(embed_dim, embed_dim)
        self.image_text_out = nn.Linear(embed_dim, embed_dim)
        self.image_image_out = nn.Linear(embed_dim, embed_dim)

        # Separate layer norms for each attention path
        self.text_text_ln1 = nn.LayerNorm(embed_dim)
        self.text_text_ln2 = nn.LayerNorm(embed_dim)
        self.text_image_ln1 = nn.LayerNorm(embed_dim)
        self.text_image_ln2 = nn.LayerNorm(embed_dim)
        self.image_text_ln1 = nn.LayerNorm(embed_dim)
        self.image_text_ln2 = nn.LayerNorm(embed_dim)
        self.image_image_ln1 = nn.LayerNorm(embed_dim)
        self.image_image_ln2 = nn.LayerNorm(embed_dim)

        # MLPs
        self.text_mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.image_mlp = MLP(embed_dim, mlp_ratio, dropout)

        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout, fused_attn)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, H_t=None, H_i=None, E_t=None, E_i=None):
        """
        Returns:
            (S_t, S_i): Each contains a tuple of (text_evidence_output, image_evidence_output)
        """
        S_t_t, S_t_i = None, None
        S_i_t, S_i_i = None, None

        if H_t is not None:
            # Text-to-text evidence attention
            S_t_t = self.attention(
                Q=self.text_WQ(H_t),
                K=self.text_evidence_key(E_t),
                V=self.text_evidence_value(E_t),
                out_proj=self.text_text_out
            )
            S_t_t = H_t + S_t_t
            S_t_t = self.text_text_ln1(S_t_t)
            S_t_t = S_t_t + self.text_mlp(S_t_t)
            S_t_t = self.text_text_ln2(S_t_t)

            # Text-to-image evidence attention
            S_t_i = self.attention(
                Q=self.text_WQ(H_t),
                K=self.image_evidence_key(E_i),
                V=self.image_evidence_value(E_i),
                out_proj=self.text_image_out
            )
            S_t_i = H_t + S_t_i
            S_t_i = self.text_image_ln1(S_t_i)
            S_t_i = S_t_i + self.text_mlp(S_t_i)
            S_t_i = self.text_image_ln2(S_t_i)

        if H_i is not None:
            # Image-to-text evidence attention
            S_i_t = self.attention(
                Q=self.image_WQ(H_i),
                K=self.text_evidence_key(E_t),
                V=self.text_evidence_value(E_t),
                out_proj=self.image_text_out
            )
            S_i_t = H_i + S_i_t
            S_i_t = self.image_text_ln1(S_i_t)
            S_i_t = S_i_t + self.image_mlp(S_i_t)
            S_i_t = self.image_text_ln2(S_i_t)

            # Image-to-image evidence attention
            S_i_i = self.attention(
                Q=self.image_WQ(H_i),
                K=self.image_evidence_key(E_i),
                V=self.image_evidence_value(E_i),
                out_proj=self.image_image_out
            )
            S_i_i = H_i + S_i_i
            S_i_i = self.image_image_ln1(S_i_i)
            S_i_i = S_i_i + self.image_mlp(S_i_i)
            S_i_i = self.image_image_ln2(S_i_i)

        return (S_t_t, S_t_i), (S_i_t, S_i_i)


class ClassificationModule(nn.Module):
    """
    Classification module that takes final text/image representations
    and outputs logits for {support, refute, not enough info}
    for each evidence path.
    """
    def __init__(self, embed_dim=768, hidden_dim=256, num_classes=3, dropout=0.1):
        super().__init__()
        # MLPs for text representations
        self.mlp_text_given_text = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.mlp_text_given_image = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # MLPs for image representations
        self.mlp_image_given_text = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.mlp_image_given_image = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, S_t=None, S_i=None):
        """
        Args:
            S_t: Tuple of (text_given_text, text_given_image) representations
            S_i: Tuple of (image_given_text, image_given_image) representations
        Returns:
            y_t: Tuple of (text_given_text_logits, text_given_image_logits)
            y_i: Tuple of (image_given_text_logits, image_given_image_logits)
        """
        y_t_t, y_t_i = None, None
        y_i_t, y_i_i = None, None

        if S_t is not None:
            S_t_t, S_t_i = S_t
            if S_t_t is not None:
                pooled_t_t = S_t_t.mean(dim=1)
                y_t_t = self.mlp_text_given_text(pooled_t_t)
            if S_t_i is not None:
                pooled_t_i = S_t_i.mean(dim=1)
                y_t_i = self.mlp_text_given_image(pooled_t_i)

        if S_i is not None:
            S_i_t, S_i_i = S_i
            if S_i_t is not None:
                pooled_i_t = S_i_t.mean(dim=1)
                y_i_t = self.mlp_image_given_text(pooled_i_t)
            if S_i_i is not None:
                pooled_i_i = S_i_i.mean(dim=1)
                y_i_i = self.mlp_image_given_image(pooled_i_i)

        return (y_t_t, y_t_i), (y_i_t, y_i_i)


class MisinformationDetectionModel(nn.Module):
    """
    End-to-end model combining:
    1) Multi-view claim representation
    2) Cross-attention evidence conditioning
    3) Classification for each evidence path
    """
    def __init__(self, 
                 text_input_dim=384,   # DeBERTa-v3-xsmall hidden size
                 image_input_dim=1024,  # Swinv2-base hidden size
                 embed_dim=512,
                 num_heads=8,
                 dropout=0.1,
                 hidden_dim=256,
                 num_classes=3,
                 mlp_ratio=4.0,
                 fused_attn=False):
        super().__init__()
        
        self.representation = MultiViewClaimRepresentation(
            text_input_dim=text_input_dim,
            image_input_dim=image_input_dim,
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            fused_attn=fused_attn
        )
        self.cross_attn = CrossAttentionEvidenceConditioning(
            text_input_dim=text_input_dim,
            image_input_dim=image_input_dim,
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            fused_attn=fused_attn
        )
        self.classifier = ClassificationModule(
            embed_dim=embed_dim, 
            hidden_dim=hidden_dim, 
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, X_t=None, X_i=None, E_t=None, E_i=None):
        """
        Args:
            X_t (Tensor): Text claim embeddings (B, L_t, D)
            X_i (Tensor): Image claim embeddings (B, L_i, D)
            E_t (Tensor): Text evidence embeddings (B, L_e_t, D)
            E_i (Tensor): Image evidence embeddings (B, L_e_i, D)
        
        Returns:
            y_t: Tuple of (text_given_text_logits, text_given_image_logits)
            y_i: Tuple of (image_given_text_logits, image_given_image_logits)
            Each logit tensor has shape (B, num_classes)
        """
        # Get fused claim representations
        H_t, H_i = self.representation(X_t, X_i)
        
        # Get evidence-conditioned representations for each path
        (S_t_t, S_t_i), (S_i_t, S_i_i) = self.cross_attn(H_t, H_i, E_t, E_i)
        
        # Get predictions for each evidence path
        (y_t_t, y_t_i), (y_i_t, y_i_i) = self.classifier(
            S_t=(S_t_t, S_t_i),
            S_i=(S_i_t, S_i_i)
        )
        
        return (y_t_t, y_t_i), (y_i_t, y_i_i)


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len_t = 5
    seq_len_i = 7
    evidence_len_t = 6
    evidence_len_i = 8
    embed_dim = 768

    # Create random embeddings
    text_claim = torch.randn(batch_size, seq_len_t, embed_dim)
    image_claim = torch.randn(batch_size, seq_len_i, embed_dim)
    text_evidence = torch.randn(batch_size, evidence_len_t, embed_dim)
    image_evidence = torch.randn(batch_size, evidence_len_i, embed_dim)

    # Build model
    model = MisinformationDetectionModel(
        embed_dim=embed_dim,
        num_heads=8,
        dropout=0.1,
        hidden_dim=256,
        num_classes=3
    )

    # Forward pass (multimodal)
    (y_t_t, y_t_i), (y_i_t, y_i_i) = model(
        X_t=text_claim,
        X_i=image_claim,
        E_t=text_evidence,
        E_i=image_evidence
    )
    print("Text-Text logits:", y_t_t.shape)      # [B, 3]
    print("Text-Image logits:", y_t_i.shape)     # [B, 3]
    print("Image-Text logits:", y_i_t.shape)     # [B, 3]
    print("Image-Image logits:", y_i_i.shape)    # [B, 3]

    # Forward pass (unimodal text)
    (y_t_t, y_t_i), (y_i_t, y_i_i) = model(
        X_t=text_claim,
        E_t=text_evidence
    )
    print("\nUnimodal Text:")
    print("Text-Text logits:", y_t_t.shape if y_t_t is not None else None)
    print("Text-Image logits:", y_t_i if y_t_i is not None else None)
    print("Image-Text logits:", y_i_t if y_i_t is not None else None)
    print("Image-Image logits:", y_i_i if y_i_i is not None else None)