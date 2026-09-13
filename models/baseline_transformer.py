import torch
import torch.nn as nn

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

    Unlike learned positional embeddings, these position vectors
    do not need to be trained individually.

    That makes them much better suited to our experiment because
    we train on lengths 8-32 but test on much longer sequences.
    """

    def __init__(
        self,
        d_model: int,
        max_length: int = 129,
    ):
        super().__init__()

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_length,
            d_model,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Shape:
        # [1, max_length, d_model]
        pe = pe.unsqueeze(0)

        # Not a trainable parameter.
        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):
        """
        x:
            [batch_size, sequence_length, d_model]
        """

        sequence_length = x.size(1)

        return x + self.pe[:, :sequence_length]

class BaselineTransformer(nn.Module):
    """
    Standard encoder-only Transformer baseline.

    Architecture:

        Token embeddings
              ↓
        Positional embeddings
              ↓
        Transformer Layer 1
              ↓
        Transformer Layer 2
              ↓
        Transformer Layer 3
              ↓
        Transformer Layer 4
              ↓
        CLS representation
              ↓
        Linear classifier
              ↓
        Balanced / Unbalanced

    Each Transformer layer has its OWN parameters.
    """

    def __init__(
        self,
        vocab_size=4,
        max_length=129,
        d_model=128,
        num_heads=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length

        # --------------------------------------------------
        # Token embeddings
        # --------------------------------------------------

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )

        # --------------------------------------------------
        # Learnable positional embeddings
        # --------------------------------------------------

        self.position_encoding = SinusoidalPositionalEncoding(
                 d_model=d_model,
                     max_length=max_length,
            )

        # --------------------------------------------------
        # Transformer encoder
        # --------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # --------------------------------------------------
        # Final normalization
        # --------------------------------------------------

        self.final_norm = nn.LayerNorm(d_model)

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------

        self.classifier = nn.Linear(
            d_model,
            num_classes,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):

        batch_size, sequence_length = input_ids.shape

        # --------------------------------------------------
        # Positional indices
        # --------------------------------------------------

        x = self.token_embedding(input_ids)

        x = self.position_encoding(x)       

        # --------------------------------------------------
        # Convert our attention mask into the form expected
        # by PyTorch TransformerEncoder.
        #
        # Our mask:
        #
        #   1 = real token
        #   0 = padding
        #
        # PyTorch padding mask:
        #
        #   False = real token
        #   True  = padding
        # --------------------------------------------------

        padding_mask = None

        if attention_mask is not None:
            padding_mask = attention_mask == 0

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.final_norm(x)

        # --------------------------------------------------
        # Use CLS representation
        #
        # CLS is always token 0 in the sequence.
        # --------------------------------------------------

        cls_representation = x[:, 0, :]

        # --------------------------------------------------
        # Binary classification
        # --------------------------------------------------

        logits = self.classifier(
            cls_representation
        )

        return logits


# ============================================================
# Utilities
# ============================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":

    model = BaselineTransformer()

    print(model)

    print(
        "\nTrainable parameters:",
        f"{count_parameters(model):,}",
    )

    # Fake batch
    batch_size = 8
    sequence_length = 129

    input_ids = torch.randint(
        0,
        4,
        (
            batch_size,
            sequence_length,
        ),
    )

    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long,
    )

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    print("\nOutput shape:")
    print(logits.shape)

    # Expected:
    #
    # torch.Size([8, 2])