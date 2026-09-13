import math
import random

import torch
import torch.nn as nn


# ============================================================
# Positional encoding
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Same positional encoding used in the baseline.

    Keeping this identical is important for a fair comparison.
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

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):

        sequence_length = x.size(1)

        return (
            x
            + self.pe[:, :sequence_length]
        )


# ============================================================
# Recurrent-depth Transformer
# ============================================================

class RecurrentTransformer(nn.Module):
    """
    Transformer with:

        1 unique prelude block
        1 shared recurrent block
        1 unique coda block

    During training:

        loop count can be randomly sampled.

    During inference:

        loop count can be specified explicitly.

    Example:

        Prelude
           ↓
        Shared
           ↓
        Shared
           ↓
        Shared
           ↓
        Shared
           ↓
        Coda

    Although Shared is executed four times,
    there is only ONE set of parameters for it.
    """

    def __init__(
        self,
        vocab_size=4,
        max_length=129,
        d_model=128,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=2,
        min_train_loops=1,
        max_train_loops=4,
    ):
        super().__init__()

        self.d_model = d_model

        self.min_train_loops = (
            min_train_loops
        )

        self.max_train_loops = (
            max_train_loops
        )

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )

        # ----------------------------------------------------
        # Same positional encoding as baseline
        # ----------------------------------------------------

        self.position_encoding = (
            SinusoidalPositionalEncoding(
                d_model=d_model,
                max_length=max_length,
            )
        )

        # ----------------------------------------------------
        # PRELUDE
        #
        # Unique parameters.
        # ----------------------------------------------------

        self.prelude = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        # ----------------------------------------------------
        # SHARED RECURRENT BLOCK
        #
        # Defined ONCE.
        #
        # This same block will be called multiple times.
        # ----------------------------------------------------

        self.shared_block = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        # ----------------------------------------------------
        # CODA
        #
        # Another unique block.
        # ----------------------------------------------------

        self.coda = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.final_norm = nn.LayerNorm(
            d_model
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            d_model,
            num_classes,
        )

    # ========================================================
    # Loop-count selection
    # ========================================================

    def choose_loop_count(
        self,
        num_loops=None,
    ):
        """
        Training behavior:

            If no explicit num_loops is supplied,
            randomly choose between 1 and 4.

        Evaluation behavior:

            Caller should supply a fixed number such as
            1, 2, 4, 6, or 8.

        This lets us train one model but test different
        amounts of internal computation.
        """

        if num_loops is not None:

            if num_loops < 1:
                raise ValueError(
                    "num_loops must be >= 1"
                )

            return num_loops

        if self.training:

            return random.randint(
                self.min_train_loops,
                self.max_train_loops,
            )

        # Default evaluation behavior if caller forgets
        # to specify it.
        return self.max_train_loops

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        input_ids,
        attention_mask=None,
        num_loops=None,
        return_hidden_states=False,
    ):

        # ----------------------------------------------------
        # Input embeddings
        # ----------------------------------------------------

        x = self.token_embedding(
            input_ids
        )

        x = self.position_encoding(x)

        # ----------------------------------------------------
        # Padding mask
        #
        # Dataset:
        #   1 = token
        #   0 = padding
        #
        # PyTorch:
        #   False = token
        #   True  = padding
        # ----------------------------------------------------

        padding_mask = None

        if attention_mask is not None:

            padding_mask = (
                attention_mask == 0
            )

        # ----------------------------------------------------
        # Decide number of recurrent passes
        # ----------------------------------------------------

        loops = self.choose_loop_count(
            num_loops
        )

        hidden_states = []

        # ----------------------------------------------------
        # Prelude
        # ----------------------------------------------------

        x = self.prelude(
            x,
            src_key_padding_mask=
                padding_mask,
        )

        if return_hidden_states:
            hidden_states.append(
                x.detach()
            )

        # ----------------------------------------------------
        # Recurrent-depth computation
        #
        # SAME BLOCK, SAME PARAMETERS.
        # ----------------------------------------------------

        for _ in range(loops):

            x = self.shared_block(
                x,
                src_key_padding_mask=
                    padding_mask,
            )

            if return_hidden_states:
                hidden_states.append(
                    x.detach()
                )

        # ----------------------------------------------------
        # Coda
        # ----------------------------------------------------

        x = self.coda(
            x,
            src_key_padding_mask=
                padding_mask,
        )

        if return_hidden_states:
            hidden_states.append(
                x.detach()
            )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.final_norm(x)

        # ----------------------------------------------------
        # CLS representation
        # ----------------------------------------------------

        cls_representation = (
            x[:, 0, :]
        )

        logits = self.classifier(
            cls_representation
        )

        if return_hidden_states:

            return {
                "logits": logits,
                "num_loops": loops,
                "hidden_states":
                    hidden_states,
            }

        return logits


# ============================================================
# Parameter count
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":

    model = RecurrentTransformer()

    print(model)

    print(
        "\nTrainable parameters:",
        f"{count_parameters(model):,}"
    )

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

    # --------------------------------------------------------
    # Explicit four-loop test
    # --------------------------------------------------------

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_loops=4,
    )

    print(
        "\n4-loop output shape:"
    )

    print(output.shape)

    # --------------------------------------------------------
    # Eight-loop test
    #
    # Notice that we can execute 8 loops even though
    # training only uses 1-4 loops.
    # --------------------------------------------------------

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_loops=8,
    )

    print(
        "\n8-loop output shape:"
    )

    print(output.shape)

    # Expected:
    #
    # torch.Size([8, 2])