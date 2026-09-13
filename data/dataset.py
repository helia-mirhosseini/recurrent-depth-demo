import torch
from torch.utils.data import Dataset
import pandas as pd


# ============================================================
# Tokenizer
# ============================================================

class ParenthesesTokenizer:
    """
    Very small tokenizer specifically for the parentheses task.

    Vocabulary:

        PAD = 0
        CLS = 1
        (   = 2
        )   = 3

    Example:

        (()())

    becomes:

        [CLS, (, (, ), (, ), )]

    or numerically:

        [1, 2, 2, 3, 2, 3, 3]
    """

    PAD_TOKEN_ID = 0
    CLS_TOKEN_ID = 1
    OPEN_TOKEN_ID = 2
    CLOSE_TOKEN_ID = 3

    def __init__(self, max_length=129):
        """
        max_length includes the CLS token.

        Dataset sequences can be as long as 128 characters,
        so we need:

            128 characters + 1 CLS token = 129
        """
        self.max_length = max_length

        self.token_to_id = {
            "[PAD]": self.PAD_TOKEN_ID,
            "[CLS]": self.CLS_TOKEN_ID,
            "(": self.OPEN_TOKEN_ID,
            ")": self.CLOSE_TOKEN_ID,
        }

        self.id_to_token = {
            value: key
            for key, value in self.token_to_id.items()
        }

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def encode(self, sequence):
        """
        Convert a parentheses sequence into token IDs.

        Returns:
            input_ids
            attention_mask
        """

        tokens = [self.CLS_TOKEN_ID]

        for char in sequence:
            if char == "(":
                tokens.append(self.OPEN_TOKEN_ID)

            elif char == ")":
                tokens.append(self.CLOSE_TOKEN_ID)

            else:
                raise ValueError(
                    f"Unexpected character: {char}"
                )

        if len(tokens) > self.max_length:
            raise ValueError(
                f"Sequence is too long. "
                f"Got {len(tokens)} tokens, "
                f"maximum is {self.max_length}."
            )

        # --------------------------------------------------
        # Padding
        # --------------------------------------------------

        padding_length = self.max_length - len(tokens)

        input_ids = (
            tokens
            + [self.PAD_TOKEN_ID] * padding_length
        )

        # 1 = real token
        # 0 = padding
        attention_mask = (
            [1] * len(tokens)
            + [0] * padding_length
        )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
        }


# ============================================================
# Dataset
# ============================================================

class ParenthesesDataset(Dataset):

    def __init__(
        self,
        csv_path,
        tokenizer,
    ):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

        required_columns = {
            "sequence",
            "label",
            "length",
        }

        if not required_columns.issubset(
            self.data.columns
        ):
            raise ValueError(
                f"CSV must contain columns: "
                f"{required_columns}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        sequence = row["sequence"]
        label = int(row["label"])

        encoded = self.tokenizer.encode(sequence)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded[
                "attention_mask"
            ],
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "sequence_length": torch.tensor(
                len(sequence),
                dtype=torch.long,
            ),
        }


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    tokenizer = ParenthesesTokenizer()

    example = "(()())"

    encoded = tokenizer.encode(example)

    print("Sequence:")
    print(example)

    print("\nInput IDs:")
    print(encoded["input_ids"][:15])

    print("\nAttention mask:")
    print(encoded["attention_mask"][:15])

    print("\nVocabulary size:")
    print(tokenizer.vocab_size)