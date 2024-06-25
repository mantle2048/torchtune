from typing import List

from tokenizers import Tokenizer


class ChameleonTextTokenizer:

    def __init__(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to strings.

        Args:
            ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(ids)
