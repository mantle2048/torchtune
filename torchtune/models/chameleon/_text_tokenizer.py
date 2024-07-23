from tokenizers import Tokenizer


class ChameleonTextTokenizer:
    def __init__(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(
        self, text: str, add_special_tokens: bool = False, **kwargs
    ) -> list[int]:
        return self.tokenizer.encode(
            text, add_special_tokens=add_special_tokens, **kwargs
        ).ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to strings.

        Args:
            ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(ids)
