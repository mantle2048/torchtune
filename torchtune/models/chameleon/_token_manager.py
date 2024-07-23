import base64
import io
import json
from typing import Optional

import torch
from PIL import Image

from torchtune.data import Message
from torchtune.data._utils import truncate
from torchtune.models.chameleon._image_tokenizer import ChameleonImageTokenizer
from torchtune.models.chameleon._text_tokenizer import ChameleonTextTokenizer
from torchtune.models.chameleon._vocab import VocabInfo, VocabTranslation


class ChameleonTokenManager:
    def __init__(
        self,
        tokenizer_path: str,
        vqgan_cfg_path: str,
        vqgan_ckpt_path: str,
        device: str | None = None,
    ):
        self.text_tokenizer = ChameleonTextTokenizer(tokenizer_path)
        self.vocab = VocabInfo(json.load(open(tokenizer_path))["model"]["vocab"])
        self.translation = VocabTranslation(self.vocab, device=device)
        self.image_tokenizer = ChameleonImageTokenizer(
            cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device=device
        )

        # During generation, stop when eos_id is encountered
        self.stop_tokens = [self.vocab.eos_id]

    def pil_from_bpe_tokens(self, bpe_tokens: torch.Tensor) -> Image.Image:
        image_tensor = self.translation.convert_bpe2img(bpe_tokens)
        if image_tensor.shape[0] < 1024:
            padding = (
                torch.ones(
                    [1024 - image_tensor.shape[0]],
                    dtype=torch.int,
                    device=image_tensor.device,
                )
                * image_tensor[0]
            )
            image_tensor = torch.cat((image_tensor, padding)).unsqueeze(0)

        return self.image_tokenizer.pil_from_img_toks(image_tensor)

    def png_from_bpe_tokens(self, bpe_tokens: torch.Tensor) -> bytes:
        pil = self.pil_from_bpe_tokens(bpe_tokens)
        img_io = io.BytesIO()
        pil.save(img_io, format="PNG")
        return img_io.getvalue()

    def tokenize_text(self, text: str) -> list[int]:
        return self.text_tokenizer.encode(text)

    def tokenize_image(self, img: Image.Image) -> list[int]:
        return (
            [self.vocab.begin_image]
            + self.translation.convert_img2bp2(
                self.image_tokenizer.img_tokens_from_pil(img)
            ).tolist()
            + [self.vocab.end_image]
        )

    def tokenize_b64img(self, b64img: str) -> list[int]:
        image_data = base64.b64decode(b64img)
        image_file = io.BytesIO(image_data)
        return self.tokenize_image(Image.open(image_file))

    def tokens_from_ui(self, inputs: list[dict[str, str]]) -> list[int]:
        tokens = [self.vocab.bos_id]
        for input_ in inputs:
            if input_["type"] == "text":
                tokens += self.tokenize_text(input_["value"].strip())
            elif input_["type"] == "image":
                if isinstance(input_["value"], str):
                    if input_["value"].startswith("data:"):
                        # Value Format: 'data:image/[^;]+;base64,[A-Za-z0-9+/]+={0,2}'
                        tokens += self.tokenize_b64img(input_["value"].split(",", 1)[1])
                    elif input_["value"].startswith("file:"):
                        tokens += self.tokenize_image(
                            Image.open(input_["value"].split(":", 1)[1])
                        )
                    else:
                        raise ValueError("Unknown image format.")
                elif isinstance(input_["value"], Image):
                    tokens += self.tokenize_image(input_["value"])
                else:
                    raise ValueError("Unknown image type.")
            elif input_["type"] == "sentinel":
                tokens += [
                    {
                        "<START-OF-IMAGE>": self.vocab.begin_image,
                        "<END-OF-TURN>": self.vocab.eot_id,
                    }[input_["value"]]
                ]
            else:
                raise ValueError("Unknown input type.")
        return tokens

    def encode(self, inputs: list[dict[str, str]]) -> list[int]:
        tokens = [self.vocab.bos_id]
        for input_ in inputs:
            if input_["type"] == "text":
                tokens += self.tokenize_text(input_["content"].strip())
            elif input_["type"] == "image":
                if isinstance(input_["content"], str):
                    if input_["content"].startswith("data:"):
                        # Value Format: 'data:image/[^;]+;base64,[A-Za-z0-9+/]+={0,2}'
                        tokens += self.tokenize_b64img(
                            input_["content"].split(",", 1)[1]
                        )
                    elif input_["content"].startswith("file:"):
                        tokens += self.tokenize_image(
                            Image.open(input_["content"].split(":", 1)[1])
                        )
                    else:
                        raise ValueError("Unknown image format.")
                elif isinstance(input_["content"], Image):
                    tokens += self.tokenize_image(input_["content"])
                else:
                    raise ValueError("Unknown image type.")
            elif input_["type"] == "sentinel":
                tokens += [
                    {
                        "<START-OF-IMAGE>": self.vocab.begin_image,
                        "<END-OF-TURN>": self.vocab.eot_id,
                    }[input_["content"]]
                ]
            else:
                raise ValueError("Unknown input type.")
        return tokens

    def decode_text(self, ids: torch.LongTensor | list[list[int]]) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        for row, values in enumerate(ids):
            try:
                ids[row] = values[: values.index(self.vocab.eos_id)]
            except ValueError:
                pass

        return [self.text_tokenizer.decode(sample) for sample in ids]

    def decode_image(self, ids: torch.LongTensor) -> list[Image.Image]:
        return [self.pil_from_bpe_tokens(sample) for sample in ids]

    def decode(
        self,
        list_of_tokens: list[list[int]],
        device: torch.device,
    ) -> tuple[list[str], list[Image.Image]]:
        texts, images = [], []
        for tokens in list_of_tokens:
            text_tokens, image_tokens = [], []
            for token in tokens:
                if token in self.vocab.image_tokens:
                    image_tokens.append(token)
                else:
                    text_tokens.append(token)
            if text_tokens:
                texts.append(self.text_tokenizer.decode(text_tokens))
            if image_tokens:
                images.append(
                    self.pil_from_bpe_tokens(torch.tensor(image_tokens, device=device))
                )
        texts = [text for text in texts if text]
        return texts, images

    def tokenize_messages(
        self, messages: list[Message], max_seq_len: Optional[int] = None
    ) -> tuple[list[int], list[bool]]:
        start_of_turn = True
        end_of_turn = False
        tokenized_messages = []
        mask = []
        for message in messages:
            # If assistant message, this is the end of a turn
            end_of_turn = message.role == "assistant"

            # Prepend BOS on start of new turns
            if start_of_turn:
                tokenized_messages.append(self.vocab.bos_id)
                mask.append(message.masked)

            # Tokenize current message, append with masks
            tokens = self.encode(message.content)
            tokenized_messages.extend(tokens)
            mask.extend([message.masked] * len(tokens))

            # If assistant message, append EOS at end
            if end_of_turn:
                tokenized_messages.append(self.vocab.eos_id)
                mask.append(message.masked)
                end_of_turn = False
                start_of_turn = True
            else:
                start_of_turn = False

            # Break out early if we reach max_seq_len
            if max_seq_len and len(tokenized_messages) >= max_seq_len:
                break

        # Finally, truncate if necessary
        if max_seq_len:
            tokenized_messages = truncate(
                tokenized_messages, max_seq_len, self.vocab.eos_id
            )
            mask = truncate(mask, max_seq_len, message.masked)

        return tokenized_messages, mask
