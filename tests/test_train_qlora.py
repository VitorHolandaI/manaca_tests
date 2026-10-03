from __future__ import annotations

import pytest
from train_qlora import garantir_pad_distinto


class TokenizerFalso:
    """Imita os atributos de pad/eos de um tokenizer do transformers."""

    def __init__(self, vocab: dict[str, int], eos: str, pad: str | None) -> None:
        self.vocab = vocab
        self.eos_token = eos
        self.pad_token = pad

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def pad_token_id(self) -> int | None:
        return None if self.pad_token is None else self.vocab[self.pad_token]

    def get_added_vocab(self) -> dict[str, int]:
        return self.vocab


def test_pad_igual_ao_eos_vira_token_de_pad_existente() -> None:
    # regressao: Qwen3 base tem pad == eos, e o collator poria -100 no EOS
    vocab = {"<|endoftext|>": 1, "<|vision_pad|>": 3, "<|fim_pad|>": 2}
    tok = TokenizerFalso(vocab, eos="<|endoftext|>", pad="<|endoftext|>")
    assert garantir_pad_distinto(tok) == "<|fim_pad|>"
    assert tok.pad_token_id != tok.eos_token_id


def test_pad_ja_distinto_nao_muda() -> None:
    tok = TokenizerFalso({"</s>": 2, "<pad>": 0}, eos="</s>", pad="<pad>")
    assert garantir_pad_distinto(tok) is None
    assert tok.pad_token == "<pad>"


def test_sem_token_de_pad_disponivel_falha_com_o_eos() -> None:
    tok = TokenizerFalso({"</s>": 2}, eos="</s>", pad=None)
    with pytest.raises(ValueError, match="</s>"):
        garantir_pad_distinto(tok)
