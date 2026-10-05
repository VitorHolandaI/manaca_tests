from __future__ import annotations

import pytest
from caixa_marcada import MAI, MAJ
from train_qlora import (
    RESPOSTA_MARCA,
    garantir_pad_distinto,
    marca_de_resposta,
    textos_de_treino,
)


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


class TokenizerV3Falso:
    eos_token = "</s>"

    def __init__(self, marcado: bool) -> None:
        self.vocab = {"a": 0, MAJ: 1, MAI: 2} if marcado else {"a": 0}

    def get_vocab(self) -> dict[str, int]:
        return self.vocab


def test_textos_de_treino_codificam_prompt_e_resposta_no_v3() -> None:
    linhas = [{"instruction": "Crie a rota", "output": "class Item(Base):"}]
    texto = textos_de_treino(linhas, TokenizerV3Falso(marcado=True))[0]  # type: ignore[arg-type]
    assert f"{MAJ}crie a rota" in texto
    assert f"class{MAJ} item({MAJ}base):</s>" in texto
    assert marca_de_resposta(TokenizerV3Falso(marcado=True)) in texto  # type: ignore[arg-type]


def test_sem_marcador_o_texto_e_o_de_sempre() -> None:
    linhas = [{"instruction": "Crie a rota", "output": "class Item(Base):"}]
    texto = textos_de_treino(linhas, TokenizerV3Falso(marcado=False))[0]  # type: ignore[arg-type]
    assert "class Item(Base):</s>" in texto
    assert marca_de_resposta(TokenizerV3Falso(marcado=False)) == RESPOSTA_MARCA  # type: ignore[arg-type]


class TokenizerV4Falso(TokenizerV3Falso):
    def __init__(self) -> None:
        from caixa_marcada import COLA

        self.vocab = {"a": 0, MAJ: 1, MAI: 2, COLA: 3}


def test_eos_fica_cru_no_fim_com_a_fronteira() -> None:
    # regressao: no v4 o `</s>` virava `</<|cola|> s>`, o EOS sumia do treino
    # e o modelo nao parava de gerar (compile_rate 0,0 em 2026-10-04)
    linhas = [{"instruction": "Crie a rota", "output": "x = 1}"}]
    texto = textos_de_treino(linhas, TokenizerV4Falso())[0]  # type: ignore[arg-type]
    assert texto.endswith("}</s>")
