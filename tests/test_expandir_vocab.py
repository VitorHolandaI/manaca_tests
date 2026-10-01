from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from expandir_vocab import (
    ARQUIVOS_COPIADOS,
    expandir_modelo,
    inicializar_linhas,
    mapa_de_pecas,
    montar_pasta_tokenizer_v2,
    pecas_antigas,
)

RAIZ = Path(__file__).resolve().parent.parent
OFICIAL = RAIZ / "models" / "manaca-hf-official"
VELHO_JSON = OFICIAL / "tokenizer.json"
NOVO_JSON = OFICIAL / "tokenizer_v2.json"

precisa_dos_tokenizers = pytest.mark.skipif(
    not (
        VELHO_JSON.is_file()
        and NOVO_JSON.is_file()
        and importlib.util.find_spec("tokenizers")
    ),
    reason="precisa do tokenizer.json, do tokenizer_v2.json e da lib tokenizers",
)
precisa_do_torch = pytest.mark.skipif(
    not importlib.util.find_spec("torch"), reason="precisa de torch"
)


def _tokenizers():  # type: ignore[no-untyped-def]
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(VELHO_JSON)), Tokenizer.from_file(str(NOVO_JSON))


def test_montar_pasta_copia_config_e_poe_o_v2_como_tokenizer_json(
    tmp_path: Path,
) -> None:
    origem = tmp_path / "origem"
    origem.mkdir()
    for nome in ARQUIVOS_COPIADOS:
        (origem / nome).write_text("{}", encoding="utf-8")
    v2 = tmp_path / "v2.json"
    v2.write_text('{"v2": true}', encoding="utf-8")

    destino = montar_pasta_tokenizer_v2(origem, v2, tmp_path / "destino")

    assert (destino / "tokenizer.json").read_text(encoding="utf-8") == '{"v2": true}'
    assert all((destino / nome).is_file() for nome in ARQUIVOS_COPIADOS)


def test_montar_pasta_sem_config_falha_com_o_caminho(tmp_path: Path) -> None:
    v2 = tmp_path / "v2.json"
    v2.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="tokenizer_config.json"):
        montar_pasta_tokenizer_v2(tmp_path, v2, tmp_path / "destino")


@precisa_dos_tokenizers
def test_maiuscula_nova_herda_a_minuscula_antiga() -> None:
    velho, _ = _tokenizers()
    assert pecas_antigas(velho, "A") == [velho.token_to_id("▁a")]


@precisa_dos_tokenizers
def test_espaco_solto_do_prefixo_sai_da_media() -> None:
    velho, _ = _tokenizers()
    ids = pecas_antigas(velho, "Ä")
    assert ids
    assert velho.token_to_id("▁") not in ids


@precisa_dos_tokenizers
def test_mapa_cobre_todo_id_novo() -> None:
    velho, novo = _tokenizers()
    mapa = mapa_de_pecas(velho, novo)
    assert sorted(mapa) == list(range(velho.get_vocab_size(), novo.get_vocab_size()))
    assert all(ids for ids in mapa.values())


@precisa_dos_tokenizers
def test_mapa_recusa_vocab_que_nao_cresceu() -> None:
    velho, _ = _tokenizers()
    with pytest.raises(ValueError, match="nao e maior"):
        mapa_de_pecas(velho, velho)


@precisa_do_torch
def test_inicializar_linhas_usa_a_media_das_pecas() -> None:
    import torch

    pesos = torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    inicializar_linhas(pesos, {3: [0, 1], 4: []}, n_antigo=3)
    assert pesos[3].tolist() == [2.0, 1.0]
    # sem decomposicao: media do vocab antigo inteiro
    assert pesos[4].tolist() == [3.0, 2.0]


class ConfigFalsa:
    tie_word_embeddings = False


class ModeloFalso:
    """Imita a superficie de PreTrainedModel que expandir_modelo usa."""

    def __init__(self, linhas: int, dim: int = 4) -> None:
        import torch

        self.config = ConfigFalsa()
        self.entrada = torch.nn.Embedding(linhas, dim)
        self.saida = torch.nn.Linear(dim, linhas, bias=False)

    def get_input_embeddings(self):  # type: ignore[no-untyped-def]
        return self.entrada

    def get_output_embeddings(self):  # type: ignore[no-untyped-def]
        return self.saida

    def resize_token_embeddings(self, novo: int, mean_resizing: bool) -> None:
        import torch

        assert not mean_resizing  # as linhas novas sao sobrescritas logo depois
        antiga = self.entrada.weight.data
        self.entrada = torch.nn.Embedding(novo, antiga.shape[1])
        self.entrada.weight.data[: antiga.shape[0]] = antiga
        antiga_saida = self.saida.weight.data
        self.saida = torch.nn.Linear(antiga.shape[1], novo, bias=False)
        self.saida.weight.data[: antiga_saida.shape[0]] = antiga_saida


@precisa_dos_tokenizers
@precisa_do_torch
def test_expandir_modelo_inicia_entrada_e_saida() -> None:
    velho, novo = _tokenizers()
    modelo = ModeloFalso(velho.get_vocab_size())
    id_a = velho.token_to_id("▁a")
    id_novo_a = novo.token_to_id("A")

    novas = expandir_modelo(modelo, VELHO_JSON, NOVO_JSON)  # type: ignore[arg-type]

    assert novas == novo.get_vocab_size() - velho.get_vocab_size()
    entrada, saida = modelo.entrada.weight, modelo.saida.weight
    assert entrada.shape[0] == novo.get_vocab_size()
    assert entrada[id_novo_a].tolist() == entrada[id_a].tolist()
    assert saida[id_novo_a].tolist() == saida[id_a].tolist()


@precisa_dos_tokenizers
@precisa_do_torch
def test_expandir_modelo_aceita_matriz_com_padding() -> None:
    # regressao: o manaca-1b-base tem 64128 linhas para 64000 pecas, e a
    # checagem exata derrubou o smoke do CPT na 3090
    velho, novo = _tokenizers()
    modelo = ModeloFalso(velho.get_vocab_size() + 128)
    novas = expandir_modelo(modelo, VELHO_JSON, NOVO_JSON)  # type: ignore[arg-type]
    assert novas == novo.get_vocab_size() - velho.get_vocab_size()
    assert modelo.entrada.weight.shape[0] == novo.get_vocab_size()


@precisa_dos_tokenizers
@precisa_do_torch
def test_expandir_modelo_ja_expandido_nao_mexe() -> None:
    _, novo = _tokenizers()
    modelo = ModeloFalso(novo.get_vocab_size())
    assert expandir_modelo(modelo, VELHO_JSON, NOVO_JSON) == 0  # type: ignore[arg-type]


@precisa_dos_tokenizers
@precisa_do_torch
def test_expandir_modelo_com_tamanho_estranho_falha() -> None:
    modelo = ModeloFalso(1000)
    with pytest.raises(ValueError, match="tem 1000 linhas"):
        expandir_modelo(modelo, VELHO_JSON, NOVO_JSON)  # type: ignore[arg-type]


class TokenizerFalso:
    """So o que mapa_de_pecas le antes de codificar: tamanho e peca por id."""

    def __init__(self, pecas: list[str]) -> None:
        self.pecas = pecas

    def get_vocab_size(self) -> int:
        return len(self.pecas)

    def id_to_token(self, i: int) -> str:
        return self.pecas[i]


def test_mapa_recusa_id_antigo_que_mudou_de_peca() -> None:
    velho = TokenizerFalso(["<unk>", "▁a", "▁b"])
    novo = TokenizerFalso(["<unk>", "▁a", "▁x", "A"])
    with pytest.raises(
        ValueError, match=r"1 ids antigos mudaram de peca \(ex.: \[2\]\)"
    ):
        mapa_de_pecas(velho, novo)  # type: ignore[arg-type]
