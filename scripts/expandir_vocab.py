#!/usr/bin/env python3
"""Leva o Manaca-1B do vocabulario oficial (64000) para o tokenizer v2.

Duas pecas que o treino precisa antes de ver texto cased:

  1. uma pasta que o AutoTokenizer carrega como v2, conferida depois de
     montada: tamanho do vocab e round-trip de 'class Item(BaseModel):'.
  2. o resize das embeddings de entrada e saida, com cada linha nova iniciada
     pela media das pecas que o tokenizer antigo usava para a mesma superficie.

Uso:
    python scripts/expandir_vocab.py \\
        --origem models/manaca-hf-official \\
        --tokenizer models/manaca-hf-official/tokenizer_v2.json \\
        --destino models/manaca-tokenizer-v2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from tokenizers import Tokenizer
    from transformers import PreTrainedModel

ESPACO_SP = "▁"
ARQUIVOS_COPIADOS = ("tokenizer_config.json", "special_tokens_map.json")


def montar_pasta_tokenizer_v2(
    origem: Path, tokenizer_json: Path, destino: Path
) -> Path:
    """Pasta carregavel por AutoTokenizer com o v2 como tokenizer.json.

    Exemplo: montar_pasta_tokenizer_v2(Path("models/manaca-hf-official"),
    Path(".../tokenizer_v2.json"), Path("models/manaca-tokenizer-v2"))
    """
    exigidos = [origem / nome for nome in ARQUIVOS_COPIADOS] + [tokenizer_json]
    for exigido in exigidos:
        if not exigido.is_file():
            raise FileNotFoundError(f"arquivo exigido para o tokenizer v2: {exigido}")
    destino.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS_COPIADOS:
        shutil.copyfile(origem / nome, destino / nome)
    shutil.copyfile(tokenizer_json, destino / "tokenizer.json")
    return destino


def pecas_antigas(velho: Tokenizer, peca: str) -> list[int]:
    """Ids do tokenizer antigo para a superficie de uma peca nova.

    O antigo prefixa '▁' em tudo e baixa a caixa, entao 'A' vira [id('▁a')]. Um
    '▁' solto na frente e artefato desse prefixo e sai da media.
    """
    codificado = velho.encode(peca.replace(ESPACO_SP, " "), add_special_tokens=False)
    ids = list(codificado.ids)
    if len(ids) > 1 and codificado.tokens[0] == ESPACO_SP:
        ids = ids[1:]
    return ids


def mapa_de_pecas(velho: Tokenizer, novo: Tokenizer) -> dict[int, list[int]]:
    """{id novo: ids antigos} para todo id >= tamanho do vocabulario antigo."""
    n_antigo, n_novo = velho.get_vocab_size(), novo.get_vocab_size()
    if n_novo <= n_antigo:
        raise ValueError(
            f"vocabulario novo ({n_novo}) nao e maior que o antigo ({n_antigo}); "
            "esperado o tokenizer v2, que so faz append"
        )
    divergentes = [
        i for i in range(n_antigo) if velho.id_to_token(i) != novo.id_to_token(i)
    ]
    if divergentes:
        # id antigo com outra peca invalida o embedding treinado daquela linha
        raise ValueError(
            f"{len(divergentes)} ids antigos mudaram de peca (ex.: {divergentes[:5]}); "
            "o v2 tem de ser construido sobre o tokenizer deste modelo"
        )
    mapa: dict[int, list[int]] = {}
    for novo_id in range(n_antigo, n_novo):
        peca = novo.id_to_token(novo_id)
        if peca is None:
            raise ValueError(
                f"id {novo_id} sem peca no tokenizer novo (buraco no vocab)"
            )
        mapa[novo_id] = pecas_antigas(velho, peca)
    return mapa


def inicializar_linhas(
    pesos: torch.Tensor, mapa: dict[int, list[int]], n_antigo: int
) -> None:
    """Escreve em `pesos` (in-place) a media das pecas antigas de cada id novo.

    Peca sem decomposicao cai na media do vocabulario antigo inteiro, que nao
    puxa a distribuicao de saida para nenhum token (Hewitt 2021).
    """
    import torch

    with torch.no_grad():
        media_geral = pesos[:n_antigo].float().mean(dim=0)
        for novo_id, ids in mapa.items():
            linha = pesos[ids].float().mean(dim=0) if ids else media_geral
            pesos[novo_id] = linha.to(pesos.dtype)


def expandir_modelo(modelo: PreTrainedModel, velho_json: Path, novo_json: Path) -> int:
    """Resize + init das embeddings de entrada e saida; devolve linhas novas.

    Modelo que ja tem o tamanho do v2 (checkpoint do pre-treino continuado)
    passa intacto e devolve 0. Linhas de padding entre o vocab antigo e o
    tamanho da matriz (o base tem 64128 para 64000 pecas) sao sobrescritas.
    """
    from tokenizers import Tokenizer

    velho, novo = (
        Tokenizer.from_file(str(velho_json)),
        Tokenizer.from_file(str(novo_json)),
    )
    atual = modelo.get_input_embeddings().weight.shape[0]
    n_antigo, n_novo = velho.get_vocab_size(), novo.get_vocab_size()
    if atual >= n_novo:
        return 0
    if atual < n_antigo:
        raise ValueError(
            f"embedding do modelo tem {atual} linhas; esperado >= {n_antigo} "
            f"(oficial, com ou sem padding) ou {n_novo} (ja expandido)"
        )
    mapa = mapa_de_pecas(velho, novo)
    modelo.resize_token_embeddings(n_novo, mean_resizing=False)
    inicializar_linhas(modelo.get_input_embeddings().weight, mapa, n_antigo)
    saida = modelo.get_output_embeddings()
    if saida is not None and not modelo.config.tie_word_embeddings:
        inicializar_linhas(saida.weight, mapa, n_antigo)
    return len(mapa)


def _conferir_carregamento(destino: Path, esperado: int) -> None:
    """Falha alto se o transformers carregar outra coisa que nao o v2."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(destino)
    if len(tok) != esperado:
        raise RuntimeError(
            f"{destino} carregou com {len(tok)} tokens; esperado {esperado} (v2). "
            "Confira o tokenizer_class do tokenizer_config.json."
        )
    frase = "class Item(BaseModel):"
    volta = tok.decode(tok(frase, add_special_tokens=False).input_ids)
    if volta != frase:
        raise RuntimeError(f"round-trip quebrado: {frase!r} -> {volta!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Monta a pasta do tokenizer v2")
    ap.add_argument("--origem", default="models/manaca-hf-official")
    ap.add_argument(
        "--tokenizer", default="models/manaca-hf-official/tokenizer_v2.json"
    )
    ap.add_argument("--destino", default="models/manaca-tokenizer-v2")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    destino = montar_pasta_tokenizer_v2(
        Path(args.origem), Path(args.tokenizer), Path(args.destino)
    )
    esperado = Tokenizer.from_file(args.tokenizer).get_vocab_size()
    _conferir_carregamento(destino, esperado)
    print(
        f"tokenizer v2 em {destino}: {esperado} tokens, round-trip ok", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
