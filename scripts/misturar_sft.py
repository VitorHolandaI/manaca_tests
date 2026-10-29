#!/usr/bin/env python3
"""Amostra o mix de instrucao do LNCC por fonte, para caber no SFT da 3060.

O `sft/prepare_data.py` do Manaca-1B-Instruct grava um JSONL por fonte
(alpaca, aya, oasst, translation, summarization; 103.843 exemplos com os
filtros da v2, sem o JASTER). Inteiro, com os 7.932 de FastAPI, seriam ~30 h
de SFT. Aqui cada fonte entra com um teto sorteado com semente fixa; o LNCC
pegaria os N primeiros (`--max_per_source`), o que pode nao representar a
fonte. Teto 0 = a fonte inteira.

Uso:
    python scripts/misturar_sft.py --entrada data/sft_lncc \\
        --tetos alpaca=6000 aya=3000 oasst=0 translation=2000 summarization=1500 \\
        --saida data/sft_geral.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

TETOS_PADRAO = (
    "alpaca=6000",
    "aya=3000",
    "oasst=0",
    "translation=2000",
    "summarization=1500",
)


def amostrar(
    linhas: list[dict[str, str]], teto: int, semente: int
) -> list[dict[str, str]]:
    """Ate `teto` linhas sorteadas, na ordem original; teto >= total devolve tudo.

    Exemplo: len(amostrar(linhas, 10, semente=42)) == 10
    """
    if teto <= 0 or teto >= len(linhas):
        return list(linhas)
    escolhidos = sorted(random.Random(semente).sample(range(len(linhas)), teto))
    return [linhas[i] for i in escolhidos]


def ler_tetos(pares: list[str]) -> dict[str, int]:
    """`["alpaca=6000", ...]` -> {"alpaca": 6000, ...}."""
    tetos: dict[str, int] = {}
    for par in pares:
        nome, sep, valor = par.partition("=")
        if not sep or not valor.isdigit():
            raise ValueError(
                f"teto invalido: {par!r}, esperado fonte=numero (ex.: alpaca=6000)"
            )
        tetos[nome] = int(valor)
    return tetos


def _ler_jsonl(caminho: Path) -> list[dict[str, str]]:
    if not caminho.is_file():
        raise FileNotFoundError(
            f"fonte nao encontrada: {caminho}; rode o prepare_data.py do LNCC"
        )
    # split("\n"): splitlines() quebra no \u2028 cru de dentro das strings
    texto = caminho.read_text(encoding="utf-8")
    return [json.loads(linha) for linha in texto.split("\n") if linha.strip()]


def misturar(
    pasta: Path, tetos: dict[str, int], semente: int
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """(linhas de todas as fontes, quantas de cada), uma fonte depois da outra."""
    linhas: list[dict[str, str]] = []
    contagem: dict[str, int] = {}
    for nome, teto in tetos.items():
        escolhidas = amostrar(_ler_jsonl(pasta / f"{nome}.jsonl"), teto, semente)
        contagem[nome] = len(escolhidas)
        linhas += escolhidas
    return linhas, contagem


def main() -> int:
    ap = argparse.ArgumentParser(description="Amostra o mix de SFT do LNCC por fonte")
    ap.add_argument("--entrada", default="data/sft_lncc")
    ap.add_argument("--tetos", nargs="+", default=list(TETOS_PADRAO))
    ap.add_argument("--saida", default="data/sft_geral.jsonl")
    ap.add_argument("--semente", type=int, default=42)
    args = ap.parse_args()

    linhas, contagem = misturar(Path(args.entrada), ler_tetos(args.tetos), args.semente)
    saida = Path(args.saida)
    saida.write_text(
        "".join(json.dumps(linha, ensure_ascii=False) + "\n" for linha in linhas),
        encoding="utf-8",
    )
    manifest = {
        "fontes": contagem,
        "total": len(linhas),
        "semente": args.semente,
        "origem": args.entrada,
    }
    saida.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[mix] {contagem} -> {len(linhas)} exemplos em {saida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
