#!/usr/bin/env python3
"""Colhe prosa pt da Wikipedia (wikimedia/wikipedia, dump 20231101.pt).

A prosa e o que ensina maiuscula em portugues -- inicio de frase, nome
proprio, `Comentário` -- que o codigo nao cobre. Licenca CC BY-SA 3.0: o
manifest registra isso para a decisao de distribuicao do modelo.

Le os shards em ordem e para no teto de caracteres. Cada shard e baixado para
um diretorio temporario e apagado depois (~200-400 MB por vez).

Uso:
    python scripts/colher_wikipedia_ptbr.py --saida data/colhido/wiki_ptbr.jsonl \\
        --max-caracteres 60000000
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = "wikimedia/wikipedia"
CONFIG = "20231101.pt"
N_SHARDS = 6
LICENCA = "cc-by-sa-3.0"
# verbete curto e quase sempre esboco ou desambiguacao: pouca frase de verdade
MIN_CARACTERES_ARTIGO = 1500


def artigos_do_shard(parquet: Path, orcamento: int) -> tuple[list[dict[str, str]], int]:
    """(documentos, caracteres usados) ate estourar o orcamento."""
    import pyarrow.parquet as pq

    tabela = pq.read_table(parquet, columns=["url", "text"])
    documentos: list[dict[str, str]] = []
    usados = 0
    for url, texto in zip(
        tabela.column("url").to_pylist(), tabela.column("text").to_pylist()
    ):
        if usados >= orcamento:
            break
        if len(texto) < MIN_CARACTERES_ARTIGO:
            continue
        documentos.append(
            {"text": texto, "origem": "prosa_wiki", "caminho": url, "licenca": LICENCA}
        )
        usados += len(texto)
    return documentos, usados


def colher(saida: Path, max_caracteres: int) -> tuple[int, int]:
    """Escreve os artigos em `saida` (sobrescreve); devolve (docs, caracteres)."""
    from huggingface_hub import hf_hub_download

    if max_caracteres < 1:
        raise ValueError(f"--max-caracteres invalido: {max_caracteres}, esperado >= 1")
    saida.parent.mkdir(parents=True, exist_ok=True)
    total_docs = total_chars = 0
    with saida.open("w", encoding="utf-8") as fh:
        for indice in range(N_SHARDS):
            if total_chars >= max_caracteres:
                break
            nome = f"{CONFIG}/train-{indice:05d}-of-{N_SHARDS:05d}.parquet"
            with tempfile.TemporaryDirectory(prefix="wiki_") as pasta:
                caminho = hf_hub_download(
                    REPO, nome, repo_type="dataset", local_dir=pasta
                )
                docs, usados = artigos_do_shard(
                    Path(caminho), max_caracteres - total_chars
                )
            for doc in docs:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total_docs += len(docs)
            total_chars += usados
            print(
                f"[wiki] shard {indice}: {total_docs} artigos, {total_chars} chars",
                file=sys.stderr,
            )
    return total_docs, total_chars


def main() -> int:
    ap = argparse.ArgumentParser(description="Colhe prosa pt da Wikipedia")
    ap.add_argument("--saida", default="data/colhido/wiki_ptbr.jsonl")
    ap.add_argument("--max-caracteres", type=int, default=60_000_000)
    args = ap.parse_args()
    docs, chars = colher(Path(args.saida), args.max_caracteres)
    print(
        f"[wiki] fim: {docs} artigos, {chars} caracteres ({LICENCA})", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
