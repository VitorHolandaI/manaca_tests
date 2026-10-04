#!/usr/bin/env python3
"""Colhe prosa pt de qualidade do GigaVerbo-v2 (Polygl0t/gigaverbo-v2, edu_high).

O CPT do tokenizer v3 precisa de muito texto com maiuscula para o modelo
aprender quando por `<|maj|>`. O GigaVerbo-v2 e o corpus do Tucano 2: cada
documento traz `edu_score` e `toxic_score` (1 a 5, nota de um LLM juiz) e
`token_count`. Licenca por fonte, sem licenca unica: cada documento leva a
sua fonte no campo `licenca`, e o manifest do corpus soma por fonte.

O edu_high tem 56 shards de ~1,3 GB. Le --shards deles, espalhados, sorteia
os row groups dentro de cada um (o shard pode vir agrupado por fonte) e para
no teto de tokens. Cada shard e baixado num diretorio temporario e apagado.

Uso:
    python scripts/colher_gigaverbo.py --saida data/colhido/gigaverbo.jsonl \\
        --max-tokens 65000000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

REPO = "Polygl0t/gigaverbo-v2"
CONFIG = "edu_high"
N_SHARDS = 56
COLUNAS = ["text", "source", "id", "edu_score", "toxic_score", "token_count"]
MINIMO_EDU_PADRAO = 3.5
MAXIMO_TOXICO_PADRAO = 1.5


def escolher_shards(total: int, quantos: int) -> list[int]:
    """Ate `quantos` indices espalhados em range(total).

    Exemplo: escolher_shards(56, 4) == [0, 14, 28, 42]
    """
    if quantos < 1:
        raise ValueError(f"--shards invalido: {quantos}, esperado >= 1")
    if quantos >= total:
        return list(range(total))
    return [i * total // quantos for i in range(quantos)]


def aceitar(edu: float, toxico: float, minimo_edu: float, maximo_toxico: float) -> bool:
    """Documento educativo o bastante e nao toxico."""
    return edu >= minimo_edu and toxico <= maximo_toxico


def documentos_do_shard(
    parquet: Path,
    orcamento: int,
    minimo_edu: float,
    maximo_toxico: float,
    rng: random.Random,
) -> tuple[list[dict[str, str]], int, int]:
    """(documentos aceitos, tokens usados, documentos vistos) ate o orcamento."""
    import pyarrow.parquet as pq

    arquivo = pq.ParquetFile(parquet)
    grupos = list(range(arquivo.num_row_groups))
    rng.shuffle(grupos)
    docs: list[dict[str, str]] = []
    usados = vistos = 0
    for grupo in grupos:
        linhas = arquivo.read_row_group(grupo, columns=COLUNAS).to_pylist()
        for linha in linhas:
            if usados >= orcamento:
                return docs, usados, vistos
            vistos += 1
            if not aceitar(
                linha["edu_score"], linha["toxic_score"], minimo_edu, maximo_toxico
            ):
                continue
            docs.append(
                {
                    "text": linha["text"],
                    "origem": "prosa_gigaverbo",
                    "caminho": f"gigaverbo-v2/{linha['id']}",
                    "licenca": f"gigaverbo:{linha['source']}",
                }
            )
            usados += int(linha["token_count"])
    return docs, usados, vistos


def colher(
    saida: Path, max_tokens: int, shards: int, minimo_edu: float, maximo_toxico: float
) -> tuple[int, int]:
    """Escreve os documentos em `saida` (sobrescreve); devolve (docs, tokens)."""
    from huggingface_hub import hf_hub_download

    if max_tokens < 1:
        raise ValueError(f"--max-tokens invalido: {max_tokens}, esperado >= 1")
    escolhidos = escolher_shards(N_SHARDS, shards)
    por_shard = -(-max_tokens // len(escolhidos))
    rng = random.Random(1234)
    saida.parent.mkdir(parents=True, exist_ok=True)
    total_docs = total_tokens = 0
    with saida.open("w", encoding="utf-8") as fh:
        for indice in escolhidos:
            nome = f"{CONFIG}/train-{indice:05d}-of-{N_SHARDS:05d}.parquet"
            with tempfile.TemporaryDirectory(prefix="giga_") as pasta:
                caminho = hf_hub_download(
                    REPO, nome, repo_type="dataset", local_dir=pasta
                )
                docs, usados, vistos = documentos_do_shard(
                    Path(caminho), por_shard, minimo_edu, maximo_toxico, rng
                )
            for doc in docs:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total_docs += len(docs)
            total_tokens += usados
            print(
                f"[giga] shard {indice}: {len(docs)}/{vistos} aceitos; "
                f"total {total_docs} docs, {total_tokens} tokens",
                file=sys.stderr,
            )
    return total_docs, total_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description="Colhe prosa pt do GigaVerbo-v2")
    ap.add_argument("--saida", default="data/colhido/gigaverbo.jsonl")
    ap.add_argument("--max-tokens", type=int, default=65_000_000)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--minimo-edu", type=float, default=MINIMO_EDU_PADRAO)
    ap.add_argument("--maximo-toxico", type=float, default=MAXIMO_TOXICO_PADRAO)
    args = ap.parse_args()
    docs, tokens = colher(
        Path(args.saida),
        args.max_tokens,
        args.shards,
        args.minimo_edu,
        args.maximo_toxico,
    )
    print(
        f"[giga] fim: {docs} docs, {tokens} tokens (contagem do GigaVerbo)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
