#!/usr/bin/env python3
"""Constroi o tokenizer v3: o original do Manaca + pecas nuas + 2 marcadores.

O v2 pos a maiuscula dentro das pecas e o modelo nunca ligou `comentario` a
`▁Com enta rio` (entidade_rate 0,32 no teste amplo). O v3 volta ao original,
que baixa a caixa (NFKC + Lowercase), e representa a maiuscula com
`<|maj|>`/`<|mai|>` (ver caixa_marcada.py). Do v2 ficam so as pecas nuas
minusculas (`comentario` sem `▁`), que servem a `deletar_comentario`.

Uso:
    python scripts/build_tokenizer_v3.py --destino models/manaca-tokenizer-v3
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from caixa_marcada import MAI, MAJ, codificar
from expandir_vocab import montar_pasta_tokenizer_v2

if TYPE_CHECKING:
    from tokenizers import Tokenizer

PecaUnigram = tuple[str, float]


def pecas_v3(original: list[PecaUnigram], v2: list[PecaUnigram]) -> list[PecaUnigram]:
    """As pecas originais seguidas das pecas do v2 que nao tem maiuscula.

    Exemplo: pecas_v3(orig, v2) == orig + [("comentario", -9.1), ...]
    """
    prefixo = [p for p, _ in v2[: len(original)]]
    if prefixo != [p for p, _ in original]:
        raise ValueError(
            "o v2 mudou pecas do vocabulario original; esperado o v2 construido "
            "por append sobre este tokenizer (build_tokenizer_v2.py)"
        )
    vistas = {p for p, _ in original}
    nuas = [
        (p, s) for p, s in v2[len(original) :] if p == p.lower() and p not in vistas
    ]
    return original + nuas


def _pecas(
    tokenizer_json: dict[str, dict[str, list[list[str | float]]]],
) -> list[PecaUnigram]:
    return [(str(p), float(s)) for p, s in tokenizer_json["model"]["vocab"]]


def construir(original_json: Path, v2_json: Path) -> Tokenizer:
    """Tokenizer v3 em memoria: vocab de pecas_v3 e os marcadores no fim."""
    from tokenizers import AddedToken, Tokenizer

    original = json.loads(original_json.read_text(encoding="utf-8"))
    v2 = json.loads(v2_json.read_text(encoding="utf-8"))
    novo = copy.deepcopy(original)
    pecas = pecas_v3(_pecas(original), _pecas(v2))
    novo["model"]["vocab"] = [[p, s] for p, s in pecas]
    tk = Tokenizer.from_str(json.dumps(novo))
    # normalized=False: o Lowercase nao pode tocar no marcador; special=False:
    # o decode com skip_special_tokens tem de manter o marcador
    tk.add_tokens([AddedToken(m, normalized=False, special=False) for m in (MAJ, MAI)])
    return tk


def _conferir(tk: Tokenizer) -> None:
    texto = codificar("class Comentario(Base):\n    raise HTTPException(404)")
    ids = tk.encode(texto, add_special_tokens=False).ids
    devolvido = tk.decode(ids, skip_special_tokens=False)
    if devolvido != texto:
        raise ValueError(f"round-trip do v3 falhou: {devolvido!r}, esperado {texto!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Constroi o tokenizer v3 (marcador)")
    ap.add_argument("--origem", default="models/manaca-hf-official")
    ap.add_argument("--v2", default="models/manaca-tokenizer-v2/tokenizer.json")
    ap.add_argument("--destino", default="models/manaca-tokenizer-v3")
    args = ap.parse_args()

    origem = Path(args.origem)
    tk = construir(origem / "tokenizer.json", Path(args.v2))
    _conferir(tk)
    with tempfile.TemporaryDirectory() as tmp:
        arquivo = Path(tmp) / "tokenizer_v3.json"
        tk.save(str(arquivo))
        destino = montar_pasta_tokenizer_v2(origem, arquivo, Path(args.destino))
    print(
        f"[v3] {tk.get_vocab_size()} tokens ({MAJ}={tk.token_to_id(MAJ)}, "
        f"{MAI}={tk.token_to_id(MAI)}) -> {destino}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
