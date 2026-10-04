#!/usr/bin/env python3
"""Constroi o tokenizer v4: o original do Manaca + <|maj|>, <|mai|>, <|cola|>.

No v3 a palavra colada (`criar_agendamento`) era outra peca que a da
instrucao (`▁agendamento`), e perdia para `agenda`, entidade do treino. Com
`<|cola|>` (caixa_marcada, fronteira=True) toda palavra vai na forma com
espaco; caixa e espaco eram as duas fontes de formas duplicadas (Land e
Meister, arXiv 2608.08847). As pecas nuas do v2/v3 deixam de ser precisas.

Uso:
    python scripts/build_tokenizer_v4.py --destino models/manaca-tokenizer-v4
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from caixa_marcada import COLA, MAI, MAJ, codificar
from expandir_vocab import montar_pasta_tokenizer_v2

if TYPE_CHECKING:
    from tokenizers import Tokenizer

MARCADORES_V4 = (MAJ, MAI, COLA)


def construir(original_json: Path) -> Tokenizer:
    """O tokenizer original com os 3 marcadores no fim (ids 64000-64002)."""
    from tokenizers import AddedToken, Tokenizer

    tk = Tokenizer.from_file(str(original_json))
    # normalized=False: o Lowercase nao pode tocar no marcador; special=False:
    # o decode com skip_special_tokens tem de manter o marcador
    tk.add_tokens(
        [AddedToken(m, normalized=False, special=False) for m in MARCADORES_V4]
    )
    return tk


def _conferir(tk: Tokenizer) -> None:
    texto = codificar("def criar_agendamento(dados: AgendamentoCriar):", True)
    ids = tk.encode(texto, add_special_tokens=False).ids
    devolvido = tk.decode(ids, skip_special_tokens=False)
    if devolvido != texto:
        raise ValueError(f"round-trip do v4 falhou: {devolvido!r}, esperado {texto!r}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Constroi o tokenizer v4 (caixa + fronteira)"
    )
    ap.add_argument("--origem", default="models/manaca-hf-official")
    ap.add_argument("--destino", default="models/manaca-tokenizer-v4")
    args = ap.parse_args()

    origem = Path(args.origem)
    tk = construir(origem / "tokenizer.json")
    _conferir(tk)
    with tempfile.TemporaryDirectory() as tmp:
        arquivo = Path(tmp) / "tokenizer_v4.json"
        tk.save(str(arquivo))
        destino = montar_pasta_tokenizer_v2(origem, arquivo, Path(args.destino))
    ids = ", ".join(f"{m}={tk.token_to_id(m)}" for m in MARCADORES_V4)
    print(f"[v4] {tk.get_vocab_size()} tokens ({ids}) -> {destino}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
