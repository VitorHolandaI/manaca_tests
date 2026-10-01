from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from colher_wikipedia_ptbr import LICENCA, MIN_CARACTERES_ARTIGO, artigos_do_shard

pytestmark = pytest.mark.skipif(
    not importlib.util.find_spec("pyarrow"), reason="precisa de pyarrow"
)


def _shard(tmp_path: Path, textos: list[str]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    urls = [f"https://pt.wikipedia.org/wiki/{i}" for i in range(len(textos))]
    caminho = tmp_path / "wiki.parquet"
    pq.write_table(pa.table({"url": urls, "text": textos}), caminho)
    return caminho


def test_pula_esboco_e_marca_licenca(tmp_path: Path) -> None:
    longo = "O Brasil é um país. " * MIN_CARACTERES_ARTIGO
    docs, usados = artigos_do_shard(_shard(tmp_path, ["curto", longo]), 10**9)
    assert [d["caminho"] for d in docs] == ["https://pt.wikipedia.org/wiki/1"]
    assert docs[0]["licenca"] == LICENCA
    assert usados == len(longo)


def test_para_no_orcamento(tmp_path: Path) -> None:
    longo = "x" * MIN_CARACTERES_ARTIGO
    docs, _ = artigos_do_shard(_shard(tmp_path, [longo] * 5), orcamento=len(longo) * 2)
    assert len(docs) == 2
