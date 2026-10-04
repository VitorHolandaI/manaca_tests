from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest
from colher_gigaverbo import aceitar, documentos_do_shard, escolher_shards

precisa_do_pyarrow = pytest.mark.skipif(
    not importlib.util.find_spec("pyarrow"), reason="precisa de pyarrow"
)


def test_escolher_shards_espalha_pelo_conjunto() -> None:
    assert escolher_shards(56, 4) == [0, 14, 28, 42]
    assert escolher_shards(3, 10) == [0, 1, 2]


def test_aceitar_exige_educativo_e_nao_toxico() -> None:
    assert aceitar(edu=4.2, toxico=1.0, minimo_edu=3.5, maximo_toxico=1.5)
    assert not aceitar(edu=3.0, toxico=1.0, minimo_edu=3.5, maximo_toxico=1.5)
    assert not aceitar(edu=4.5, toxico=2.0, minimo_edu=3.5, maximo_toxico=1.5)


def _shard(tmp_path: Path, n: int) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    tabela = pa.table(
        {
            "text": [f"Texto {i}." for i in range(n)],
            "source": ["fonte_a"] * n,
            "id": [str(i) for i in range(n)],
            "edu_score": [4.0 if i % 2 == 0 else 2.0 for i in range(n)],
            "toxic_score": [1.0] * n,
            "token_count": [10] * n,
        }
    )
    caminho = tmp_path / "giga.parquet"
    pq.write_table(tabela, caminho, row_group_size=2)
    return caminho


@precisa_do_pyarrow
def test_documentos_do_shard_filtra_conta_e_marca_a_fonte(tmp_path: Path) -> None:
    docs, tokens, vistos = documentos_do_shard(
        _shard(tmp_path, 8), 10**9, 3.5, 1.5, random.Random(0)
    )
    assert vistos == 8
    assert tokens == 40
    assert sorted(d["caminho"] for d in docs) == [
        "gigaverbo-v2/0",
        "gigaverbo-v2/2",
        "gigaverbo-v2/4",
        "gigaverbo-v2/6",
    ]
    assert {d["licenca"] for d in docs} == {"gigaverbo:fonte_a"}


@precisa_do_pyarrow
def test_documentos_do_shard_para_no_orcamento(tmp_path: Path) -> None:
    docs, tokens, _ = documentos_do_shard(
        _shard(tmp_path, 8), 20, 3.5, 1.5, random.Random(0)
    )
    assert len(docs) == 2
    assert tokens == 20
