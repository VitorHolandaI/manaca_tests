from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from colher_codigo_ptbr import MARCADORES_FORTES, aceitar, filtrar_shard, marcadores_pt

CODIGO_PT = (
    "def obterUsuario(senha):\n"
    "    # retorna o nome do usuario quando a senha nao bate\n"
    "    return lista_de_dados[valor]\n"
)
CODIGO_EN = (
    "def get_user(password):\n    # return the user name as it is\n    return data\n"
)

precisa_do_pyarrow = pytest.mark.skipif(
    not importlib.util.find_spec("pyarrow"), reason="precisa de pyarrow"
)


def test_marcadores_quebram_camel_case_e_snake_case() -> None:
    achados = marcadores_pt(CODIGO_PT)
    assert {"obter", "usuario", "senha", "retorna", "nao", "lista", "dados"} <= achados


def test_palavras_que_colidem_com_ingles_nao_sao_marcadores() -> None:
    assert not {"do", "as", "os", "com", "se"} & MARCADORES_FORTES
    assert marcadores_pt(CODIGO_EN) == set()


def test_aceitar_exige_python_licenca_permissiva_e_portugues() -> None:
    assert aceitar("Python", "mit", CODIGO_PT, minimo=5)
    assert not aceitar("Python", "gpl-3.0", CODIGO_PT, minimo=5)
    assert not aceitar("Java", "mit", CODIGO_PT, minimo=5)
    assert not aceitar("Python", "mit", CODIGO_EN, minimo=5)


def test_aceitar_recusa_arquivo_gigante() -> None:
    assert not aceitar("Python", "mit", CODIGO_PT * 5000, minimo=5)


@precisa_do_pyarrow
def test_filtrar_shard_devolve_documentos_do_corpus(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    tabela = pa.table(
        {
            "language": ["Python", "Python", "Python"],
            "license": ["mit", "gpl-3.0", "mit"],
            "code": [CODIGO_PT, CODIGO_PT, CODIGO_EN],
            "repo_name": ["a/pt", "b/gpl", "c/en"],
            "path": ["x.py", "y.py", "z.py"],
        }
    )
    parquet = tmp_path / "shard.parquet"
    pq.write_table(tabela, parquet)

    docs = filtrar_shard(parquet, minimo=5)

    assert docs == [
        {
            "text": CODIGO_PT,
            "origem": "codigo_ptbr",
            "caminho": "a/pt/x.py",
            "licenca": "mit",
        }
    ]
