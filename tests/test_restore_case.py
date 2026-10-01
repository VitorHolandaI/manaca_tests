from __future__ import annotations

import pytest
from restore_case import nomes_a_restaurar, restaurar


def test_nome_de_modulo_no_from_nao_e_restaurado() -> None:
    codigo = "from fastapi import fastapi, httpexception, depends\n"
    assert restaurar(codigo) == "from fastapi import FastAPI, HTTPException, Depends\n"


def test_modulo_pontilhado_preserva_caminho_e_restaura_nome() -> None:
    codigo = "from sqlalchemy.orm import sessionmaker, session\n"
    assert restaurar(codigo) == "from sqlalchemy.orm import sessionmaker, Session\n"


def test_import_simples_nao_restaura_modulo() -> None:
    assert restaurar("import fastapi\n") == "import fastapi\n"


def test_nome_depois_do_import_volta_a_ser_restaurado() -> None:
    codigo = "from fastapi import fastapi\napp = fastapi()\n"
    assert restaurar(codigo) == "from fastapi import FastAPI\napp = FastAPI()\n"


def test_strings_e_comentarios_ficam_intactos() -> None:
    codigo = 'x = "true"  # none\ny = true\n'
    assert restaurar(codigo) == 'x = "true"  # none\ny = True\n'


def test_nomes_a_restaurar_ignora_modulo() -> None:
    codigo = "from fastapi import fastapi\n"
    assert [t.string for t in nomes_a_restaurar(codigo)] == ["fastapi"]
    assert nomes_a_restaurar(codigo)[0].start == (1, 20)


@pytest.mark.parametrize(
    "quebrado", ["def f(:\n    x = (\n", "if x:\n        a\n    b\n"]
)
def test_codigo_nao_tokenizavel_vira_value_error(quebrado: str) -> None:
    with pytest.raises(ValueError, match="nao tokenizavel"):
        restaurar(quebrado)
