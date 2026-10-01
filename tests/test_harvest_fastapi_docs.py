from __future__ import annotations

import json
from pathlib import Path

import pytest
from harvest_fastapi_docs import (
    cabecalhos,
    codigo_valido,
    colher,
    escrever,
    instrucao,
    limpar_prosa,
    minusculizar,
    resolver,
    secao_antes,
    titulo_do_doc,
)

CODIGO = "from fastapi import FastAPI\n\napp = FastAPI()\n"


def _repo(tmp_path: Path) -> Path:
    """Arvore falsa de repo: um doc pt que cita um codigo em docs_src."""
    docs = tmp_path / "docs" / "pt" / "docs" / "tutorial"
    docs.mkdir(parents=True)
    src = tmp_path / "docs_src" / "first_steps"
    src.mkdir(parents=True)
    (src / "tutorial001.py").write_text(CODIGO, encoding="utf-8")
    (docs / "a.md").write_text(
        "# Primeiros Passos { #first-steps }\n"
        "\n"
        "O arquivo mais simples se parece com:\n"
        "\n"
        "{* ../../docs_src/first_steps/tutorial001.py *}\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def repo_falso(tmp_path: Path) -> Path:
    return _repo(tmp_path)


def test_resolver_normaliza_o_caminho_do_docs_src() -> None:
    assert (
        resolver("docs/pt/docs/tutorial/a.md", "../../docs_src/x/y.py")
        == "docs_src/x/y.py"
    )


def test_resolver_com_caminho_ja_relativo() -> None:
    assert resolver("docs/pt/docs/a.md", "../docs_src/x/y.py") == "docs_src/x/y.py"


def test_titulo_do_doc_tira_a_ancora() -> None:
    linhas = ["# Primeiros Passos { #first-steps }", "", "texto"]
    assert titulo_do_doc(linhas) == "Primeiros Passos"


def test_limpar_prosa_pega_o_ultimo_paragrafo_e_tira_marcacao() -> None:
    texto = (
        "Primeiro paragrafo.\n"
        "\n"
        "Segundo com [link](http://x) e `code`.\n"
    )
    assert limpar_prosa(texto) == "Segundo com link e `code`."


def test_limpar_prosa_remove_include_e_admonicao() -> None:
    texto = (
        "/// tip | Dica\n"
        "\n"
        "Isso e dica.\n"
        "\n"
        "///\n"
        "\n"
        "Copie para `main.py`:\n"
        "\n"
        "{* ../../docs_src/x/y.py *}\n"
    )
    assert limpar_prosa(texto) == "Copie para `main.py`:"


def test_secao_antes_pega_o_ultimo_cabecalho() -> None:
    linhas = ["# Doc", "## Secao A", "texto", "## Secao B", "texto", "{* x.py *}"]
    assert secao_antes(cabecalhos(linhas), 5) == "Secao B"


def test_instrucao_junta_titulo_secao_e_prosa() -> None:
    assert instrucao("Doc", "Secao", "Faca isso.") == "Doc - Secao. Faca isso."
    assert instrucao("Doc", "", "") == "Doc"
    assert instrucao("", "", "") == "exemplo de fastapi"


def test_codigo_valido() -> None:
    assert codigo_valido("x = 1\n") is True
    assert codigo_valido("def f(:\n") is False


def test_colher_monta_o_par(repo_falso: Path) -> None:
    pares, conta = colher(repo_falso, "pt")
    assert len(pares) == 1
    par = pares[0]
    assert par["output"] == CODIGO
    assert par["instruction"] == "Primeiros Passos. O arquivo mais simples se parece com:"
    assert par["input"] == ""
    assert par["_origem"].endswith("docs_src/first_steps/tutorial001.py")
    assert conta == {
        "docs": 1,
        "citacoes": 1,
        "ausentes": 0,
        "invalidos": 0,
        "duplicados": 0,
    }


def test_colher_ignora_codigo_que_nao_compila(repo_falso: Path) -> None:
    quebrado = repo_falso / "docs_src" / "first_steps" / "tutorial002.py"
    quebrado.write_text("def f(:\n", encoding="utf-8")
    doc = repo_falso / "docs" / "pt" / "docs" / "tutorial" / "a.md"
    doc.write_text(
        doc.read_text(encoding="utf-8") + "\n{* ../../docs_src/first_steps/tutorial002.py *}\n",
        encoding="utf-8",
    )
    pares, conta = colher(repo_falso, "pt")
    assert len(pares) == 1
    assert conta["invalidos"] == 1


def test_colher_conta_citacao_repetida_como_duplicada(repo_falso: Path) -> None:
    doc = repo_falso / "docs" / "pt" / "docs" / "tutorial" / "a.md"
    doc.write_text(
        doc.read_text(encoding="utf-8")
        + "\n{* ../../docs_src/first_steps/tutorial001.py hl[1] *}\n",
        encoding="utf-8",
    )
    pares, conta = colher(repo_falso, "pt")
    assert len(pares) == 1
    assert conta["duplicados"] == 1
    assert conta["citacoes"] == 2


def test_colher_conta_include_ausente(repo_falso: Path) -> None:
    doc = repo_falso / "docs" / "pt" / "docs" / "tutorial" / "a.md"
    doc.write_text(
        doc.read_text(encoding="utf-8") + "\n{* ../../docs_src/nao_existe/x.py *}\n",
        encoding="utf-8",
    )
    pares, conta = colher(repo_falso, "pt")
    assert len(pares) == 1
    assert conta["ausentes"] == 1


def test_colher_sem_o_idioma_vira_erro(repo_falso: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nao achei docs/es/docs"):
        colher(repo_falso, "es")


def test_minusculizar_baixa_instrucao_e_saida(repo_falso: Path) -> None:
    pares, _ = colher(repo_falso, "pt")
    baixos = minusculizar(pares)
    assert baixos[0]["output"] == CODIGO.lower()
    assert baixos[0]["instruction"] == pares[0]["instruction"].lower()


def test_escrever_omite_origem_por_padrao(tmp_path: Path) -> None:
    pares, _ = colher(_repo(tmp_path), "pt")
    destino = tmp_path / "saida.jsonl"
    info = escrever(pares, destino, com_origem=False)
    assert info["exemplos"] == 1
    linha = json.loads(destino.read_text(encoding="utf-8").splitlines()[0])
    assert set(linha) == {"instruction", "input", "output"}


def test_escrever_inclui_origem_quando_pedido(tmp_path: Path) -> None:
    pares, _ = colher(_repo(tmp_path), "pt")
    destino = tmp_path / "saida.jsonl"
    escrever(pares, destino, com_origem=True)
    linha = json.loads(destino.read_text(encoding="utf-8").splitlines()[0])
    assert "_origem" in linha
