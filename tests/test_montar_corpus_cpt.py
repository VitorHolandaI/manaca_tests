from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from montar_corpus_cpt import (
    FONTES,
    Documento,
    ler_extra,
    limpar,
    marcas_de_contaminacao,
    montar,
    separar_val,
)


def _upstream_falso(raiz: Path, extra: dict[str, str] | None = None) -> Path:
    arquivos = {
        "fastapi/applications.py": "class FastAPI:\n    pass\n",
        "docs_src/app/tutorial001.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "docs/pt/docs/index.md": "# Início\n\nO FastAPI é rápido.\n{* ../../docs_src/app/tutorial001.py *}\n",
    }
    arquivos.update(extra or {})
    for rel, texto in arquivos.items():
        destino = raiz / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(texto, encoding="utf-8")
    return raiz


def test_marcas_cobrem_classe_e_instrucao_de_teste() -> None:
    marcas = marcas_de_contaminacao(1234)
    assert "class Comentario" in marcas
    assert "class Endereco" in marcas
    assert "class Recibo" in marcas  # teste amplo
    assert any("comentario" in m for m in marcas if not m.startswith("class "))


def test_limpar_tira_diretiva_so_da_prosa() -> None:
    diretiva = "antes {* ../../docs_src/x.py hl[2] *} depois"
    assert limpar(diretiva, "prosa_ptbr") == "antes  depois"
    assert limpar(diretiva, "codigo") == diretiva


def test_separar_val_e_deterministico_e_nunca_vazio() -> None:
    docs = [Documento("codigo", f"a{i}.py", "x") for i in range(10)]
    treino, val = separar_val(docs, 0.02)
    assert len(val) == 1
    assert separar_val(list(reversed(docs)), 0.02)[1] == val
    assert not {d.caminho for d in treino} & {d.caminho for d in val}


def test_montar_junta_as_tres_fontes_e_mantem_o_caso(tmp_path: Path) -> None:
    treino, val, descartados = montar(_upstream_falso(tmp_path), 1234)
    todos = treino + val
    assert descartados == 0
    assert {d.origem for d in todos} == {f.rotulo for f in FONTES}
    assert any("O FastAPI é rápido." in d.texto for d in todos)
    assert not any("docs_src/app" in d.texto for d in todos if d.origem == "prosa_ptbr")


def test_montar_descarta_documento_contaminado(tmp_path: Path) -> None:
    contaminado = {"docs_src/app/vazou.py": "class Comentario(Base):\n    pass\n"}
    treino, val, descartados = montar(_upstream_falso(tmp_path, contaminado), 1234)
    assert descartados == 1
    assert not any("class Comentario" in d.texto for d in treino + val)


def test_montar_sem_a_pasta_da_fonte_falha_com_o_caminho(tmp_path: Path) -> None:
    raiz = _upstream_falso(tmp_path)
    shutil.rmtree(raiz / "docs_src")
    with pytest.raises(FileNotFoundError, match="docs_src"):
        montar(raiz, 1234)


def _extra(tmp_path: Path, linhas: list[dict[str, str]]) -> Path:
    caminho = tmp_path / "extra.jsonl"
    caminho.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in linhas) + "\n",
        encoding="utf-8",
    )
    return caminho


def test_ler_extra_agrupa_por_origem(tmp_path: Path) -> None:
    extra = _extra(
        tmp_path,
        [
            {
                "text": "def obter(): pass",
                "origem": "codigo_ptbr",
                "caminho": "a",
                "licenca": "mit",
            },
            {
                "text": "O Brasil.",
                "origem": "prosa_wiki",
                "caminho": "b",
                "licenca": "cc-by-sa-3.0",
            },
        ],
    )
    grupos = ler_extra(extra)
    assert sorted(grupos) == ["codigo_ptbr", "prosa_wiki"]
    assert grupos["prosa_wiki"][0].licenca == "cc-by-sa-3.0"


def test_ler_extra_aponta_linha_sem_licenca(tmp_path: Path) -> None:
    extra = _extra(tmp_path, [{"text": "x", "origem": "o", "caminho": "c"}])
    with pytest.raises(ValueError, match=r"extra\.jsonl:1"):
        ler_extra(extra)


def test_montar_com_extra_aplica_guarda_e_val_por_origem(tmp_path: Path) -> None:
    raiz = _upstream_falso(tmp_path / "up")
    linhas = [
        {
            "text": f"def f{i}(): pass",
            "origem": "codigo_ptbr",
            "caminho": f"r/{i}.py",
            "licenca": "mit",
        }
        for i in range(10)
    ] + [
        {
            "text": "class Endereco(Base): pass",
            "origem": "codigo_ptbr",
            "caminho": "r/x.py",
            "licenca": "mit",
        }
    ]
    treino, val, descartados = montar(raiz, 1234, (_extra(tmp_path, linhas),))
    assert descartados == 1
    assert any(d.origem == "codigo_ptbr" for d in val)
    assert sum(d.origem == "codigo_ptbr" for d in treino + val) == 10


def test_ler_extra_aguenta_separador_unicode_dentro_do_texto(tmp_path: Path) -> None:
    # regressao: splitlines() partia a linha no \u2028 cru que o json.dumps
    # (ensure_ascii=False) deixa na string, e a 3090 caiu na linha 3131
    texto = "x = 'a\u2028b\x85c'"
    extra = _extra(
        tmp_path,
        [{"text": texto, "origem": "codigo_ptbr", "caminho": "a", "licenca": "mit"}],
    )
    assert ler_extra(extra)["codigo_ptbr"][0].texto == texto
