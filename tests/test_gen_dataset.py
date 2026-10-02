from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from gen_dataset import (
    ENTIDADES,
    ENTIDADES_EXTRAS,
    ENTIDADES_TESTE,
    ENTIDADES_TESTE_AMPLO,
    EXEMPLOS_POR_EXTRA,
    Entidade,
    capitalizar_frase,
    escrever,
    exemplos_teste_amplo,
    gerar_exemplos,
    montar_saida,
    nome_de_classe,
    separar_teste,
)
from restore_case import nomes_a_restaurar

TODAS_OPS = ("criar", "listar", "buscar", "atualizar", "deletar")


def test_gerador_e_deterministico(tmp_path: Path) -> None:
    a = escrever(gerar_exemplos(1234), tmp_path / "a.jsonl")
    b = escrever(gerar_exemplos(1234), tmp_path / "b.jsonl")
    assert a["sha256"] == b["sha256"]


def test_split_nao_vaza_instrucao_nem_entidade() -> None:
    treino, teste = separar_teste(gerar_exemplos(1234), ENTIDADES_TESTE)
    instrucoes_treino = {r["instruction"] for r in treino}
    assert teste
    assert not instrucoes_treino & {r["instruction"] for r in teste}
    for entidade in ENTIDADES_TESTE:
        assert not any(
            f"class {nome_de_classe(entidade)}(" in r["output"] for r in treino
        )


def test_split_de_teste_tem_as_duas_entidades() -> None:
    # regressao: a marca ficou 'class comentario(base):' depois da saida virar
    # cased, e o split de teste saia vazio sem nenhum erro
    _, teste = separar_teste(gerar_exemplos(1234), ENTIDADES_TESTE)
    for entidade in ENTIDADES_TESTE:
        assert any(
            f"class {nome_de_classe(entidade)}(Base):" in r["output"] for r in teste
        )


def test_nome_de_classe_capitaliza_so_a_primeira() -> None:
    assert nome_de_classe("endereco") == "Endereco"
    assert nome_de_classe("") == ""


def test_capitalizar_frase_corrige_bibliotecas() -> None:
    assert (
        capitalizar_frase("crie uma rota no fastapi com sqlalchemy e pydantic")
        == "Crie uma rota no FastAPI com SQLAlchemy e Pydantic"
    )


def test_saida_ja_nasce_com_caso_de_python() -> None:
    for exemplo in gerar_exemplos(1234):
        assert nomes_a_restaurar(exemplo["output"]) == []
        assert (
            "from fastapi import FastAPI, HTTPException, Depends" in exemplo["output"]
        )


def test_instrucao_comeca_com_maiuscula() -> None:
    assert all(r["instruction"][0].isupper() for r in gerar_exemplos(1234))


def _docstrings_das_rotas(codigo: str) -> list[str | None]:
    funcoes = [
        n
        for n in ast.walk(ast.parse(codigo))
        if isinstance(n, ast.FunctionDef) and n.decorator_list
    ]
    return [ast.get_docstring(f) for f in funcoes]


@pytest.mark.parametrize("op", TODAS_OPS)
def test_toda_rota_tem_docstring_capitalizada(op: str) -> None:
    docs = _docstrings_das_rotas(montar_saida(ENTIDADES[0], (op,), False))
    assert docs
    assert all(d and d[0].isupper() for d in docs)


def test_jsonl_escrito_e_valido(tmp_path: Path) -> None:
    destino = tmp_path / "x.jsonl"
    escrever(gerar_exemplos(1234), destino)
    linhas = destino.read_text(encoding="utf-8").splitlines()
    assert all(json.loads(linha)["output"] for linha in linhas)


@pytest.mark.parametrize("comentar", [True, False])
@pytest.mark.parametrize("ent", ENTIDADES, ids=lambda e: e.singular)
def test_referencia_roda_o_crud_inteiro(
    ent: Entidade, comentar: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app_de_referencia import AppDeReferencia

    monkeypatch.chdir(tmp_path)  # o codigo gerado grava ./app.db
    app = AppDeReferencia(montar_saida(ent, TODAS_OPS, comentar), ent)
    rota = f"/{ent.plural}"

    criado = app.cliente.post(rota, json=app.payload())
    assert criado.status_code == 201, criado.text
    item = f"{rota}/{criado.json()['id']}"
    assert len(app.cliente.get(rota).json()) == 1
    assert app.cliente.get(item).status_code == 200
    assert app.cliente.put(item, json=app.payload()).status_code == 200
    assert app.cliente.delete(item).status_code == 200
    assert app.cliente.get(item).status_code == 404


# sha256 do split de teste medido nas rodadas de 2026-10-01; mudar o treino
# nao pode mudar o teste, senao os numeros deixam de ser comparaveis
SHA_TESTE_MEDIDO = "44be4950f1475730c7b7b8e5751670ae36b17da1a2f58702e6dbe94c816b3ed7"


def _campos_do_modelo(codigo: str) -> tuple[str, ...]:
    classe = next(
        n
        for n in ast.walk(ast.parse(codigo))
        if isinstance(n, ast.ClassDef)
        and any(isinstance(b, ast.Name) and b.id == "Base" for b in n.bases)
    )
    return tuple(
        alvo.id
        for n in classe.body
        if isinstance(n, ast.Assign)
        for alvo in n.targets
        if isinstance(alvo, ast.Name) and alvo.id not in ("id", "__tablename__")
    )


def test_campos_do_treino_variam_dentro_da_mesma_entidade() -> None:
    treino, _ = separar_teste(gerar_exemplos(1234), ENTIDADES_TESTE)
    usuario = [r["output"] for r in treino if "class Usuario(Base):" in r["output"]]
    assert len({_campos_do_modelo(c) for c in usuario}) > 1


def test_split_de_teste_continua_identico_ao_medido(tmp_path: Path) -> None:
    _, teste = separar_teste(gerar_exemplos(1234), ENTIDADES_TESTE)
    assert escrever(teste, tmp_path / "t.jsonl")["sha256"] == SHA_TESTE_MEDIDO


def test_extras_nao_repetem_nem_colidem_com_o_molde() -> None:
    import keyword

    from nomes_entidades import NOMES_EXTRAS

    singulares = [s for s, _ in NOMES_EXTRAS]
    originais = {e.singular for e in ENTIDADES} | {e.plural for e in ENTIDADES}
    assert len(set(singulares)) == len(singulares)
    assert not set(singulares) & originais
    assert not set(singulares) & set(ENTIDADES_TESTE)
    proibidos = {"registro", "sessao", "resposta", "dados", "item_id", "base"}
    assert not any(keyword.iskeyword(s) or s in proibidos for s in singulares)


def test_cada_extra_entra_no_treino_com_poucos_exemplos() -> None:
    treino, _ = separar_teste(gerar_exemplos(1234), ENTIDADES_TESTE)
    for ent in ENTIDADES_EXTRAS[:5]:
        marca = f"class {nome_de_classe(ent.singular)}(Base):"
        assert sum(marca in r["output"] for r in treino) == EXEMPLOS_POR_EXTRA


@pytest.mark.parametrize("ent", ENTIDADES_EXTRAS[::40], ids=lambda e: e.singular)
def test_referencia_de_extra_roda_o_crud_inteiro(
    ent: Entidade, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app_de_referencia import AppDeReferencia

    monkeypatch.chdir(tmp_path)
    app = AppDeReferencia(montar_saida(ent, TODAS_OPS, False), ent)
    rota = f"/{ent.plural}"
    criado = app.cliente.post(rota, json=app.payload())
    assert criado.status_code == 201, criado.text
    assert app.cliente.delete(f"{rota}/{criado.json()['id']}").status_code == 200


def test_teste_amplo_usa_so_nomes_fora_do_treino() -> None:
    from nomes_entidades import NOMES_ENTIDADES, NOMES_EXTRAS, NOMES_TESTE_AMPLO

    treino_nomes = {s for s, _ in NOMES_ENTIDADES + NOMES_EXTRAS}
    assert len(NOMES_TESTE_AMPLO) == 10
    assert not {s for s, _ in NOMES_TESTE_AMPLO} & treino_nomes
    treino, _ = separar_teste(gerar_exemplos(1234), ENTIDADES_TESTE)
    for ent in ENTIDADES_TESTE_AMPLO:
        marca = f"class {nome_de_classe(ent.singular)}("
        assert not any(marca in r["output"] for r in treino)


def test_teste_amplo_contem_o_teste_original_e_doze_entidades() -> None:
    _, teste = separar_teste(gerar_exemplos(1234), ENTIDADES_TESTE)
    amplo = exemplos_teste_amplo(1234)
    assert all(ex in amplo for ex in teste)
    classes = {
        linha
        for ex in amplo
        for linha in ex["output"].split("\n")
        if linha.endswith("(Base):")
    }
    assert len(classes) == 12


@pytest.mark.parametrize("ent", ENTIDADES_TESTE_AMPLO, ids=lambda e: e.singular)
def test_referencia_do_teste_amplo_roda_o_crud(
    ent: Entidade, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app_de_referencia import AppDeReferencia

    monkeypatch.chdir(tmp_path)
    app = AppDeReferencia(montar_saida(ent, TODAS_OPS, True), ent)
    rota = f"/{ent.plural}"
    criado = app.cliente.post(rota, json=app.payload())
    assert criado.status_code == 201, criado.text
    assert app.cliente.get(f"{rota}/{criado.json()['id']}").status_code == 200


def test_cada_colhido_entra_com_poucos_exemplos() -> None:
    from gen_dataset import EXEMPLOS_POR_COLHIDO

    colhidos = (("fatia", "fatias"), ("lagoa", "lagoas"))
    sem = gerar_exemplos(1234)
    com = gerar_exemplos(1234, colhidos=colhidos)
    assert com[: len(sem)] == sem
    for singular, _ in colhidos:
        marca = f"class {nome_de_classe(singular)}(Base):"
        assert sum(marca in r["output"] for r in com) == EXEMPLOS_POR_COLHIDO


@pytest.mark.parametrize("nome", ["comentario", "recibo", "usuario"])
def test_colhido_que_ja_existe_e_recusado(nome: str) -> None:
    with pytest.raises(ValueError, match=nome):
        gerar_exemplos(1234, colhidos=((nome, nome + "s"),))


def test_carregar_nomes_colhidos_le_e_aponta_linha_ruim(tmp_path: Path) -> None:
    from nomes_entidades import carregar_nomes_colhidos

    bom = tmp_path / "bom.tsv"
    bom.write_text("fatia\tfatias\nlagoa\tlagoas\n", encoding="utf-8")
    assert carregar_nomes_colhidos(bom) == (("fatia", "fatias"), ("lagoa", "lagoas"))
    ruim = tmp_path / "ruim.tsv"
    ruim.write_text("fatia\tfatias\nLagoa lagoas\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"ruim\.tsv:2"):
        carregar_nomes_colhidos(ruim)
