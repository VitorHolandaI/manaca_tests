from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from colher_nomes_classe import (
    dobrar_acentos,
    escrever_tsv,
    frequencias_de_palavras,
    ler_textos,
    nomes_de_classe,
    nomes_por_sufixo,
    parece_entidade,
    pluralizar,
    selecionar,
)


@pytest.mark.parametrize(
    ("singular", "plural"),
    [
        ("pedido", "pedidos"),
        ("transacao", "transacoes"),
        ("animal", "animais"),
        ("funil", "funis"),
        ("item", "itens"),
        ("fornecedor", "fornecedores"),
        ("rapaz", "rapazes"),
        ("mes", "meses"),
    ],
)
def test_pluralizar_segue_as_regras_do_portugues(singular: str, plural: str) -> None:
    assert pluralizar(singular) == plural


def test_dobrar_acentos_casa_wikipedia_com_identificador() -> None:
    assert dobrar_acentos("comentário açúcar") == "comentario acucar"


def test_nomes_de_classe_pega_so_palavra_unica_capitalizada() -> None:
    codigo = (
        "class Pedido(Base):\n"
        "    pass\n"
        "class ItemPedido(Base):\n"
        "    pass\n"
        "class Nota:\n"
        "    class Interna(object):\n"
        "        pass\n"
        "x = 'class Falso('\n"
    )
    assert nomes_de_classe(codigo) == ["Pedido", "Nota", "Interna"]


def test_frequencias_de_palavras_dobra_acento_e_caixa() -> None:
    contagem = frequencias_de_palavras(["O Comentário e o comentario."])
    assert contagem["comentario"] == 2


@pytest.mark.parametrize(
    "nome", ["eliminar", "receber", "visitas", "emitido", "object", "music", "fermat"]
)
def test_parece_entidade_recusa_verbo_plural_participio_e_ingles(nome: str) -> None:
    # regressao: a primeira colheita na 3060 (2026-10-02) aceitou todos esses
    assert not parece_entidade(nome)


@pytest.mark.parametrize("nome", ["fatura", "hotel", "colecao", "promotor", "item"])
def test_parece_entidade_aceita_substantivo(nome: str) -> None:
    assert parece_entidade(nome)


def test_nomes_por_sufixo_pega_substantivo_frequente_da_wikipedia() -> None:
    wiki = Counter({"informacao": 50, "cidade": 40, "melhor": 90, "andamento": 5})
    assert nomes_por_sufixo(wiki, minimo=30) == {"informacao", "cidade"}


def test_selecionar_junta_fontes_exclui_e_fica_com_os_mais_frequentes() -> None:
    classes = Counter({"Fatura": 3, "Manager": 9, "Recibo": 2, "Receber": 4})
    wiki = Counter(
        {
            "fatura": 40,
            "manager": 1,
            "recibo": 80,
            "receber": 500,
            "cidade": 300,
            "informacao": 900,
        }
    )
    nomes = selecionar(classes, wiki, minimo_wiki=20, excluidos={"recibo"}, maximo=2)
    assert nomes == [("cidade", "cidades"), ("informacao", "informacoes")]


def test_ler_textos_aguenta_separador_unicode(tmp_path: Path) -> None:
    caminho = tmp_path / "c.jsonl"
    linhas = [{"text": "class A\u2028B:"}, {"text": "class Pedido:"}]
    caminho.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in linhas) + "\n",
        encoding="utf-8",
    )
    assert ler_textos(caminho) == ["class A\u2028B:", "class Pedido:"]


def test_ler_textos_sem_arquivo_falha_com_o_caminho(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nao.jsonl"):
        ler_textos(tmp_path / "nao.jsonl")


def test_escrever_tsv_ordena_e_e_lido_de_volta(tmp_path: Path) -> None:
    destino = tmp_path / "n.tsv"
    escrever_tsv([("recibo", "recibos"), ("fatura", "faturas")], destino)
    assert destino.read_text(encoding="utf-8") == "fatura\tfaturas\nrecibo\trecibos\n"
