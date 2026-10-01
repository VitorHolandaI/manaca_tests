from __future__ import annotations

import ast

from gen_dataset import COMANDOS, ENTIDADES, montar_saida
from sandbox_gerado import (
    payload_da_entrada,
    rodar_rotas,
    rotas_declaradas,
    rotas_do_verbo,
)

TODAS_OPS = tuple(op for op, _, _ in COMANDOS)


def _saida(indice: int = 0, ops: tuple[str, ...] = TODAS_OPS) -> str:
    return montar_saida(ENTIDADES[indice], ops, True)


def test_payload_da_entrada_monta_a_partir_dos_campos() -> None:
    codigo = (
        "from pydantic import BaseModel\n"
        "class usuariocriar(BaseModel):\n"
        "    nome: str\n"
        "    idade: int\n"
        "    ativo: bool\n"
    )
    assert payload_da_entrada(codigo) == {"nome": "x", "idade": 1, "ativo": True}


def test_payload_da_entrada_acha_a_classe_cased() -> None:
    # regressao: endswith("criar") nao via UsuarioCriar, o POST ia sem corpo e
    # o 422 passava como sucesso (so 5xx reprova)
    assert payload_da_entrada(_saida(ops=("criar",))) == {
        "nome": "x",
        "email": "x",
        "ativo": True,
    }


def test_payload_da_entrada_sem_classe_de_entrada_vira_none() -> None:
    assert payload_da_entrada("x = 1\n") is None


def test_payload_da_entrada_com_tipo_desconhecido_vira_none() -> None:
    assert payload_da_entrada("class xcriar(BaseModel):\n    valor: Decimal\n") is None


def test_rotas_do_verbo_pega_o_caminho() -> None:
    codigo = '@app.post("/itens", status_code=201)\ndef criar():\n    pass\n'
    assert rotas_do_verbo(codigo, "post") == ["/itens"]
    assert rotas_do_verbo(codigo, "get") == []


def test_rotas_declaradas_junta_verbo_e_caminho() -> None:
    codigo = '@app.get("/a")\ndef f():\n    pass\n@app.delete("/b")\ndef g():\n    pass\n'
    assert rotas_declaradas(codigo) == [("get", "/a"), ("delete", "/b")]


def test_rodar_rotas_aceita_o_app_completo() -> None:
    passou, motivo = rodar_rotas(_saida())
    assert passou, motivo


def test_rodar_rotas_aceita_saida_de_uma_operacao_so() -> None:
    """O dataset gera UMA operacao por exemplo: uma saida de delete nao tem POST.

    A primeira versao do sandbox exigia o CRUD inteiro e reprovava 8 das 32
    saidas por "sem rota POST" -- medindo a premissa errada, nao o modelo.
    """
    passou, motivo = rodar_rotas(_saida(ops=("deletar",)))
    assert passou, motivo
    assert "post" not in [verbo for verbo, _ in rotas_declaradas(_saida(ops=("deletar",)))]


def test_rodar_rotas_pega_nome_trocado_que_compila() -> None:
    """O caso real: a classe vira `comentar` e as referencias ficam `comentario`.

    Isso passa no ast.parse e so quebra em runtime, com NameError -- exatamente
    o defeito que o marker_rate acusa e o compile_rate nao. E o motivo de
    existir o --executar.
    """
    original = _saida()
    trocado = original.replace("class Usuario(Base):", "class Usuariox(Base):", 1)
    assert trocado != original
    ast.parse(trocado)  # compila: o compile_rate nao veria nada
    passou, motivo = rodar_rotas(trocado)
    assert not passou
    assert "NameError" in (motivo or "")


def test_rodar_rotas_reprova_codigo_sem_rota() -> None:
    passou, motivo = rodar_rotas("x = 1\n")
    assert not passou
    assert "nenhuma rota" in (motivo or "")
