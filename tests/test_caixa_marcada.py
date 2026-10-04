from __future__ import annotations

import pytest
from caixa_marcada import MAI, MAJ, codificar, decodificar


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("class Comentario(Base):", f"class{MAJ} comentario({MAJ}base):"),
        ("HTTPException", f"{MAI}http{MAJ}exception"),
        ("from fastapi import FastAPI", f"from fastapi import{MAJ} fast{MAI}api"),
        ("ComentarioCriar", f"{MAJ}comentario{MAJ}criar"),
        ("iPhone", f"i{MAJ}phone"),
        ("A casa", f"{MAJ}a casa"),
        ("ÁGUA e Ética", f"{MAI}água e{MAJ} ética"),
        ("deletar_comentario", "deletar_comentario"),
        ("Base64", f"{MAJ}base64"),
    ],
)
def test_codificar_poe_o_marcador_antes_do_espaco(texto: str, esperado: str) -> None:
    assert codificar(texto) == esperado


@pytest.mark.parametrize(
    "texto",
    [
        "class Comentario(Base):\n    __tablename__ = 'comentarios'\n",
        "raise HTTPException(status_code=404, detail='Comentario nao encontrado')",
        "a  B\n\tC  DE fG hIJ kLm",
        "ABİstanbul İzmir straße STRASSE ǅemal",
        "XMLs JSONParser ABCª Nº 5",
        "",
        "   ",
        "O Brasil é o maior país da América do Sul.",
    ],
)
def test_ida_e_volta_e_exata(texto: str) -> None:
    assert decodificar(codificar(texto)) == texto


def test_ida_e_volta_em_todo_o_dataset() -> None:
    from gen_dataset import exemplos_teste_amplo, gerar_exemplos

    for exemplo in gerar_exemplos(1234) + exemplos_teste_amplo(1234):
        for campo in ("instruction", "output"):
            assert decodificar(codificar(exemplo[campo])) == exemplo[campo]


def test_codificar_recusa_texto_que_ja_tem_marcador() -> None:
    with pytest.raises(ValueError, match="posicao 4"):
        codificar(f"abc {MAJ}x")


def test_decodificar_ignora_marcador_solto_do_modelo() -> None:
    # o modelo pode gerar marcador no fim ou antes de pontuacao
    assert decodificar(f"x{MAJ}") == "x"
    assert decodificar(f"{MAI}({MAJ} ") == "( "


class TokenizerFalso:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self.vocab


def test_usa_caixa_marcada_olha_o_vocabulario() -> None:
    from caixa_marcada import usa_caixa_marcada

    assert usa_caixa_marcada(TokenizerFalso({"a": 0, MAJ: 1, MAI: 2}))
    assert not usa_caixa_marcada(TokenizerFalso({"a": 0, "A": 1}))


def test_codificar_documentos_descarta_e_conta_o_que_tem_marcador() -> None:
    from caixa_marcada import codificar_documentos

    textos, descartados = codificar_documentos(["Oi", f"x {MAJ}", "Tchau"])
    assert textos == [f"{MAJ}oi", f"{MAJ}tchau"]
    assert descartados == 1
