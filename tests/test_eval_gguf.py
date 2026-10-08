from __future__ import annotations

from eval_gguf import escolher_por_entidade


def test_escolher_por_entidade_espalha_pelas_entidades() -> None:
    # os primeiros itens do teste amplo sao so de comentario e endereco
    ref = [
        {"instruction": f"pedido {i}", "output": f"class {nome}(Base):\n"}
        for nome in ("Comentario", "Endereco", "Recibo")
        for i in range(5)
    ]
    escolhidos = escolher_por_entidade(ref, por_entidade=2)
    classes = [e["output"].split("(")[0] for e in escolhidos]
    assert (
        classes
        == ["class Comentario"] * 2 + ["class Endereco"] * 2 + ["class Recibo"] * 2
    )
