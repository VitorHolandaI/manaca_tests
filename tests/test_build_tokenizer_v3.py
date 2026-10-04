from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from build_tokenizer_v3 import construir, pecas_v3
from caixa_marcada import MAI, MAJ, codificar

RAIZ = Path(__file__).resolve().parent.parent
ORIGINAL_JSON = RAIZ / "models" / "manaca-hf-official" / "tokenizer.json"
V2_JSON = RAIZ / "models" / "manaca-tokenizer-v2" / "tokenizer.json"

precisa_dos_tokenizers = pytest.mark.skipif(
    not (
        ORIGINAL_JSON.is_file()
        and V2_JSON.is_file()
        and importlib.util.find_spec("tokenizers")
    ),
    reason="precisa do tokenizer original, do v2 e da lib tokenizers",
)

ORIGINAL = [("<unk>", 0.0), ("▁a", -1.0), ("▁comentario", -2.0)]


def test_pecas_v3_fica_com_as_originais_e_as_nuas_minusculas() -> None:
    v2 = ORIGINAL + [
        ("A", -6.0),
        ("▁Comentario", -9.0),
        ("Comentario", -9.0),
        ("comentario", -2.0),
    ]
    assert pecas_v3(ORIGINAL, v2) == ORIGINAL + [("comentario", -2.0)]


def test_pecas_v3_recusa_v2_que_mudou_o_prefixo() -> None:
    v2 = [("<unk>", 0.0), ("▁b", -1.0), ("▁comentario", -2.0), ("x", -3.0)]
    with pytest.raises(ValueError, match="mudou"):
        pecas_v3(ORIGINAL, v2)


@precisa_dos_tokenizers
def test_v3_tem_os_marcadores_no_fim_e_nenhuma_peca_com_maiuscula() -> None:
    tk = construir(ORIGINAL_JSON, V2_JSON)
    n = tk.get_vocab_size()
    assert tk.token_to_id(MAJ) == n - 2
    assert tk.token_to_id(MAI) == n - 1
    pecas = [tk.id_to_token(i) for i in range(64000, n - 2)]
    assert pecas and all(p == p.lower() for p in pecas)


@precisa_dos_tokenizers
def test_classe_usa_a_mesma_peca_da_instrucao() -> None:
    tk = construir(ORIGINAL_JSON, V2_JSON)
    instrucao = tk.encode("um comentario", add_special_tokens=False)
    classe = tk.encode(codificar("class Comentario(Base):"), add_special_tokens=False)
    assert instrucao.ids[-1] in classe.ids
    assert MAJ in classe.tokens


@precisa_dos_tokenizers
def test_v3_devolve_o_texto_codificado_do_dataset() -> None:
    from gen_dataset import gerar_exemplos

    tk = construir(ORIGINAL_JSON, V2_JSON)
    for exemplo in gerar_exemplos(1234)[:200]:
        texto = codificar(exemplo["output"])
        ids = tk.encode(texto, add_special_tokens=False).ids
        assert tk.decode(ids, skip_special_tokens=False) == texto
