from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from chat_gguf import corpo_da_requisicao

RAIZ = Path(__file__).resolve().parent.parent
TOKENIZER_V4 = RAIZ / "models" / "manaca-tokenizer-v4"


def test_corpo_manda_ids_greedy_e_pede_os_ids_de_volta() -> None:
    corpo = corpo_da_requisicao([1, 64002, 884], 512)
    assert corpo["prompt"] == [1, 64002, 884]
    assert corpo["n_predict"] == 512
    assert corpo["return_tokens"] is True
    assert corpo["temperature"] == 0.0
    # sem cache: o servidor compararia com o pedido anterior e pularia tokens
    assert corpo["cache_prompt"] is False


class ClienteFalso:
    """Faz o papel do llama-server: devolve ids prontos e guarda o que recebeu."""

    def __init__(self, resposta: list[int]) -> None:
        self.resposta = resposta
        self.recebido: list[int] = []

    def gerar(self, ids: list[int], max_novos: int) -> list[int]:
        self.recebido = ids
        return self.resposta


@pytest.mark.skipif(
    not (TOKENIZER_V4.is_dir() and importlib.util.find_spec("transformers")),
    reason="precisa do tokenizer v4 e do transformers",
)
def test_responder_codifica_o_pedido_e_decodifica_a_saida() -> None:
    from caixa_marcada import codificar_para
    from chat_gguf import responder
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_V4)
    codigo = "class Recibo(Base):\n    pass"
    ids_saida = tok(codificar_para(tok, codigo), add_special_tokens=False)["input_ids"]
    cliente = ClienteFalso(ids_saida + [tok.eos_token_id, 999])

    texto = responder(tok, cliente, "Crie um recibo", 64)

    assert texto == codigo
    assert cliente.recebido[0] == tok.bos_token_id
    assert tok.convert_tokens_to_ids("<|maj|>") in cliente.recebido


@pytest.mark.skipif(
    not (TOKENIZER_V4.is_dir() and importlib.util.find_spec("transformers")),
    reason="precisa do tokenizer v4 e do transformers",
)
def test_responder_usa_o_template_que_receber() -> None:
    from caixa_marcada import codificar_para
    from chat_gguf import responder
    from template_alpaca import LNCC, montar_prompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_V4)
    cliente = ClienteFalso([tok.eos_token_id])
    responder(tok, cliente, "oi", 8, template=LNCC)
    esperado = codificar_para(tok, montar_prompt(LNCC, "oi"))
    assert cliente.recebido == tok(esperado, add_special_tokens=True)["input_ids"]
