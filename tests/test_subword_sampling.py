from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import subword_sampling
from subword_sampling import (
    AmostradorUnigram,
    ids_com_amostragem,
    metaspace,
    normalizar,
    span_da_resposta,
    span_do_prefixo,
)

RAIZ = Path(__file__).resolve().parent.parent
TOKENIZER_JSON = RAIZ / "models" / "manaca-hf-official" / "tokenizer.json"
DATASET = RAIZ / "data" / "fastapi_ptbr.jsonl"

precisa_do_modelo = pytest.mark.skipif(
    not TOKENIZER_JSON.is_file(), reason="precisa do tokenizer.json do Manaca"
)

VOCAB_MINIMO = json.dumps(
    {"model": {"vocab": [["<unk>", 0.0]], "unk_id": 0, "byte_fallback": False}}
)


def test_metaspace_troca_espaco_pelo_marcador() -> None:
    assert metaspace("a b  c") == "a▁b▁▁c"


def test_normalizar_baixa_sem_mexer_no_espaco() -> None:
    # NFKC + minuscula: "Abaixo  Esta" -> "abaixo  esta" (dois espacos intactos)
    assert normalizar("Abaixo  Esta") == "abaixo  esta"


def test_span_do_prefixo_prepende_o_marcador() -> None:
    assert span_do_prefixo("A b") == "▁a▁b"
    assert span_da_resposta("A b") == "a▁b"


@pytest.mark.parametrize("ruim", [0.0, -0.1, 1.5])
def test_alpha_fora_da_faixa_vira_erro(tmp_path: Path, ruim: float) -> None:
    caminho = tmp_path / "t.json"
    caminho.write_text(VOCAB_MINIMO, encoding="utf-8")
    with pytest.raises(ValueError, match="alpha invalido"):
        AmostradorUnigram(caminho, alpha=ruim)


@precisa_do_modelo
def test_viterbi_reproduz_o_tokenizer() -> None:
    """O portao de sanidade: se isto diverge, os scores ou o byte-fallback erram."""
    tokenizers = pytest.importorskip("tokenizers")
    import train_qlora as tq

    tk = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))
    amostrador = AmostradorUnigram(TOKENIZER_JSON, alpha=0.2)
    linhas = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines()]
    for linha in linhas[:40]:
        texto = tq.montar_texto(linha["instruction"], linha["output"], "</s>")
        assert amostrador.viterbi(span_do_prefixo(texto)) == tk.encode(texto).ids[1:]


@precisa_do_modelo
def test_amostrar_varia_e_decodifica_de_volta() -> None:
    """A propriedade que importa: segmentacoes diferentes, texto identico."""
    tokenizers = pytest.importorskip("tokenizers")

    tk = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))
    amostrador = AmostradorUnigram(TOKENIZER_JSON, alpha=0.2)
    rng = random.Random(0)
    span = span_da_resposta("def deletar_comentario(): return sessao.query(item)")
    esperado = tk.decode(amostrador.viterbi(span), skip_special_tokens=True)
    vistos = set()
    for _ in range(50):
        ids = amostrador.amostrar(span, rng)
        assert ids is not None
        assert tk.decode(ids, skip_special_tokens=True) == esperado
        vistos.add(tuple(ids))
    assert len(vistos) > 10


@precisa_do_modelo
def test_ids_com_amostragem_mantem_o_marcador() -> None:
    """O prefixo e deterministico, entao o marcador nunca se desfaz."""
    pytest.importorskip("tokenizers")
    import train_qlora as tq

    amostrador = AmostradorUnigram(TOKENIZER_JSON, alpha=0.2)
    ids_marca = None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(RAIZ / "models" / "manaca-hf-official"))
    ids_marca = tq.ids_marcador(tok, tq.RESPOSTA_MARCA)
    texto = tq.montar_texto("crie um endpoint", "from fastapi import FastAPI\n", "</s>")
    rng = random.Random(0)
    for _ in range(20):
        ids = ids_com_amostragem(amostrador, texto, tq.RESPOSTA_MARCA, rng)
        assert ids is not None
        assert subword_sampling._indice(ids_marca, ids) is not None


@precisa_do_modelo
def test_ids_com_amostragem_sem_marca_vira_none() -> None:
    pytest.importorskip("tokenizers")
    amostrador = AmostradorUnigram(TOKENIZER_JSON, alpha=0.2)
    assert (
        ids_com_amostragem(
            amostrador, "texto sem marcador", "\n\n### Resposta:\n", random.Random(0)
        )
        is None
    )


def test_exemplo_para_collator_tira_o_text_para_o_collator_do_trl() -> None:
    """Regressao: o collator do trl da pad na linha inteira e estoura com `text`.

    O `DataCollatorForCompletionOnlyLM` recebia a linha com a coluna `text`
    (string) e morria com "too many dimensions 'str'".
    """
    import train_qlora as tq

    linha = {"text": "abc", "input_ids": [1, 2], "attention_mask": [1, 1]}
    assert tq.exemplo_para_collator(object(), linha) == {
        "input_ids": [1, 2],
        "attention_mask": [1, 1],
    }


@precisa_do_modelo
def test_exemplo_para_collator_mantem_o_text_para_o_amostrador() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    import train_qlora as tq
    from transformers import AutoTokenizer

    assert torch is not None
    tok = AutoTokenizer.from_pretrained(str(RAIZ / "models" / "manaca-hf-official"))
    amostrador = AmostradorUnigram(TOKENIZER_JSON, alpha=0.2)
    collator = subword_sampling.CollatorComAmostragem(
        tok, amostrador, tq.ids_marcador(tok, tq.RESPOSTA_MARCA)
    )
    linha = {"text": "abc", "input_ids": [1], "attention_mask": [1]}
    assert tq.exemplo_para_collator(collator, linha) is linha


@precisa_do_modelo
def test_collator_mascara_o_prompt_e_amostra_a_resposta() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    import train_qlora as tq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(RAIZ / "models" / "manaca-hf-official"))
    amostrador = AmostradorUnigram(TOKENIZER_JSON, alpha=0.2)
    collator = subword_sampling.CollatorComAmostragem(
        tok, amostrador, tq.ids_marcador(tok, tq.RESPOSTA_MARCA)
    )
    texto = tq.montar_texto("crie um endpoint", "from fastapi import FastAPI\n", "</s>")
    batch = collator([{"text": texto}, {"text": texto}])

    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    com_loss = (batch["labels"][0] != -100).sum().item()
    mascarados = (batch["labels"][0] == -100).sum().item()
    assert mascarados > 0, "o prompt tem de ser mascarado"
    assert com_loss > 0, "a resposta nao pode ser mascarada (loss viraria NaN)"
    assert collator.sem_caminho == 0
    assert isinstance(batch["input_ids"], torch.Tensor)
