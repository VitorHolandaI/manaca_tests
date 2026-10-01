from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval_code import (
    LOTE_PADRAO,
    MAX_NOVOS_TOKENS_PADRAO,
    PREAMBULO,
    _base_do_adapter,
    _gerar_em_lotes,
    _prompt,
    avaliar,
    docstrings_capitalizadas,
    exigencias,
)
from gen_dataset import ENTIDADES_TESTE, gerar_exemplos, separar_teste

REFERENCIA = (
    "from fastapi import fastapi\n"
    "app = fastapi()\n"
    '@app.get("/itens")\n'
    "def listar_itens():\n"
    "    return []\n"
)


def _ref() -> list[dict[str, str]]:
    return [{"instruction": "liste os itens", "output": REFERENCIA}]


def test_referencia_pontua_um() -> None:
    relatorio = avaliar(_ref(), {"liste os itens": REFERENCIA}, None)
    assert relatorio["compile_rate"] == 1.0
    assert relatorio["marker_rate"] == 1.0


def test_saida_nao_tokenizavel_conta_como_falha_sem_derrubar() -> None:
    relatorio = avaliar(
        _ref(), {"liste os itens": "def listar_itens(:\n    x = (\n"}, None
    )
    assert relatorio["compile_rate"] == 0.0
    assert relatorio["falhas"][0]["motivo"].startswith("nao tokenizavel")


def test_modulo_fastapi_nao_conta_como_vazamento() -> None:
    relatorio = avaliar(
        _ref(), {"liste os itens": "from fastapi import FastAPI\n"}, None
    )
    assert relatorio["precisaram_de_restauro"] == 0


def test_orcamento_de_geracao_cobre_a_maior_referencia() -> None:
    # a maior saida do dataset mede 697 tokens; 512 truncava 220/256 exemplos
    assert MAX_NOVOS_TOKENS_PADRAO >= 1024


def test_base_do_adapter_devolve_o_proprio_id_para_modelo_normal() -> None:
    assert _base_do_adapter("AIInstituteLNCC/manaca-1b-base") == (
        "AIInstituteLNCC/manaca-1b-base"
    )


def test_base_do_adapter_le_o_base_de_um_adapter_peft(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "AIInstituteLNCC/manaca-1b-base"}),
        encoding="utf-8",
    )
    assert _base_do_adapter(str(tmp_path)) == "AIInstituteLNCC/manaca-1b-base"


def test_prompt_usa_o_template_alpaca_pt() -> None:
    texto = _prompt("liste os itens")
    assert texto.startswith(PREAMBULO)
    assert texto.endswith("### Instrucao:\nliste os itens\n\n### Resposta:\n")


class LoteFalso(dict):
    """dict com .to(), como o BatchEncoding que o tokenizer real devolve."""

    def to(self, device):
        return self


class TokenizadorFalso:
    """Tokenizador de mentira: cada palavra vira um id (1-based), pad a esquerda."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 0
    padding_side = "right"

    def __call__(self, textos, **kwargs):
        import torch

        sequencias = [[i + 1 for i in range(len(t.split()))] for t in textos]
        largura = max(len(s) for s in sequencias)
        ids, mascara = [], []
        for sequencia in sequencias:
            falta = largura - len(sequencia)
            ids.append([0] * falta + sequencia)
            mascara.append([0] * falta + [1] * len(sequencia))
        return LoteFalso(
            {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(mascara),
            }
        )

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids if int(i) != self.pad_token_id)


class ModeloFalso:
    """Devolve a entrada mais uma continuacao fixa, como o generate faria."""

    CONTINUACAO = (101, 102)

    def generate(self, input_ids, attention_mask, **kwargs):
        import torch

        extra = torch.tensor([self.CONTINUACAO]).expand(input_ids.shape[0], -1)
        return torch.cat([input_ids, extra], dim=1)


def test_geracao_em_lotes_nao_cola_o_prompt_na_saida() -> None:
    """Regressao do bug que sujou o eval de 75 entidades.

    Com padding a esquerda, cortar pela attention_mask.sum() (tamanho SEM pad)
    nao pula o pad, entao os ultimos tokens do prompt entravam na resposta: a
    saida comecava com um pedaco da instrucao e o ast.parse quebrava na linha 1.
    """
    pytest.importorskip("torch", reason="exercita o fatiamento da geracao")
    tok = TokenizadorFalso()
    prompts = ["a b c d e", "a b"]  # tamanhos diferentes -> o 2o leva padding
    saidas = _gerar_em_lotes(ModeloFalso(), tok, prompts, "cpu", 4, 2)
    assert saidas == ["101 102", "101 102"]


def test_lote_padrao_e_maior_que_um() -> None:
    # sem lote o base gasta 1024 tokens por prompt em serie; era ~15 min
    assert LOTE_PADRAO > 1


def test_exigencias_separa_por_categoria() -> None:
    ex = exigencias(REFERENCIA)
    assert ex.rotas == ['@app.get("/itens"']
    assert ex.funcoes == ["def listar_itens("]
    assert ex.status == []


def test_exigencias_pega_o_status_code() -> None:
    assert exigencias('x\n@app.post("/x", status_code=201)\n').status == [
        "status_code=201"
    ]


def test_por_marcador_mostra_que_a_rota_bateu_e_a_funcao_nao() -> None:
    """O caso real: Python valido, rota certa, nome de funcao errado.

    Sem o por_marcador, isso aparece so como marker_rate 0.0 e nao da para
    saber se o modelo errou o nome, a rota ou o status.
    """
    pred = {
        "liste os itens": (
            "from fastapi import fastapi\n"
            "app = fastapi()\n"
            '@app.get("/itens")\n'
            "def outra_coisa():\n"
            "    return []\n"
        )
    }
    relatorio = avaliar(_ref(), pred, None)
    assert relatorio["compile_rate"] == 1.0
    assert relatorio["por_marcador"]["rotas"] == 1.0
    assert relatorio["por_marcador"]["funcoes"] == 0.0
    assert relatorio["marker_rate"] == 0.0


def test_avaliar_com_reparo_conta_os_reparados() -> None:
    """`--reparar` conserta nome usado e nao definido antes de pontuar.

    O caso real: `itensrespost` nao existe, mas `itensresposta` esta definida
    logo acima -- o codigo quebrava em runtime e passa a compilar. O typo tem de
    ser no sufixo: o guarda de prefixo comum (>=4) e o que impede mapear
    qualquer coisa em qualquer coisa.
    """
    pred = {
        "liste os itens": (
            "from fastapi import fastapi\n"
            "app = fastapi()\n"
            "class itensresposta(basemodel):\n"
            "    nome: str\n"
            '@app.get("/itens")\n'
            "def listar_itens():\n"
            "    return itensrespost()\n"
        )
    }
    sem = avaliar(_ref(), pred, None)
    com = avaliar(_ref(), pred, None, reparar_nomes=True)
    assert sem["reparados"] == 0
    assert com["reparados"] == 1
    assert com["compile_rate"] == 1.0


REFERENCIA_CASED = (
    "from fastapi import FastAPI\n"
    "app = FastAPI()\n"
    '@app.get("/itens")\n'
    "def listar_itens():\n"
    '    """Lista os itens."""\n'
    "    return []\n"
)


def test_case_rate_separa_minusculo_de_cased() -> None:
    minusculo = avaliar(_ref(), {"liste os itens": REFERENCIA}, None)
    cased = avaliar(_ref(), {"liste os itens": REFERENCIA_CASED}, None)
    assert minusculo["case_rate"] == 0.0
    assert cased["case_rate"] == 1.0


def test_sem_restauro_nao_esconde_o_minusculo() -> None:
    """Com restauro, `fastapi()` vira `FastAPI()` e roda; sem, e NameError."""
    pred = {"liste os itens": REFERENCIA}
    com = avaliar(_ref(), pred, None, executar=True)
    sem = avaliar(_ref(), pred, None, executar=True, sem_restauro=True)
    assert com["runs_rate"] == 1.0
    assert sem["runs_rate"] == 0.0


def test_docstrings_capitalizadas() -> None:
    assert docstrings_capitalizadas(REFERENCIA_CASED)
    assert not docstrings_capitalizadas(REFERENCIA)
    assert not docstrings_capitalizadas(REFERENCIA_CASED.replace("Lista", "lista"))
    assert not docstrings_capitalizadas("x = (\n")


def test_ground_truth_passa_sem_nenhum_remendo() -> None:
    """Criterio do eval sem remendo: a referencia roda crua, sem restauro e
    sem reparo, e ja nasce com caso e docstring."""
    _, teste = separar_teste(gerar_exemplos(1234), ENTIDADES_TESTE)
    pred = {e["instruction"]: e["output"] for e in teste}
    relatorio = avaliar(teste, pred, None, executar=True, sem_restauro=True)
    assert relatorio["runs_rate"] == 1.0
    assert relatorio["case_rate"] == 1.0
    assert relatorio["docstring_rate"] == 1.0
    assert relatorio["precisaram_de_restauro"] == 0
    assert relatorio["reparados"] == 0
