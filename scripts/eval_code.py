#!/usr/bin/env python3
"""Avalia geracao de codigo FastAPI PT-BR com checagem por codigo, sem juiz LLM.

Tres metricas objetivas:
  1. compile_rate  -- a saida, apos restaurar o caso, passa em ast.parse?
  2. marker_rate   -- contem os marcadores obrigatorios da operacao pedida
                      (rota, nome da funcao, status_code, tratamento de 404)?
  3. precisaram_de_restauro -- quantas saidas tinham nome de biblioteca
                      minusculo que restore_case.py teve de corrigir.
  4. case_rate      -- fracao de saidas que ja nascem com o caso de Python
                      (nenhum nome para restaurar).
  5. docstring_rate -- fracao em que toda rota tem docstring capitalizada.
  6. entidade_rate  -- fracao em que a classe do model SQLAlchemy e a da
                      referencia (pega a troca por uma entidade do treino).

Modos:
  --pred arquivo.jsonl   pontua predicoes ja geradas ({instruction, output})
  --model <hf_id>        gera com transformers e pontua (precisa GPU/XPU)
  --limit N              avalia so os N primeiros (default: todos)
  --max-new-tokens N     teto da geracao (default: 1024)
  --sem-restauro         pontua o texto cru, sem restore_case.py; junto com
                         --executar e sem --reparar e o eval sem remendo

Uso:
  python scripts/eval_code.py --ref data/fastapi_ptbr.jsonl --pred saida.jsonl
  python scripts/eval_code.py --ref data/fastapi_ptbr.jsonl --model caminho/do/base --limit 5
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple, TypedDict

from caixa_marcada import TokenizerComVocab, codificar, decodificar, usa_caixa_marcada
from restore_case import nomes_a_restaurar, restaurar

PREAMBULO = (
    "Abaixo esta uma instrucao que descreve uma tarefa. "
    "Escreva uma resposta que atenda adequadamente ao pedido."
)

ROTAS = re.compile(r'@app\.(?:post|get|put|delete)\("[^"]+"')
FUNCOES = re.compile(r"def \w+\(")
STATUS = re.compile(r"status_code=(\d{3})")
MODELO_SQL = re.compile(r"^class (\w+)\(Base\):", re.MULTILINE)

# a maior referencia mede 697 tokens; com 512 a geracao cortava 220/256
MAX_NOVOS_TOKENS_PADRAO = 1024

# prompts por lote na geracao. O base nao emite EOS de forma confiavel, entao
# ele gasta os 1024 tokens inteiros em cada prompt; sem lote o eval demora
# ~15 min e parece travado.
LOTE_PADRAO = 8


class Exigencia(NamedTuple):
    rotas: list[str]
    funcoes: list[str]
    status: list[str]

    @property
    def todas(self) -> list[str]:
        return self.rotas + self.funcoes + self.status


def exigencias(referencia: str) -> Exigencia:
    """Marcadores estruturais da referencia, separados por categoria.

    Nao parseia a instrucao em linguagem natural -- isso e fragil. A referencia
    ja diz qual rota, qual funcao e qual status_code a resposta precisa ter.

    Exemplo: separa '@app.post("/produtos"' (rota), 'def criar_produto('
    (funcao) e 'status_code=201', para o relatorio dizer qual categoria falhou.
    """
    return Exigencia(
        ROTAS.findall(referencia),
        FUNCOES.findall(referencia),
        [f"status_code={c}" for c in sorted(set(STATUS.findall(referencia)))],
    )


class Pontuacao(NamedTuple):
    compila: bool
    marcadores: bool
    precisou_restauro: bool
    motivo: str | None
    rotas: bool = False
    funcoes: bool = False
    status: bool = False
    executou: bool | None = None
    reparou: int = 0
    caso_ok: bool = False
    docstring_ok: bool = False
    entidade_ok: bool = False


class Relatorio(TypedDict):
    total: int
    compile_rate: float
    marker_rate: float
    runs_rate: float | None
    case_rate: float
    docstring_rate: float
    entidade_rate: float
    precisaram_de_restauro: int
    reparados: int
    por_marcador: dict[str, float]
    falhas: list[dict[str, str]]


def docstrings_capitalizadas(codigo: str) -> bool:
    """Toda funcao decorada (rota) tem docstring que comeca em maiuscula.

    Exemplo: uma rota cuja docstring e 'Lista os itens.' devolve True; sem
    docstring, ou com 'lista os itens.', devolve False.
    """
    try:
        arvore = ast.parse(codigo)
    except SyntaxError:
        return False
    rotas = [
        n
        for n in ast.walk(arvore)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.decorator_list
    ]
    docs = [ast.get_docstring(f) for f in rotas]
    return bool(docs) and all(d and d[0].isupper() for d in docs)


def entidade_certa(referencia: str, saida: str) -> bool:
    """A saida declara exatamente os models SQLAlchemy da referencia.

    Exemplo: referencia com 'class Comentario(Base):' e saida com
    'class Membro(Base):' devolve False, mesmo compilando e rodando.
    """
    esperadas = set(MODELO_SQL.findall(referencia))
    return bool(esperadas) and set(MODELO_SQL.findall(saida)) == esperadas


def _pontuar(
    referencia: str,
    saida: str,
    executar: bool = False,
    reparar_nomes: bool = False,
    sem_restauro: bool = False,
) -> Pontuacao:
    """Nota de UMA predicao; saida quebrada vira falha, nunca derruba o eval."""
    if not saida:
        return Pontuacao(False, False, False, "sem predicao")
    try:
        faltando = nomes_a_restaurar(saida)
        restaurado = saida if sem_restauro else restaurar(saida)
    except ValueError as err:
        return Pontuacao(False, False, False, f"nao tokenizavel: {err}")
    reparou = 0
    if reparar_nomes:
        from reparar_identificadores import reparar

        restaurado, mapa = reparar(restaurado)
        reparou = len(mapa)
    motivo: str | None = None
    try:
        ast.parse(restaurado)
    except SyntaxError as err:
        motivo = f"syntax: {err.msg}"
    # compila e SO sintaxe: o erro de runtime vai pro motivo, mas nao derruba
    # o compile_rate -- sao metricas diferentes
    compila = motivo is None
    ex = exigencias(referencia)
    bate_rotas = all(m in restaurado for m in ex.rotas)
    bate_funcoes = all(m in restaurado for m in ex.funcoes)
    bate_status = all(m in restaurado for m in ex.status)
    marcou = bool(ex.todas) and bate_rotas and bate_funcoes and bate_status
    executou: bool | None = None
    if executar and compila:
        from sandbox_gerado import rodar_rotas

        executou, erro_runtime = rodar_rotas(restaurado)
        if not executou:
            motivo = f"runtime: {erro_runtime}"
    return Pontuacao(
        compila,
        marcou,
        bool(faltando),
        motivo,
        bate_rotas,
        bate_funcoes,
        bate_status,
        executou,
        reparou,
        caso_ok=not faltando,
        docstring_ok=docstrings_capitalizadas(restaurado),
        entidade_ok=entidade_certa(referencia, restaurado),
    )


def _taxa(acertos: int, total: int) -> float:
    return round(acertos / total, 4) if total else 0.0


def avaliar(
    ref: list[dict[str, str]],
    pred: dict[str, str],
    limite: int | None,
    executar: bool = False,
    reparar_nomes: bool = False,
    sem_restauro: bool = False,
) -> Relatorio:
    itens = ref[:limite] if limite else ref
    notas = [
        _pontuar(
            e["output"],
            pred.get(e["instruction"], ""),
            executar,
            reparar_nomes,
            sem_restauro,
        )
        for e in itens
    ]
    falhas = [
        {"instruction": e["instruction"], "motivo": n.motivo}
        for e, n in zip(itens, notas)
        if n.motivo
    ]
    total = len(itens)
    return {
        "total": total,
        "compile_rate": _taxa(sum(n.compila for n in notas), total),
        "marker_rate": _taxa(sum(n.marcadores for n in notas), total),
        "runs_rate": (
            _taxa(sum(1 for n in notas if n.executou), total) if executar else None
        ),
        "case_rate": _taxa(sum(n.caso_ok for n in notas), total),
        "docstring_rate": _taxa(sum(n.docstring_ok for n in notas), total),
        "entidade_rate": _taxa(sum(n.entidade_ok for n in notas), total),
        "precisaram_de_restauro": sum(n.precisou_restauro for n in notas),
        "reparados": sum(1 for n in notas if n.reparou),
        "por_marcador": {
            "rotas": _taxa(sum(n.rotas for n in notas), total),
            "funcoes": _taxa(sum(n.funcoes for n in notas), total),
            "status": _taxa(sum(n.status for n in notas), total),
        },
        "falhas": falhas[:10],
    }


def _carregar_jsonl(caminho: Path) -> list[dict[str, str]]:
    if not caminho.is_file():
        raise FileNotFoundError(f"arquivo nao encontrado: {caminho}")
    return [
        json.loads(l)
        # split("\n"): splitlines() quebra no \u2028 cru de dentro das strings
        for l in caminho.read_text(encoding="utf-8").split("\n")
        if l.strip()
    ]


def _dedupe_por_instrucao(ref: list[dict[str, str]]) -> list[dict[str, str]]:
    """O dataset repete instrucoes com variacao de saida; a metrica e por instrucao."""
    vistos: set[str] = set()
    unicos: list[dict[str, str]] = []
    for exemplo in ref:
        chave = exemplo["instruction"]
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(exemplo)
    return unicos


def _base_do_adapter(model_id: str) -> str:
    """Se model_id for um adapter PEFT, devolve o modelo base que ele precisa.

    Exemplo: _base_do_adapter("checkpoints/manaca-fastapi-qlora") devolve o
    base_model_name_or_path do adapter_config.json; para qualquer outro id
    devolve o proprio id.
    """
    config = Path(model_id) / "adapter_config.json"
    if not config.is_file():
        return model_id
    return json.loads(config.read_text(encoding="utf-8"))["base_model_name_or_path"]


def _carregar_para_geracao(model_id: str, device: str):
    """Tokenizer + modelo no device; aplica o adapter PEFT se model_id for um."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    base = _base_do_adapter(model_id)
    modelo = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    if base != model_id:
        from peft import PeftModel

        modelo = PeftModel.from_pretrained(modelo, model_id)
    return tok, modelo.to(device).eval()


def _prompt(instruction: str) -> str:
    return f"{PREAMBULO}\n\n### Instrucao:\n{instruction}\n\n### Resposta:\n"


def prompts_para(itens: list[dict[str, str]], tok: TokenizerComVocab) -> list[str]:
    """Prompt Alpaca de cada item; com o tokenizer v3, com marcador de caixa."""
    prompts = [_prompt(e["instruction"]) for e in itens]
    return [codificar(p) for p in prompts] if usa_caixa_marcada(tok) else prompts


def desfazer_marcador(textos: list[str], tok: TokenizerComVocab) -> list[str]:
    """Volta a maiuscula nas saidas do v3; outros tokenizers passam intactos."""
    return [decodificar(t) for t in textos] if usa_caixa_marcada(tok) else textos


def cortar_no_fim(ids: list[int], fins: set[int]) -> list[int]:
    """Ids gerados ate o primeiro token de fim (eos ou pad), exclusive.

    Exemplo: cortar_no_fim([5, 6, 9, 7], {9}) == [5, 6]
    """
    for i, token in enumerate(ids):
        if token in fins:
            return ids[:i]
    return ids


def _gerar_em_lotes(
    modelo, tok, prompts: list[str], device: str, max_novos: int, lote: int
) -> list[str]:
    """Gera em lotes com padding a esquerda; devolve um texto por prompt.

    Padding a esquerda porque o modelo continua a partir do ultimo token -- com
    padding a direita ele continuaria o pad. O corte usa a LARGURA do lote
    (input_ids.shape[1]), nao a soma da attention_mask: com padding a esquerda
    o seq comeca pelo pad, entao cortar pela soma deixa entrar os ultimos
    tokens do prompt na saida (vira eco da instrucao na frente do codigo).
    """
    import torch
    from tqdm import tqdm

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    saidas: list[str] = []
    faixas = tqdm(range(0, len(prompts), lote), unit="lote", desc="gerando")
    for inicio in faixas:
        enc = tok(
            prompts[inicio : inicio + lote],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        entrada = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            out = modelo.generate(
                **entrada,
                max_new_tokens=max_novos,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        largura = int(entrada["input_ids"].shape[1])
        fins = {t for t in (tok.eos_token_id, tok.pad_token_id) if t is not None}
        for seq in out:
            gerados = cortar_no_fim(seq[largura:].tolist(), fins)
            saidas.append(tok.decode(gerados, skip_special_tokens=True).strip())
    return saidas


def gerar_com_modelo(
    model_id: str,
    ref: list[dict[str, str]],
    limite: int | None,
    max_novos: int,
    lote: int,
) -> dict[str, str]:
    import torch

    device = (
        "xpu"
        if torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tok, modelo = _carregar_para_geracao(model_id, device)
    itens = ref[:limite] if limite else ref
    prompts = prompts_para(itens, tok)
    textos = desfazer_marcador(
        _gerar_em_lotes(modelo, tok, prompts, device, max_novos, lote), tok
    )
    return {e["instruction"]: t for e, t in zip(itens, textos)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Eval de codigo FastAPI PT-BR (sem juiz LLM)"
    )
    ap.add_argument("--ref", default="data/fastapi_ptbr_teste.jsonl")
    ap.add_argument("--pred", default=None, help="jsonl com {instruction, output}")
    ap.add_argument("--model", default=None, help="hf id ou caminho local para gerar")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NOVOS_TOKENS_PADRAO)
    ap.add_argument("--lote", type=int, default=LOTE_PADRAO, help="prompts por lote")
    ap.add_argument("--out", default=None, help="salva o relatorio json")
    ap.add_argument(
        "--salvar-pred",
        default=None,
        help="grava as geracoes em jsonl, para ler o que o modelo escreveu",
    )
    ap.add_argument(
        "--executar",
        action="store_true",
        help="sobe cada saida como app FastAPI e roda o CRUD (precisa de fastapi)",
    )
    ap.add_argument(
        "--reparar",
        action="store_true",
        help="repara identificador inexistente e acento antes de pontuar",
    )
    ap.add_argument(
        "--sem-restauro",
        action="store_true",
        help="nao passa a saida pelo restore_case.py (mede o caso que o modelo emitiu)",
    )
    args = ap.parse_args()

    ref = _dedupe_por_instrucao(_carregar_jsonl(Path(args.ref)))
    if args.pred:
        pred = {
            r["instruction"]: r.get("output", "")
            for r in _carregar_jsonl(Path(args.pred))
        }
    elif args.model:
        pred = gerar_com_modelo(
            args.model, ref, args.limit, args.max_new_tokens, args.lote
        )
    else:
        raise SystemExit("informe --pred ou --model")

    if args.salvar_pred:
        destino = Path(args.salvar_pred)
        destino.write_text(
            "\n".join(
                json.dumps({"instruction": k, "output": v}, ensure_ascii=False)
                for k, v in pred.items()
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"geracoes salvas em {destino}", file=sys.stderr)

    relatorio = avaliar(
        ref, pred, args.limit, args.executar, args.reparar, args.sem_restauro
    )
    texto = json.dumps(relatorio, ensure_ascii=False, indent=2)
    print(texto)
    if args.out:
        Path(args.out).write_text(texto + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
