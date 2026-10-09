#!/usr/bin/env python3
"""Chat com o GGUF do Manaca-FastAPI v4 servido pelo llama-server, por ids.

O llama.cpp nao tokeniza igual ao treino: o Manaca e Unigram com byte
fallback, e o modo Unigram do llama.cpp nao tem byte fallback (`\\n` vira
<unk>); o modo SentencePiece segmenta por fusoes. Em 100 textos, 6 sairam
identicos ao transformers (2026-10-06). Por isso o prompt vai como lista de
ids (o /completion aceita, e sem string nao poe BOS sozinho) e os ids gerados
voltam com `return_tokens`; tokenizar e decodificar ficam deste lado.

Com --oficial, conversa com o Manaca instruct original do LNCC (o GGUF de
models/manaca-instruct-ugm), com o tokenizer oficial e o template lncc,
que e o do chat_oficial.py. --template lncc serve o SFT com conversa.

Uso:
    llama-server -m models/manaca-fastapi-v4-q8_0.gguf --port 8080 -ngl 99
    uv run --with transformers==4.46.3 python scripts/chat_gguf.py
    # instruct original:
    llama-server -m models/manaca-instruct-ugm/F16-UGM.gguf --port 8080 -ngl 99
    uv run --with transformers==4.46.3 python scripts/chat_gguf.py --oficial
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Protocol

from eval_code import cortar_no_fim, desfazer_marcador, prompts_para
from template_alpaca import LEGADO, LNCC, TEMPLATES, Template

URL_PADRAO = "http://127.0.0.1:8080"
TOKENIZER_PADRAO = "models/manaca-tokenizer-v4"
TOKENIZER_OFICIAL = "models/manaca-hf-official"
MAX_NOVOS_PADRAO = 1024


class GeradorDeTokens(Protocol):
    def gerar(self, ids: list[int], max_novos: int) -> list[int]: ...


def corpo_da_requisicao(ids: list[int], max_novos: int) -> dict[str, object]:
    """JSON do /completion: ids no prompt, greedy, ids gerados de volta.

    Exemplo: corpo_da_requisicao([1, 884], 512)["prompt"] == [1, 884]
    """
    return {
        "prompt": ids,
        "n_predict": max_novos,
        "temperature": 0.0,
        "top_k": 1,
        "return_tokens": True,
        "cache_prompt": False,
        "stream": False,
    }


class ClienteLlamaServer:
    """Fala com o /completion do llama-server; so a stdlib."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/") + "/completion"

    def gerar(self, ids: list[int], max_novos: int) -> list[int]:
        dados = json.dumps(corpo_da_requisicao(ids, max_novos)).encode("utf-8")
        pedido = urllib.request.Request(
            self.url, data=dados, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(pedido, timeout=1800) as resposta:
                corpo = json.loads(resposta.read())
        except urllib.error.URLError as err:
            raise ConnectionError(
                f"llama-server nao respondeu em {self.url} ({err}); esperado o "
                "servidor de pe: llama-server -m <gguf> --port 8080"
            ) from err
        if "tokens" not in corpo:
            raise ValueError(
                f"resposta sem `tokens`: {list(corpo)[:8]}; esperado return_tokens"
            )
        return [int(t) for t in corpo["tokens"]]


def responder(  # type: ignore[no-untyped-def]
    tok,
    cliente: GeradorDeTokens,
    pedido: str,
    max_novos: int,
    template: Template = LEGADO,
) -> str:
    """Resposta para um pedido, no template com que o modelo foi treinado."""
    prompt = prompts_para(
        [{"instruction": pedido.strip(), "output": ""}], tok, template
    )[0]
    ids = tok(prompt, add_special_tokens=True)["input_ids"]
    fins = {t for t in (tok.eos_token_id, tok.pad_token_id) if t is not None}
    gerados = cortar_no_fim(cliente.gerar(ids, max_novos), fins)
    texto = tok.decode(gerados, skip_special_tokens=True).strip()
    return desfazer_marcador([texto], tok)[0]


def _laco(  # type: ignore[no-untyped-def]
    tok, cliente: GeradorDeTokens, max_novos: int, template: Template
) -> None:
    print("Pedido em portugues (Enter envia; 'sair' ou Ctrl-D encerra).")
    while True:
        try:
            pedido = input("\n> ")
        except EOFError:
            return
        if pedido.strip().lower() in {"sair", "exit", "quit"}:
            return
        if not pedido.strip():
            continue
        inicio = time.monotonic()
        print(responder(tok, cliente, pedido, max_novos, template))
        print(f"[{time.monotonic() - inicio:.0f} s]", file=sys.stderr)


def main() -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description="Chat com o GGUF do Manaca-FastAPI")
    ap.add_argument("--url", default=URL_PADRAO)
    ap.add_argument("--tokenizer", default=None, help="padrao: v4, ou o oficial")
    ap.add_argument(
        "--template",
        choices=sorted(TEMPLATES),
        default="legado",
        help="o do treino do modelo servido (lncc no SFT com conversa)",
    )
    ap.add_argument(
        "--oficial",
        action="store_true",
        help="Manaca instruct original: tokenizer oficial e template lncc",
    )
    ap.add_argument("--prompt", default=None, help="um pedido so, sem o laco")
    ap.add_argument("--max-novos", type=int, default=MAX_NOVOS_PADRAO)
    args = ap.parse_args()

    template = LNCC if args.oficial else TEMPLATES[args.template]
    padrao = TOKENIZER_OFICIAL if args.oficial else TOKENIZER_PADRAO
    tok = AutoTokenizer.from_pretrained(args.tokenizer or padrao)
    cliente = ClienteLlamaServer(args.url)
    if args.prompt:
        print(responder(tok, cliente, args.prompt, args.max_novos, template))
        return 0
    _laco(tok, cliente, args.max_novos, template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
