#!/usr/bin/env bash
# Um comando para conversar com o modelo nesta maquina: sobe o llama-server
# com o GGUF na GPU (Intel, build SYCL do llama.cpp), espera ficar pronto,
# abre o chat_gguf.py com o template certo e derruba o servidor na saida.
# Uso, da raiz do projeto:
#   scripts/rodar_chat_local.sh            # instruct: conversa e FastAPI
#   scripts/rodar_chat_local.sh v4         # so FastAPI (v4)
#   scripts/rodar_chat_local.sh instruct "Crie um endpoint para listar livros no FastAPI"
set -euo pipefail

cd "$(dirname "$0")/.."
MODELO="${1:-instruct}"
PORTA="${PORTA:-8080}"
case "$MODELO" in
    instruct) GGUF=models/manaca-fastapi-instruct-q8_0.gguf; TEMPLATE=lncc ;;
    v4) GGUF=models/manaca-fastapi-v4-q8_0.gguf; TEMPLATE=legado ;;
    *) echo "modelo invalido: $MODELO, esperado instruct ou v4" >&2; exit 1 ;;
esac
[[ -f "$GGUF" ]] || { echo "sem $GGUF; veja docs/ENTREGA.md, secao Gerar o GGUF" >&2; exit 1; }
command -v llama-server >/dev/null || { echo "sem llama-server no PATH (llama.cpp)" >&2; exit 1; }
if curl -s -m 2 "localhost:${PORTA}/health" >/dev/null; then
    echo "a porta ${PORTA} ja tem um servidor; pare ele ou use PORTA=8081" >&2
    exit 1
fi

LOG="$(mktemp -t manaca-servidor.XXXXXX.log)"
llama-server -m "$GGUF" --port "$PORTA" -ngl 99 -c 4096 >"$LOG" 2>&1 &
SERVIDOR=$!
trap 'kill "$SERVIDOR" 2>/dev/null || true' EXIT

echo "carregando $GGUF (log em $LOG)..."
for _ in $(seq 1 90); do
    curl -s "localhost:${PORTA}/health" 2>/dev/null | grep -q ok && break
    kill -0 "$SERVIDOR" 2>/dev/null || { echo "o servidor caiu; veja $LOG" >&2; exit 1; }
    sleep 1
done

CHAT=(uv run --with transformers==4.46.3 python scripts/chat_gguf.py
    --url "http://127.0.0.1:${PORTA}" --template "$TEMPLATE")
if [[ $# -ge 2 ]]; then
    "${CHAT[@]}" --prompt "$2" 2>/dev/null
else
    "${CHAT[@]}" 2>/dev/null
fi
