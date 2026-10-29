#!/usr/bin/env bash
# Benchmark de qualquer modelo do Hugging Face nas mesmas condicoes do Manaca:
# SFT QLoRA nos mesmos dados, teste amplo de FastAPI (182 pedidos, 12
# entidades fora do treino) e eval de conversa, com o base em 4 bits.
# O modelo precisa ser suportado pelo transformers 4.46.3 e ter as camadas
# q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj (Llama, Qwen2,
# Mistral, Gemma); o pad igual ao EOS e trocado pelo train_qlora.
# Uso (na 3060, dentro do tmux):
#   scripts/comparar_modelo_3060.sh Qwen/Qwen2.5-0.5B qwen05
#   MODO=fastapi scripts/comparar_modelo_3060.sh Qwen/Qwen2.5-0.5B qwen05-fastapi
# MODO=instruct (padrao): FastAPI + mix do LNCC, template lncc (como o entregue)
# MODO=fastapi: so FastAPI, template legado (como o benchmark de 2026-10-03)
set -euo pipefail

cd "$(dirname "$0")/.."
[[ $# -ge 2 ]] || { echo "uso: $0 <id do Hugging Face> <nome curto>" >&2; exit 1; }
MODELO="$1"
NOME="$2"
MODO="${MODO:-instruct}"
EPOCAS="${EPOCAS:-3}"
PY=.venv/bin/python
SFT="checkpoints/comparar-${NOME}"
case "$MODO" in
    instruct) DADOS=(data/fastapi_ptbr.jsonl data/sft_geral.jsonl); TEMPLATE=lncc ;;
    fastapi) DADOS=(data/fastapi_ptbr.jsonl); TEMPLATE=legado ;;
    *) echo "MODO invalido: $MODO, esperado instruct ou fastapi" >&2; exit 1 ;;
esac
mkdir -p logs results
[[ -x "$PY" ]] || { echo "sem $PY; rode antes o rodar_tudo_3060.sh" >&2; exit 1; }

etapa() {
    local nome="$1"
    shift
    echo "[$(date '+%F %T')] >>> ${nome}"
    "$@" 2>&1 | tee "logs/${nome}.log"
    echo "[$(date '+%F %T')] <<< ${nome} ok"
}

etapa "c1_smoke_${NOME}" $PY scripts/train_qlora.py --model "$MODELO" \
    --data "${DADOS[@]}" --template "$TEMPLATE" --out "checkpoints/smoke-${NOME}" --smoke
rm -rf "checkpoints/smoke-${NOME}"
etapa "c2_sft_${NOME}" $PY scripts/train_qlora.py --model "$MODELO" \
    --data "${DADOS[@]}" --template "$TEMPLATE" --epochs "$EPOCAS" --out "$SFT"
rm -rf "$SFT"/checkpoint-*
etapa "c3_fastapi_${NOME}" $PY scripts/eval_code.py --model "$SFT" \
    --ref data/fastapi_ptbr_teste_amplo.jsonl --executar --sem-restauro --base-4bit \
    --salvar-pred "results/comparar_${NOME}_pred.jsonl" --out "results/comparar_${NOME}.json"
etapa "c4_conversa_${NOME}" $PY scripts/eval_conversa.py --model "$SFT" --base-4bit \
    --saida "results/comparar_${NOME}_conversa.jsonl"

$PY - "$NOME" <<'EOF'
import json, sys
nome = sys.argv[1]
r = json.load(open(f"results/comparar_{nome}.json"))
conversa = [json.loads(l) for l in open(f"results/comparar_{nome}_conversa.jsonl")]
texto = sum(not c["virou_codigo"] for c in conversa)
print(f"| {nome} | {r['entidade_rate']} | {r['compile_rate']} | {r['runs_rate']} | {texto} de {len(conversa)} |")
EOF
