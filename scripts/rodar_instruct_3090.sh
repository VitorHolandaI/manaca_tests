#!/usr/bin/env bash
# SFT do Manaca-FastAPI que conversa: parte do CPT v4 e treina, no template do
# Manaca instruct (lncc), os 7.932 pedidos de FastAPI junto com a amostra do
# mix de instrucao do LNCC (data/sft_geral.jsonl, de misturar_sft.py). Depois
# mede as duas coisas que nao podem quebrar: o teste amplo de FastAPI e a
# conversa (eval_conversa.py). Os dois evals carregam o base em 4 bits, como
# o QLoRA treinou o adapter.
# Uso (na 3090, dentro do tmux):
#   scripts/rodar_instruct_3090.sh 2>&1 | tee logs/instruct.log
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results
PY=.venv/bin/python
CPT="${CPT:-checkpoints/manaca-cpt-v4}"
SFT=checkpoints/manaca-fastapi-instruct
EPOCAS="${EPOCAS:-3}"
DADOS=(data/fastapi_ptbr.jsonl data/sft_geral.jsonl)

etapa() {
    local nome="$1"
    shift
    echo "[$(date '+%F %T')] >>> ${nome}"
    "$@" 2>&1 | tee "logs/${nome}.log"
    echo "[$(date '+%F %T')] <<< ${nome} ok"
}

[[ -x "$PY" ]] || { echo "sem $PY; rode antes o rodar_tudo_3090.sh" >&2; exit 1; }
[[ -d "$CPT" ]] || { echo "sem $CPT; rode o rodar_v4_3090.sh" >&2; exit 1; }
for f in "${DADOS[@]}"; do
    [[ -f "$f" ]] || { echo "sem $f; rode o sync_3090.sh" >&2; exit 1; }
done

etapa i1_smoke $PY scripts/train_qlora.py --model "$CPT" --data "${DADOS[@]}" \
    --template lncc --out checkpoints/smoke-instruct --smoke
rm -rf checkpoints/smoke-instruct

etapa i2_sft $PY scripts/train_qlora.py --model "$CPT" --data "${DADOS[@]}" \
    --template lncc --epochs "$EPOCAS" --out "$SFT"
rm -rf "$SFT"/checkpoint-*

etapa i3_eval_fastapi $PY scripts/eval_code.py --model "$SFT" \
    --ref data/fastapi_ptbr_teste_amplo.jsonl --executar --sem-restauro --base-4bit \
    --salvar-pred results/amplo_instruct_pred.jsonl --out results/amplo_instruct.json
etapa i4_eval_conversa $PY scripts/eval_conversa.py --model "$SFT" --base-4bit \
    --saida results/conversa_instruct.jsonl

echo "[$(date '+%F %T')] pronto"
head -9 results/amplo_instruct.json
