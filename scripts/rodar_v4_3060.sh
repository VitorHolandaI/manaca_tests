#!/usr/bin/env bash
# Pipeline do tokenizer v4 (caixa + fronteira como marcadores) na 3060: smoke,
# CPT a partir do CPT v3 (remapeado peca por peca, so <|cola|> nasce), SFT
# so-LoRA e eval no teste amplo. Reusa data/cpt_v3: o corpus guarda texto cru
# e a codificacao acontece ao carregar, pelo vocabulario do tokenizer.
# Uso (na 3060, dentro do tmux):
#   scripts/rodar_v4_3060.sh 2>&1 | tee logs/v4.log
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results
PY=.venv/bin/python
ORIGEM="${ORIGEM:-checkpoints/manaca-cpt-v3}"
V4=models/manaca-tokenizer-v4
CPT=checkpoints/manaca-cpt-v4
SFT=checkpoints/manaca-fastapi-v4

etapa() {
    local nome="$1"
    shift
    echo "[$(date '+%F %T')] >>> ${nome}"
    "$@" 2>&1 | tee "logs/${nome}.log"
    echo "[$(date '+%F %T')] <<< ${nome} ok"
}

[[ -x "$PY" ]] || { echo "sem $PY; rode antes o rodar_tudo_3060.sh" >&2; exit 1; }
[[ -f "$V4/tokenizer.json" ]] || { echo "sem $V4; rode o sync_3060.sh" >&2; exit 1; }
[[ -d "$ORIGEM" ]] || { echo "sem $ORIGEM; o v4 parte do CPT v3" >&2; exit 1; }
[[ -f data/cpt_v3/treino.jsonl ]] || { echo "sem data/cpt_v3; rode o rodar_v3_3060.sh" >&2; exit 1; }

# smoke de 2 passos: pega erro de remapeamento ou de marcador antes das ~10 h
etapa w1_smoke_cpt $PY scripts/pretreino_continuado.py --model "$ORIGEM" \
    --tokenizer "$V4" --treino data/cpt_v3/val.jsonl --val data/cpt_v3/val.jsonl \
    --out checkpoints/smoke-v4 --smoke
rm -rf checkpoints/smoke-v4

etapa w2_cpt $PY scripts/pretreino_continuado.py --model "$ORIGEM" \
    --tokenizer "$V4" --treino data/cpt_v3/treino.jsonl --val data/cpt_v3/val.jsonl \
    --out "$CPT"
# o estado do otimizador nos checkpoint-* passa de 10 GB e o disco esta apertado
rm -rf "$CPT"/checkpoint-*

etapa w3_sft $PY scripts/train_qlora.py --model "$CPT" \
    --data data/fastapi_ptbr.jsonl --out "$SFT"
rm -rf "$SFT"/checkpoint-*

etapa w4_eval $PY scripts/eval_code.py --model "$SFT" \
    --ref data/fastapi_ptbr_teste_amplo.jsonl --executar --sem-restauro \
    --salvar-pred results/amplo_v4_pred.jsonl --out results/amplo_v4.json

echo "[$(date '+%F %T')] tudo pronto: results/amplo_v4.json"
head -12 results/amplo_v4.json
