#!/usr/bin/env bash
# Roda o pipeline inteiro na 3090, sem ninguem olhando: venv, smokes, CPT, SFT
# e eval. Para no primeiro erro; cada etapa grava log em logs/.
# Uso (na 3090): nohup scripts/rodar_tudo_3090.sh > logs/rodar_tudo.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results
BASE="${BASE:-AIInstituteLNCC/manaca-1b-base}"
CPT=checkpoints/manaca-cpt-cased
SFT=checkpoints/manaca-fastapi-cased

etapa() {
    local nome="$1"
    shift
    echo "[$(date '+%F %T')] >>> ${nome}"
    "$@" 2>&1 | tee "logs/${nome}.log"
    echo "[$(date '+%F %T')] <<< ${nome} ok"
}

if [[ ! -d .venv ]]; then
    uv venv --python 3.12
fi
# requirements-dev traz fastapi/sqlalchemy/httpx, que o eval --executar exige
uv pip install --python .venv/bin/python -r requirements-3090.txt \
    --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python -r requirements-dev.txt
PY=.venv/bin/python

# smokes primeiro: um erro de API aparece em minutos, nao depois de horas
etapa 01_cpt_smoke $PY scripts/pretreino_continuado.py \
    --model "$BASE" --out checkpoints/smoke-cpt --smoke
etapa 02_sft_smoke $PY scripts/train_qlora.py \
    --model checkpoints/smoke-cpt --data data/fastapi_ptbr.jsonl \
    --out checkpoints/smoke-sft --smoke
rm -rf checkpoints/smoke-cpt checkpoints/smoke-sft

etapa 03_cpt $PY scripts/pretreino_continuado.py --model "$BASE" --out "$CPT"
# esquecimento nao aborta: o eval do SFT continua sendo informativo
$PY - "$CPT/val_loss.json" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1]))
aviso = "ESQUECEU (val subiu)" if v["depois"] > v["antes"] else "ok"
print(f"[cpt] val loss antes {v['antes']:.4f} depois {v['depois']:.4f}: {aviso}")
EOF

etapa 04_sft $PY scripts/train_qlora.py \
    --model "$CPT" --data data/fastapi_ptbr.jsonl --out "$SFT"

etapa 05_eval_sem_remendo $PY scripts/eval_code.py --model "$SFT" \
    --executar --sem-restauro \
    --salvar-pred results/cased_pred.jsonl --out results/cased.json

echo "[$(date '+%F %T')] tudo pronto: results/cased.json"
