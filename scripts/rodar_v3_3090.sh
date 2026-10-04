#!/usr/bin/env bash
# Pipeline do tokenizer v3 (maiuscula como marcador), sem ninguem olhando:
# colhe prosa do GigaVerbo-v2, monta o corpus, smoke + CPT a partir do CPT
# grande (remapeado peca por peca), SFT so-LoRA e eval no teste amplo.
# Para no primeiro erro; cada etapa grava em logs/. Pre-requisitos: o .venv do
# rodar_tudo_3090.sh, o CPT grande em checkpoints/ e os JSONL de data/colhido/.
# Uso (na 3090, dentro do tmux):
#   scripts/rodar_v3_3090.sh 2>&1 | tee logs/v3.log
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results data/colhido
PY=.venv/bin/python
ORIGEM="${ORIGEM:-checkpoints/manaca-cpt-grande}"
V3=models/manaca-tokenizer-v3
GIGA_TOKENS="${GIGA_TOKENS:-65000000}"
CPT=checkpoints/manaca-cpt-v3
SFT=checkpoints/manaca-fastapi-v3

etapa() {
    local nome="$1"
    shift
    echo "[$(date '+%F %T')] >>> ${nome}"
    "$@" 2>&1 | tee "logs/${nome}.log"
    echo "[$(date '+%F %T')] <<< ${nome} ok"
}

[[ -x "$PY" ]] || { echo "sem $PY; rode antes o rodar_tudo_3090.sh" >&2; exit 1; }
[[ -f "$V3/tokenizer.json" ]] || { echo "sem $V3; rode o sync_3090.sh" >&2; exit 1; }
[[ -d "$ORIGEM" ]] || { echo "sem $ORIGEM; o v3 parte do CPT grande" >&2; exit 1; }
for f in codigo_ptbr wiki_ptbr; do
    [[ -f "data/colhido/$f.jsonl" ]] || { echo "sem data/colhido/$f.jsonl" >&2; exit 1; }
done

etapa v1_colher_giga $PY scripts/colher_gigaverbo.py \
    --saida data/colhido/gigaverbo.jsonl --max-tokens "$GIGA_TOKENS"
etapa v2_montar_corpus $PY scripts/montar_corpus_cpt.py --saida data/cpt_v3 \
    --commit a3d205bf19640528718cb4f05ab77f4dfca6ad9a \
    --extra data/colhido/codigo_ptbr.jsonl data/colhido/wiki_ptbr.jsonl \
    data/colhido/gigaverbo.jsonl

# smoke de 2 passos: pega erro de remapeamento ou de marcador antes das ~9 h
etapa v3_smoke_cpt $PY scripts/pretreino_continuado.py --model "$ORIGEM" \
    --tokenizer "$V3" --treino data/cpt_v3/val.jsonl --val data/cpt_v3/val.jsonl \
    --out checkpoints/smoke-v3 --smoke
rm -rf checkpoints/smoke-v3

etapa v4_cpt $PY scripts/pretreino_continuado.py --model "$ORIGEM" \
    --tokenizer "$V3" --treino data/cpt_v3/treino.jsonl --val data/cpt_v3/val.jsonl \
    --out "$CPT"
# o estado do otimizador nos checkpoint-* passa de 10 GB e o disco esta apertado
rm -rf "$CPT"/checkpoint-*

etapa v5_sft $PY scripts/train_qlora.py --model "$CPT" \
    --data data/fastapi_ptbr.jsonl --out "$SFT"
rm -rf "$SFT"/checkpoint-*

etapa v6_eval $PY scripts/eval_code.py --model "$SFT" \
    --ref data/fastapi_ptbr_teste_amplo.jsonl --executar --sem-restauro \
    --salvar-pred results/amplo_v3_pred.jsonl --out results/amplo_v3.json

echo "[$(date '+%F %T')] tudo pronto: results/amplo_v3.json"
head -9 results/amplo_v3.json
