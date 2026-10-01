#!/usr/bin/env bash
# Pipeline do pre-treino continuado grande, sem ninguem olhando: colhe codigo
# pt do GitHub e prosa da Wikipedia, monta o corpus, CPT, SFT so-LoRA e eval.
# Para no primeiro erro; cada etapa grava em logs/. A colheita retoma de onde
# parou (data/colhido/codigo_ptbr.feitos), entao rodar de novo e seguro.
# Uso (na 3090, com o .venv do rodar_tudo_3090.sh):
#   scripts/rodar_cpt_grande_3090.sh 2>&1 | tee logs/cpt_grande.log
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results data/colhido
PY=.venv/bin/python
BASE="${BASE:-AIInstituteLNCC/manaca-1b-base}"
SHARDS_FIM="${SHARDS_FIM:-880}"
WIKI_CHARS="${WIKI_CHARS:-60000000}"
CPT=checkpoints/manaca-cpt-grande
SFT=checkpoints/manaca-fastapi-cpt-grande

etapa() {
    local nome="$1"
    shift
    echo "[$(date '+%F %T')] >>> ${nome}"
    "$@" 2>&1 | tee "logs/${nome}.log"
    echo "[$(date '+%F %T')] <<< ${nome} ok"
}

[[ -x "$PY" ]] || { echo "sem $PY; rode antes o rodar_tudo_3090.sh" >&2; exit 1; }
[[ -n "${HF_TOKEN:-}" ]] || echo "aviso: sem HF_TOKEN, o Hub limita requisicao anonima" >&2

etapa g1_colher_codigo $PY scripts/colher_codigo_ptbr.py \
    --saida data/colhido/codigo_ptbr.jsonl --fim "$SHARDS_FIM"
etapa g2_colher_wiki $PY scripts/colher_wikipedia_ptbr.py \
    --saida data/colhido/wiki_ptbr.jsonl --max-caracteres "$WIKI_CHARS"
etapa g3_montar_corpus $PY scripts/montar_corpus_cpt.py --saida data/cpt_grande \
    --commit a3d205bf19640528718cb4f05ab77f4dfca6ad9a \
    --extra data/colhido/codigo_ptbr.jsonl data/colhido/wiki_ptbr.jsonl

etapa g4_cpt $PY scripts/pretreino_continuado.py --model "$BASE" \
    --treino data/cpt_grande/treino.jsonl --val data/cpt_grande/val.jsonl --out "$CPT"
# o estado do otimizador nos checkpoint-* passa de 10 GB e o disco esta a 95%
rm -rf "$CPT"/checkpoint-*

# so-LoRA de proposito: lm_head treinavel no SFT rebaixou a entidade inedita
etapa g5_sft $PY scripts/train_qlora.py --model "$CPT" \
    --data data/fastapi_ptbr.jsonl --out "$SFT"
rm -rf "$SFT"/checkpoint-*

etapa g6_eval $PY scripts/eval_code.py --model "$SFT" --executar --sem-restauro \
    --salvar-pred results/cpt_grande_pred.jsonl --out results/cpt_grande.json

echo "[$(date '+%F %T')] tudo pronto: results/cpt_grande.json"
