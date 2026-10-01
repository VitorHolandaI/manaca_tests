#!/usr/bin/env bash
# Sincroniza o projeto com a maquina da RTX 3090.
# Uso: scripts/sync_3090.sh [--dry-run]
#
# Pesos ficam de fora (o base vem do Hub), exceto a pasta do tokenizer v2,
# que o pretreino_continuado.py e o train_qlora.py --tokenizer leem do disco.
# data/upstream/ vai junto: o montar_corpus_cpt.py le o checkout do FastAPI.
# status/ e estado do ZeroTier (identity.secret, authtoken.secret), nao projeto.
set -euo pipefail

DESTINO="${DESTINO:-usuario@maquina-3090:~/manaca_build/}"
ORIGEM="$(cd "$(dirname "$0")/.." && pwd)/"

if [[ ! -f "${ORIGEM}models/manaca-tokenizer-v2/tokenizer.json" ]]; then
    echo "falta ${ORIGEM}models/manaca-tokenizer-v2; rode: python scripts/expandir_vocab.py" >&2
    exit 1
fi

# a ordem importa: o include da pasta do v2 precisa vir antes do exclude models/*
rsync -avz --human-readable "$@" \
    --include 'models/' \
    --include 'models/manaca-tokenizer-v2/***' \
    --exclude 'models/*' \
    --exclude '.venv/' \
    --exclude 'status/' \
    --exclude '__pycache__/' --exclude '.mypy_cache/' --exclude '.ruff_cache/' \
    --exclude '.pytest_cache/' --exclude 'checkpoints/' \
    --exclude '*.gguf' --exclude '*.safetensors' --exclude '*.pt' --exclude '*.pth' \
    "$ORIGEM" "$DESTINO"
