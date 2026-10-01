"""Os scripts nao sao pacote; eval_code importa restore_case pelo diretorio."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
