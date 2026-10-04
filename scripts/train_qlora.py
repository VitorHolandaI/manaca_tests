#!/usr/bin/env python3
"""QLoRA do Manaca-1B base no dataset FastAPI PT-BR.

Fiel ao sft/train.py oficial:
  - template Alpaca-PT (preambulo + ### Instrucao: / ### Resposta:)
  - loss SO na resposta (DataCollatorForCompletionOnlyLM)
  - marcador de resposta passado como token-ids com corte [1:] do fronteira do
    SentencePiece; sem isso o collator mascara tudo e a loss vira NaN.

Contrato de caixa: o dataset atual e cased e exige o tokenizer v2
(--tokenizer models/manaca-tokenizer-v2) ou o checkpoint do pre-treino
continuado (scripts/pretreino_continuado.py), que ja tem o vocabulario do v2.
Partindo do base, as linhas novas sao iniciadas aqui e treinam via
`modules_to_save`. Partindo do CPT o padrao e so-LoRA (lm_head congelado):
treinar o lm_head no SFT rebaixa toda peca que o SFT nao pede, inclusive a
entidade inedita do teste. `--treinar-vocab` liga mesmo assim.

ATENCAO: este script NAO foi executado na maquina de desenvolvimento (sem
NVIDIA). Rode na RTX 3090 com --smoke primeiro, no stack fixado em
requirements-3090.txt (trl 0.12.2: `processing_class`, `max_seq_length`).

Uso:
    accelerate launch scripts/train_qlora.py \\
        --model AIInstituteLNCC/manaca-1b-base \\
        --data data/fastapi_ptbr.jsonl \\
        --out checkpoints/manaca-fastapi-qlora --smoke
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Protocol

from caixa_marcada import codificar_para, usa_caixa_marcada
from expandir_vocab import expandir_modelo
from subword_sampling import AmostradorUnigram, ids_com_amostragem

PREAMBULO = (
    "Abaixo esta uma instrucao que descreve uma tarefa. "
    "Escreva uma resposta que atenda adequadamente ao pedido."
)
INSTRUCAO_MARCA = "\n\n### Instrucao:\n"
RESPOSTA_MARCA = "\n\n### Resposta:\n"

# a LoRA congela as embeddings: sem isso as linhas novas do v2 ficariam com o
# init por media para sempre
MODULOS_DO_VOCAB = ["embed_tokens", "lm_head"]

LORA_ALVOS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def montar_texto(instruction: str, saida: str, eos: str) -> str:
    return f"{PREAMBULO}{INSTRUCAO_MARCA}{instruction.strip()}{RESPOSTA_MARCA}{saida.strip()}{eos}"


def carregar_jsonl(caminho: Path) -> list[dict[str, str]]:
    if not caminho.is_file():
        raise FileNotFoundError(f"dataset nao encontrado: {caminho}")
    linhas: list[dict[str, str]] = []
    # split("\n"): splitlines() quebra no \u2028 cru de dentro das strings
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").split("\n"), 1):
        if not linha.strip():
            continue
        try:
            linhas.append(json.loads(linha))
        except json.JSONDecodeError as err:
            raise ValueError(f"{caminho}:{numero}: JSON invalido: {err}") from err
    if not linhas:
        raise ValueError(f"dataset vazio: {caminho}")
    return linhas


class TokenizerComPad(Protocol):
    eos_token: str
    pad_token: str | None

    @property
    def eos_token_id(self) -> int: ...

    @property
    def pad_token_id(self) -> int | None: ...

    def get_added_vocab(self) -> dict[str, int]: ...


def garantir_pad_distinto(tokenizer: TokenizerComPad) -> str | None:
    """Troca o pad por um token especial existente quando ele e o proprio EOS.

    O collator do transformers 4.46.3 poe -100 em todo label igual ao pad, e
    pad == eos (Qwen3 base) apagaria o EOS do treino. Devolve o pad novo.
    """
    if (
        tokenizer.pad_token is not None
        and tokenizer.pad_token_id != tokenizer.eos_token_id
    ):
        return None
    candidatos = sorted(
        t
        for t in tokenizer.get_added_vocab()
        if "pad" in t and t != tokenizer.eos_token
    )
    if not candidatos:
        raise ValueError(
            f"pad ausente ou igual ao eos {tokenizer.eos_token!r} e nenhum token "
            "especial com 'pad' no vocabulario; esperado um pad distinto do eos"
        )
    tokenizer.pad_token = candidatos[0]
    return candidatos[0]


def ids_marcador(tokenizer, texto: str) -> list[int]:
    """Token-ids do marcador, com corte [1:] do fronteira inicial do SentencePiece."""
    ids = tokenizer.encode(texto, add_special_tokens=False)
    return ids[1:] if len(ids) > 1 else ids


def exemplo_para_collator(collator, linha: dict) -> dict:
    """O collator de amostragem le `text`; o do trl so aceita os ids.

    O `DataCollatorForCompletionOnlyLM` da pad na linha inteira, e a coluna
    `text` (string) estoura com "too many dimensions 'str'". No treino real o
    Trainer ja remove essa coluna; aqui, na checagem, ela tem de sair na mao.
    """
    if isinstance(collator, CollatorComAmostragem):
        return linha
    return {"input_ids": linha["input_ids"], "attention_mask": linha["attention_mask"]}


def checar_collator(collator, exemplo: dict) -> None:
    """Confirma que o collator mascara o prompt e deixa a resposta com loss real."""
    batch = collator([exemplo])
    labels = batch["labels"][0].tolist()
    com_loss = sum(1 for x in labels if x != -100)
    mascarados = sum(1 for x in labels if x == -100)
    if com_loss == 0:
        raise RuntimeError(
            f"collator mascarou TODOS os tokens (loss viraria NaN). O marcador "
            f"{RESPOSTA_MARCA!r} nao casou; verifique o corte [1:] do SentencePiece."
        )
    if mascarados == 0:
        raise RuntimeError(
            "collator nao mascarou o prompt; a loss incidiria no prompt inteiro."
        )
    print(
        f"[qlora] check do collator OK: {mascarados} tokens mascarados, {com_loss} com loss"
    )


class TokenizerDeTreino(Protocol):
    eos_token: str | None

    def get_vocab(self) -> dict[str, int]: ...


def textos_de_treino(
    linhas: list[dict[str, str]], tokenizer: TokenizerDeTreino
) -> list[str]:
    """Prompt + resposta + eos; com o tokenizer v3, a maiuscula vira marcador.

    Exemplo: textos_de_treino([{"instruction": "x", "output": "y"}], tok)
    """
    eos = tokenizer.eos_token or ""
    textos = [montar_texto(r["instruction"], r["output"], eos) for r in linhas]
    if usa_caixa_marcada(tokenizer):
        return [codificar_para(tokenizer, t) for t in textos]
    return textos


def marca_de_resposta(tokenizer: TokenizerDeTreino) -> str:
    """O `### Resposta:` como ele aparece no texto de treino (v3: com marcador)."""
    if not usa_caixa_marcada(tokenizer):
        return RESPOSTA_MARCA
    return codificar_para(tokenizer, RESPOSTA_MARCA)


def montar_dataset(linhas: list[dict[str, str]], tokenizer):
    """Dataset com o texto cru E uma tokenizacao deterministica.

    O `input_ids` existe para o SFTTrainer pular a tokenizacao dele (ele devolve
    o dataset como esta quando ja ve `input_ids`) e nao apagar a coluna `text`,
    que e de onde o collator de amostragem re-tokeniza a cada lote.
    """
    from datasets import Dataset

    textos = textos_de_treino(linhas, tokenizer)
    codificado = tokenizer(textos, add_special_tokens=True)
    return Dataset.from_dict(
        {
            "text": textos,
            "input_ids": codificado["input_ids"],
            "attention_mask": codificado["attention_mask"],
        }
    )


def _indice(alvo: list[int], ids: list[int]) -> int | None:
    """Posicao de `alvo` como subsequencia contigua de `ids`, ou None."""
    n = len(alvo)
    for i in range(len(ids) - n + 1):
        if ids[i : i + n] == alvo:
            return i
    return None


def caminho_do_tokenizer_json(model_id: str) -> Path:
    """tokenizer.json do modelo: local se existir, senao do cache do Hub."""
    local = Path(model_id) / "tokenizer.json"
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model_id, "tokenizer.json"))


class CollatorComAmostragem:
    """Completion-only com a resposta re-segmentada por amostragem, a cada lote.

    Subword regularization (Kudo 2018; Provilkov 2020): a mesma palavra aparece
    com segmentacoes diferentes, o que quebra a dependencia de uma tokenizacao
    canonica -- que e como o modelo decorou a grafia das 73 entidades. O prefixo
    fica deterministico (Viterbi), entao o marcador de resposta sempre casa.

    Exige `remove_unused_columns=False`: a coluna `text` precisa sobreviver ate
    aqui, porque e dela que cada lote e re-tokenizado.
    """

    def __init__(
        self,
        tokenizer,
        amostrador: AmostradorUnigram,
        ids_marca: list[int],
        semente: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.amostrador = amostrador
        self.ids_marca = list(ids_marca)
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.rng = random.Random(semente)
        self.sem_caminho = 0

    def _ids_do_exemplo(self, texto: str) -> list[int]:
        ids = ids_com_amostragem(self.amostrador, texto, RESPOSTA_MARCA, self.rng)
        if ids is None:
            # o vocabulario nao cobre algum caractere: cai no deterministico
            self.sem_caminho += 1
            return list(self.tokenizer(texto).input_ids)
        return [self.bos_id] + ids

    def __call__(self, exemplos: list[dict]) -> dict:
        import torch

        entradas, rotulos = [], []
        for exemplo in exemplos:
            ids = self._ids_do_exemplo(exemplo["text"])
            posicao = _indice(self.ids_marca, ids)
            corte = posicao + len(self.ids_marca) if posicao is not None else 0
            rotulo = list(ids)
            rotulo[:corte] = [-100] * corte
            entradas.append(ids)
            rotulos.append(rotulo)
        largura = max(len(x) for x in entradas)
        return {
            "input_ids": torch.tensor(
                [x + [self.pad_id] * (largura - len(x)) for x in entradas]
            ),
            "attention_mask": torch.tensor(
                [[1] * len(x) + [0] * (largura - len(x)) for x in entradas]
            ),
            "labels": torch.tensor([x + [-100] * (largura - len(x)) for x in rotulos]),
        }


def _carregar_modelo(args: argparse.Namespace, torch):
    """(modelo, lora). `--full` treina todos os parametros; o default e QLoRA 4-bit.

    No modo completo as **embeddings ficam treinaveis** -- que e onde mora a
    composicao de caractere (SpellingBee, NAACL 2022) e exatamente o que a LoRA
    congela. E onde o defeito de grafia de sub-palavra vive.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if args.full:
        modelo = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        _expandir_se_v2(modelo, args)
        modelo.config.use_cache = False
        return modelo, None

    from peft import LoraConfig, prepare_model_for_kbit_training

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # device explicito: com "auto" o accelerate as vezes manda modulo pra CPU e o
    # bitsandbytes 4-bit recusa, mesmo com a GPU inteira livre (3090, 23 GiB)
    modelo = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map={"": 0}
    )
    # antes do prepare: ele converte as embeddings para fp32 e as linhas novas
    # tem de existir nesse momento
    novas = _expandir_se_v2(modelo, args)
    modelo = prepare_model_for_kbit_training(modelo)
    modelo.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_ALVOS,
        # opt-in partindo do checkpoint do CPT: em 2026-09-30 o lm_head treinavel
        # desempatou o caso (case_rate 1.0) mas rebaixou toda peca que o SFT nao
        # pede, e a entidade inedita sumiu (rotas 0.0, `deletar_ comentario`)
        modules_to_save=MODULOS_DO_VOCAB if (novas or args.treinar_vocab) else None,
    )
    return modelo, lora


def _expandir_se_v2(modelo, args: argparse.Namespace) -> int:  # type: ignore[no-untyped-def]
    """Resize + init para o vocabulario do --tokenizer; 0 sem v2 ou ja expandido."""
    if not args.tokenizer:
        return 0
    novas = expandir_modelo(
        modelo,
        caminho_do_tokenizer_json(args.model),
        Path(args.tokenizer) / "tokenizer.json",
    )
    print(f"[qlora] {novas} linhas novas de embedding (tokenizer v2)")
    return novas


def treinar(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    try:
        from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    except ImportError as err:
        # sem fallback pro transformers: ele nao exporta esse collator
        # (conferido no 4.46.3 fixado e no 5.17 do venv XPU)
        raise ImportError(
            f"trl incompativel ({err}); esperado trl==0.12.2, "
            "instale com: pip install -r requirements-3090.txt"
        ) from err

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    pad_novo = garantir_pad_distinto(tokenizer)
    if pad_novo:
        print(f"[qlora] pad era o eos; usando {pad_novo} como pad")

    modelo, lora = _carregar_modelo(args, torch)

    linhas = carregar_jsonl(Path(args.data))
    treino = montar_dataset(linhas, tokenizer)
    print(f"[qlora] {len(treino)} exemplos de {args.data}")

    ids_marca = ids_marcador(tokenizer, marca_de_resposta(tokenizer))
    if args.subword_alpha > 0:
        amostrador = AmostradorUnigram(
            caminho_do_tokenizer_json(args.model), args.subword_alpha
        )
        collator = CollatorComAmostragem(tokenizer, amostrador, ids_marca)
        print(f"[qlora] subword regularization: alpha={args.subword_alpha}")
    else:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=ids_marca, tokenizer=tokenizer
        )
    checar_collator(collator, exemplo_para_collator(collator, treino[0]))

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        max_seq_length=args.max_seq,
        logging_steps=1,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to=[],
        dataset_text_field="text",
        max_steps=2 if args.smoke else -1,
        # Adam fp32 de um 1,7B inteiro custaria 13,8 GB so de estados; 8-bit cabe
        # nos 24 GB da 3090 junto com pesos bf16 (3,4) e gradientes (3,4)
        # em full FT cada checkpoint e o modelo inteiro (3,4 GB); sem limite, 3
        # epocas deixariam 10 GB de disco a toa
        save_total_limit=1 if args.full else None,
        optim="adamw_bnb_8bit" if args.full else "adamw_torch",
        # o collator de amostragem le a coluna `text` a cada lote, entao ela nao
        # pode ser descartada; com amostragem desligada vale o default
        remove_unused_columns=args.subword_alpha <= 0,
    )
    treinador = SFTTrainer(
        model=modelo,
        args=config,
        train_dataset=treino,
        data_collator=collator,
        peft_config=lora,
        processing_class=tokenizer,
    )
    treinador.train()
    if isinstance(collator, CollatorComAmostragem):
        print(
            f"[qlora] amostragem: {collator.sem_caminho} exemplo(s) cairam no "
            "deterministico por falta de caminho no vocabulario"
        )
    treinador.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[qlora] adapter salvo em {args.out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="QLoRA Manaca-1B (FastAPI PT-BR)")
    ap.add_argument("--model", default="AIInstituteLNCC/manaca-1b-base")
    ap.add_argument("--data", default="data/fastapi_ptbr.jsonl")
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="pasta do tokenizer v2 (expandir_vocab.py); omita partindo do checkpoint do CPT",
    )
    ap.add_argument("--out", default="checkpoints/manaca-fastapi-qlora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument(
        "--full",
        action="store_true",
        help="fine-tune completo (destrava embeddings); default e QLoRA 4-bit",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="default: 2e-5 com --full, 2e-4 sem",
    )
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument(
        "--treinar-vocab",
        action="store_true",
        help="embed_tokens/lm_head treinaveis mesmo partindo de modelo ja expandido",
    )
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true", help="2 passos e para")
    ap.add_argument(
        "--subword-alpha",
        type=float,
        default=0.0,
        help="subword regularization: 0 desliga; 0.1-0.2 e a faixa recomendada",
    )
    args = ap.parse_args()
    if args.tokenizer and args.subword_alpha > 0:
        # o AmostradorUnigram le o tokenizer.json do --model, com Lowercase
        raise ValueError(
            f"--subword-alpha {args.subword_alpha} com --tokenizer {args.tokenizer}: "
            "a amostragem so conhece o vocabulario oficial; use --subword-alpha 0"
        )
    if args.lr is None:
        # full FT atualiza 1,7B de parametros; 2e-4 esqueceria o pre-treino
        args.lr = 2e-5 if args.full else 2e-4
    treinar(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
