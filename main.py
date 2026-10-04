"""Project entry point.

Orchestrates: load config -> tokenizers -> dataset/DataLoaders -> Transformer
-> train -> evaluate -> checkpoint -> inference. All logic lives in src/.
"""

import os
import sys

import torch
import yaml

from src import dataset as data
from src.evaluator import evaluate
from src.inference import translate
from src.model import TransformerSeq2Seq
from src.tokenizer import Tokenizer
from src.trainer import Trainer


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_or_load_tokenizer(path, texts, min_freq, lowercase):
    if os.path.exists(path):
        print(f"Loading tokenizer vocab from {path}")
        return Tokenizer.load(path)
    print(f"Building tokenizer vocabulary from {len(texts)} sentences")
    tok = Tokenizer().build_vocab(texts, min_freq=min_freq, lowercase=lowercase)
    tok.save(path)
    return tok


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(config_path="configs/config.yaml"):
    config = load_config(config_path)
    device = resolve_device(config["training"].get("device", "auto"))
    print(f"Using device: {device}")

    # 1. Data -----------------------------------------------------------
    if config.get("demo_data", False) or not os.path.exists(config["data"]["train_src"]):
        print("Creating demo dataset (word-reversal task)...")
        data.create_demo_dataset(config["data"]["raw_dir"])
    train_src, _ = data.load_parallel_file(config["data"]["train_src"],
                                           config["data"]["train_tgt"])

    # 2. Tokenizers -----------------------------------------------------
    tok_cfg = config["tokenizer"]
    lowercase = tok_cfg.get("lowercase", True)
    src_tok = build_or_load_tokenizer(tok_cfg["src_vocab"], train_src,
                                      tok_cfg["min_freq"], lowercase)
    tgt_tok = build_or_load_tokenizer(tok_cfg["tgt_vocab"], train_src,
                                      tok_cfg["min_freq"], lowercase)
    print(f"Source vocab: {src_tok.vocab_size} | Target vocab: {tgt_tok.vocab_size}")

    # 3. DataLoaders ----------------------------------------------------
    train_loader, val_loader = data.build_dataloaders(config, src_tok, tgt_tok)

    # 4. Model ----------------------------------------------------------
    mc = config["model"]
    model_config = {
        "src_vocab_size": src_tok.vocab_size,
        "tgt_vocab_size": tgt_tok.vocab_size,
        "d_model": mc["d_model"],
        "num_heads": mc["num_heads"],
        "num_encoder_layers": mc["num_encoder_layers"],
        "num_decoder_layers": mc["num_decoder_layers"],
        "d_ff": mc["d_ff"],
        "dropout": mc["dropout"],
        "max_len": mc["max_seq_len"],
        "src_pad_idx": src_tok.pad_id,
        "tgt_pad_idx": tgt_tok.pad_id,
    }
    model = TransformerSeq2Seq(**model_config)
    print(f"Transformer parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 5. Train ----------------------------------------------------------
    trainer = Trainer(model, train_loader, val_loader, model_config,
                      config["training"], device, tgt_tok.pad_id)
    ckpt_path = trainer.train()

    # 6. Evaluate -------------------------------------------------------
    print("\nEvaluating on validation set (greedy decoding)...")
    results = evaluate(model, val_loader, device, tgt_tok.pad_id,
                       src_tok, tgt_tok, config["inference"]["max_decode_len"],
                       criterion=trainer.criterion)
    print(f"Validation | loss={results['loss']:.4f} | "
          f"ppl={results['perplexity']:.2f} | BLEU={results['bleu'] * 100:.2f}")

    # 7. Inference samples ----------------------------------------------
    print("\nSample translations (greedy):")
    for src_text in train_src[:5]:
        out = translate(model, src_text, src_tok, tgt_tok, device,
                        config["inference"]["max_decode_len"])
        print(f"  src: {src_text}")
        print(f"  out: {out}")

    print(f"\nDone. Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")