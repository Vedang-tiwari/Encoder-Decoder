"""Training loop with teacher forcing, validation and checkpointing."""

import math
import os
import time

import torch
import torch.nn as nn


class NoamLR:
    """Learning-rate schedule from 'Attention Is All You Need'.

    lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    """

    def __init__(self, optimizer, d_model, warmup_steps, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.factor * self.d_model ** -0.5
              * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


class Trainer:
    """Teacher-forced Seq2Seq training with validation and best-checkpoint saving.

    Target:        <SOS> I love AI <EOS>
    Decoder input: <SOS> I love AI
    Expected out:        I love AI <EOS>
    The causal mask guarantees the decoder never sees future target tokens.
    """

    def __init__(self, model, train_loader, val_loader, model_config,
                 train_config, device, tgt_pad_idx):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.tgt_pad_idx = tgt_pad_idx

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config["learning_rate"],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=train_config.get("weight_decay", 0.0),
        )

        sched = train_config.get("scheduler", "none")
        if sched == "noam":
            self.scheduler = NoamLR(self.optimizer, model_config["d_model"],
                                    train_config.get("warmup_steps", 400))
        elif sched == "cosine":
            total_steps = max(1, train_config["num_epochs"] * len(train_loader))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps)
        else:
            self.scheduler = None

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tgt_pad_idx,
            label_smoothing=train_config.get("label_smoothing", 0.0),
        )
        self.best_val_loss = float("inf")
        self.checkpoint_dir = train_config["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for src, tgt in loader:
            src, tgt = src.to(self.device), tgt.to(self.device)
            decoder_in, target = tgt[:, :-1], tgt[:, 1:]        # target shifting
            with torch.set_grad_enabled(train):
                logits = self.model(src, decoder_in)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_config.get("clip_grad", 1.0))
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
            ntok = target.ne(self.tgt_pad_idx).sum().item()
            total_loss += loss.item() * ntok
            total_tokens += ntok
        return total_loss / max(1, total_tokens)

    def save_checkpoint(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config,
            "val_loss": self.best_val_loss,
        }, path)

    def train(self):
        epochs = self.train_config["num_epochs"]
        ckpt_path = os.path.join(self.checkpoint_dir,
                                 self.train_config.get("checkpoint_name", "best_model.pt"))
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            ppl = math.exp(min(20.0, val_loss))
            print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_ppl={ppl:.2f} | {time.time() - t0:.1f}s")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(ckpt_path)
                print(f"  -> saved best checkpoint to {ckpt_path}")
        return ckpt_path