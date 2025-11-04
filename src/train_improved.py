import os, argparse, time, math, json, csv, datetime, logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
# --- 修复 1：导入新的调度器 ---
from utils import set_seed, load_config, count_parameters, plot_losses, WarmupCosineSchedule
from dataset import build_dataloaders
from model_improved import Transformer
from evaluate_improved import compute_bleu
from logging_utils import setup_logging

def make_masks(src, trg, pad_id=0):
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
    seq_len = trg.size(1)
    tgt_mask = (trg != pad_id).unsqueeze(1).unsqueeze(2)
    subsequent = torch.triu(torch.ones((1, seq_len, seq_len), device=trg.device), diagonal=1).bool()
    tgt_mask = tgt_mask & ~subsequent
    return src_mask, tgt_mask

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.ignore_index = ignore_index
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            mask = (target == self.ignore_index)
            if mask.any():
                true_dist[mask] = 0
        loss = torch.sum(-true_dist * pred, dim=-1)
        loss[mask] = 0.0
        return loss.sum() / (~mask).sum()

def parse_ablation_arg(ablation_str):
    out = {}
    if not ablation_str:
        return out
    for k in ablation_str.split(","):
        k = k.strip()
        if k:
            out[k] = True
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/improved_25plus.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ablation", type=str, default="", help="e.g. no_positional,no_residual")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cli_ablation = parse_ablation_arg(args.ablation)
    cfg.setdefault("ablation", {}).update(cli_ablation)

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    exp_name = "run_baseline" if not cli_ablation else "run_" + "_".join(cli_ablation.keys())
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("results") / exp_name / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=save_dir, log_filename="train.log")
    logging.info(f"Save dir: {save_dir}")
    logging.info(f"Ablation: {cfg.get('ablation', {})}")

    data_dir = cfg["data"]["path"]
    src_vocab, trg_vocab, train_loader, val_loader, test_loader = build_dataloaders(
        data_dir,
        batch_size=cfg["training"]["batch_size"],
        min_freq=cfg["misc"]["vocab_min_freq"],
        max_len=cfg["data"]["max_len"]
    )

    model = Transformer(
        len(src_vocab), len(trg_vocab),
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        num_layers=cfg["model"]["num_layers"],
        max_len=cfg["data"]["max_len"],
        dropout=cfg["model"]["dropout"],
        ablation=cfg.get("ablation", {}),
        use_pos=not cfg.get("ablation", {}).get("no_positional", False)
    ).to(device)

    logging.info(f"Model params: {count_parameters(model):,}")

    # --- 修复 2：使用新的优化器参数 ---
    optimizer = AdamW(model.parameters(), 
                      lr=float(cfg["training"]["lr"]), 
                      weight_decay=float(cfg["training"]["weight_decay"]),
                      betas=(float(cfg["training"]["adam_beta1"]), float(cfg["training"]["adam_beta2"])),
                      eps=float(cfg["training"]["adam_eps"])
                     )
    
    # --- 修复 3：使用新的调度器 ---
    if cfg["training"]["scheduler"] == "warmup_cosine":
        total_steps = len(train_loader) * cfg["training"]["epochs"]
        warmup_steps = cfg["training"]["warmup_steps"]
        logging.info(f"Using WarmupCosineSchedule with {total_steps} total steps and {warmup_steps} warmup steps.")
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=1e-7)
    else:
        # 保留旧的，以防万一
        logging.info("Using CosineAnnealingWarmRestarts.")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    criterion = LabelSmoothingLoss(classes=len(trg_vocab), 
                                   smoothing=float(cfg["training"]["label_smoothing"]), 
                                   ignore_index=0)
    
    clip_norm = float(cfg["training"]["clip_grad_norm"])
    logging.info(f"Using Grad Clip Norm: {clip_norm}")
    
    best_val = float("inf")
    csv_path = save_dir / "train_log.csv"
    
    train_losses_history = []
    val_losses_history = []

    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "bleu"])

    logging.info("Starting training...")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_loss = 0
        start_time = time.time()
        
        for i, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)
            trg_in, trg_out = trg[:, :-1], trg[:, 1:]
            
            src_mask, tgt_mask = make_masks(src, trg_in, pad_id=0)
            
            logits = model(src, trg_in, src_mask=src_mask, tgt_mask=tgt_mask)
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), trg_out.reshape(-1))
            
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            
            # --- 修复 4：调度器现在按步数更新 ---
            if cfg["training"]["scheduler"] == "warmup_cosine":
                scheduler.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses_history.append(train_loss)
        
        # --- 修复 4：如果不是 warmup_cosine，则按 epoch 更新 ---
        if cfg["training"]["scheduler"] != "warmup_cosine":
            scheduler.step()

        # Validation
        model.eval(); val_loss = 0
        with torch.no_grad():
            for src, trg in val_loader:
                src, trg = src.to(device), trg.to(device)
                trg_in, trg_out = trg[:, :-1], trg[:, 1:]
                src_mask, tgt_mask = make_masks(src, trg_in, pad_id=0)
                logits = model(src, trg_in, src_mask=src_mask, tgt_mask=tgt_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), trg_out.reshape(-1))
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses_history.append(val_loss)

        epoch_time = time.time() - start_time
        bleu = None
        
        # --- (这是您修改后的高效评估逻辑) ---
        if epoch % 10 == 0 or epoch == cfg["training"]["epochs"]:
            bleu = compute_bleu(model, test_loader, trg_vocab, device)
            logging.info(f"[Epoch {epoch:02d}/{cfg['training']['epochs']}] Time={epoch_time:.0f}s, train={train_loss:.4f}, val={val_loss:.4f}, BLEU={bleu:.2f}")
        else:
            logging.info(f"[Epoch {epoch:02d}/{cfg['training']['epochs']}] Time={epoch_time:.0f}s, train={train_loss:.4f}, val={val_loss:.4f}")

        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, bleu if bleu else ""])

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": cfg
            }, save_dir / "best_model.pt")
            logging.info(f"✅ Saved new best model (val_loss={val_loss:.4f})")

    plot_losses(train_losses_history, val_losses_history, out_dir=str(save_dir))
    
    logging.info("Training complete. Loading best model for final evaluation...")
    best_model_path = save_dir / "best_model.pt"
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logging.info(f"Loaded best model from epoch {ckpt.get('epoch', '?')} (val_loss={ckpt.get('val_loss', '?'):.4f})")
    else:
        logging.warning("No best_model.pt found. Running final BLEU on last model state.")

    bleu = compute_bleu(model, test_loader, trg_vocab, device)
    logging.info(f"Final BLEU (from best model) = {bleu:.2f}")

if __name__ == "__main__":
    main()