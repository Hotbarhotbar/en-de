import random, os, yaml, json, math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_model(model, optimizer, epoch, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, path)

def load_model_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", None)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_losses(train_losses, valid_losses, out_dir="results"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("epochs") # 修复 2：这里应该是 epochs，因为我们是每个 epoch 记录一次
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(Path(out_dir)/"train_valid_loss.png")
    plt.close()

# 修复的学习率预热调度器 (保留，以备将来使用)
class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
        # 保存初始学习率
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # 线性预热
            return self.min_lr + (self.initial_lrs[0] - self.min_lr) * self.current_step / self.warmup_steps
        else:
            # 余弦衰减
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.initial_lrs[0] - self.min_lr) * cosine_decay
