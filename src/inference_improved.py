import argparse, torch, yaml
from pathlib import Path
from dataset import read_lines, Vocab
from model_improved import Transformer

# --- 修复 1：修改 build_vocab 函数 ---
# 它现在需要传入 config 文件，以读取正确的 min_freq
def build_vocab(data_dir, cfg):
    src_lines = read_lines(Path(data_dir) / "train.en")
    trg_lines = read_lines(Path(data_dir) / "train.de")
    
    # 从 config 中获取 min_freq，而不是硬编码 1
    min_freq = cfg["misc"]["vocab_min_freq"]
    
    src_vocab = Vocab(min_freq=min_freq); trg_vocab = Vocab(min_freq=min_freq)
    src_vocab.build(src_lines); trg_vocab.build(trg_lines)
    return src_vocab, trg_vocab

def encode_sentence(sentence, vocab, max_len=128):
    toks = sentence.strip().split()
    ids = [vocab.stoi.get("<bos>", 2)]
    ids += [vocab.stoi.get(t, vocab.stoi.get("<unk>", 3)) for t in toks[:max_len-2]]
    ids.append(vocab.stoi.get("<eos>", 3))
    return ids

def decode_ids(ids, vocab):
    toks = []
    for i in ids:
        t = vocab.itos[i] if i < len(vocab.itos) else "<unk>"
        if t in ("<pad>", "<bos>"): continue
        if t == "<eos>": break
        toks.append(t)
    return " ".join(toks)

def greedy_decode(model, src_tensor, src_mask, trg_vocab, device, max_len=80):
    model.eval()
    with torch.no_grad():
        memory = model.encode(src_tensor.to(device), src_mask.to(device))
        ys = torch.full((1,1), trg_vocab.stoi["<bos>"], dtype=torch.long, device=device)
        for _ in range(max_len):
            tgt_mask = torch.triu(torch.ones((1, ys.size(1), ys.size(1)), device=device), 1).bool()
            out = model.decode(ys, memory, src_mask=src_mask.to(device), tgt_mask=~tgt_mask)
            logits = model.out(out)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
        return ys[0].cpu().tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/improved_25plus/best_model.pt")
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/en-de")
    args = parser.parse_args()

    # 加载 config 文件
    cfg = yaml.safe_load(open("configs/improved_25plus.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 修复 1 (续)：将 cfg 传递给 build_vocab ---
    src_vocab, trg_vocab = build_vocab(args.data_dir, cfg)

    # 现在 len(src_vocab) 和 len(trg_vocab) 将与训练时完全一致
    model = Transformer(
        len(src_vocab), len(trg_vocab),
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        num_layers=cfg["model"]["num_layers"],
        max_len=cfg["data"]["max_len"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    # --- 修复 2：解决 PyTorch 的 FutureWarning (推荐) ---
    # 添加 weights_only=True 更安全
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    src_ids = encode_sentence(args.sentence, src_vocab)
    src_tensor = torch.tensor([src_ids], dtype=torch.long)
    src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)

    out_ids = greedy_decode(model, src_tensor, src_mask, trg_vocab, device)
    print("Input:", args.sentence)
    print("Output:", decode_ids(out_ids, trg_vocab))

if __name__ == "__main__":
    main()