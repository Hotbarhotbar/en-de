import torch
from tqdm import tqdm
import sacrebleu
import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataset import build_dataloaders
from model_improved import Transformer
from utils import load_config

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

def beam_search_decode(model, src, trg_vocab, device, beam_size=4, max_len=100):
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        memory = model.encode(src, src_mask)
        
        batch_size = src.size(0)
        results = []
        
        for i in range(batch_size):
            # 对每个样本单独进行beam search
            memory_i = memory[i].unsqueeze(0)
            src_mask_i = src_mask[i].unsqueeze(0)
            
            # 初始beam: (score, sequence)
            beams = [(0.0, [trg_vocab.stoi[BOS]])]
            
            for step in range(max_len):
                new_beams = []
                for score, seq in beams:
                    if seq[-1] == trg_vocab.stoi[EOS]:
                        new_beams.append((score, seq))
                        continue
                    
                    # 准备输入
                    seq_tensor = torch.tensor(seq, device=device).unsqueeze(0)
                    tgt_mask = (seq_tensor != 0).unsqueeze(1).unsqueeze(2)
                    seq_len = seq_tensor.size(1)
                    subsequent = torch.triu(torch.ones((1, seq_len, seq_len), device=device), diagonal=1).bool()
                    tgt_mask = tgt_mask & ~subsequent
                    
                    # 解码
                    out = model.decode(seq_tensor, memory_i, src_mask_i, tgt_mask)
                    logits = model.out(out[:, -1:])
                    probs = torch.log_softmax(logits, dim=-1)
                    
                    # 取top-k
                    topk_probs, topk_indices = probs.topk(beam_size, dim=-1)
                    
                    for j in range(beam_size):
                        token_prob = topk_probs[0, 0, j].item()
                        token_id = topk_indices[0, 0, j].item()
                        new_score = score + token_prob
                        new_seq = seq + [token_id]
                        new_beams.append((new_score, new_seq))
                
                # 选择top beam_size个beam
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_size]
                
                # 如果所有beam都以EOS结束，提前停止
                if all(seq[-1] == trg_vocab.stoi[EOS] for _, seq in beams):
                    break
            
            # 选择最佳beam
            best_beam = max(beams, key=lambda x: x[0] / len(x[1]))  # 长度归一化
            results.append(best_beam[1])
        
        return results

def detokenize(ids_batch, vocab):
    rev = vocab.itos
    sents = []
    for ids in ids_batch:
        toks = []
        for i in ids:
            if i >= len(rev):
                t = "<unk>"
            else:
                t = rev[i]
            if t == PAD or t == BOS:
                continue
            if t == EOS:
                break
            toks.append(t)
        sents.append(" ".join(toks))
    return sents

def compute_bleu(model, test_loader, trg_vocab, device, beam_size=4):
    hyps = []
    refs = []

    for src, trg in tqdm(test_loader, desc="Evaluating"):
        out_ids = beam_search_decode(model, src, trg_vocab, device, beam_size=beam_size)
        hyps += detokenize(out_ids, trg_vocab)

        # --- 修复 2：使用相同的 detokenize 函数处理参考译文 ---
        # 确保 trg 也在 CPU 上并转换为 list of lists
        trg_ids_batch = trg.cpu().tolist()
        refs += detokenize(trg_ids_batch, trg_vocab)
        # -----------------------------------------------

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score