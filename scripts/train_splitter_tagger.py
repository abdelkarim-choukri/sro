from __future__ import annotations
import os, json, random, argparse
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

LABELS = ["PRT_B","PRT_I","PRT_E","BND","O"]
L2ID = {l:i for i,l in enumerate(LABELS)}
ID2L = {i:l for l,i in L2ID.items()}


def set_seed(s=13):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# Device plumbing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _cuda_info():
    if DEVICE.type == "cuda":
        name = torch.cuda.get_device_name(0)
        print(f"[CUDA] enabled = True  device = {name}")
    else:
        print("[CUDA] enabled = False  (CPU fallback)")


class JsonDataset(Dataset):
    def __init__(self, path):
        # support both JSON array files and newline-delimited JSON (jsonl)
        if path.endswith('.jsonl'):
            with open(path, encoding='utf-8') as fh:
                self.rows = [json.loads(l) for l in fh if l.strip()]
        else:
            with open(path, encoding='utf-8') as fh:
                self.rows = json.load(fh)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def build_token_piece_spans(offsets: List[Tuple[int, int]], tokens: List[dict]):
    """
    offsets: list of (piece_start, piece_end) from HF
    tokens: list of {'start','end','text'} from our tokenizer
    returns list of (a,z) piece-span indices for each token (a<z) or (0,0) if none
    """
    spans = []
    for tok in tokens:
        t0, t1 = tok.get("start", 0), tok.get("end", 0)
        a = None
        z = None
        for i, (p0, p1) in enumerate(offsets):
            if p0 == p1:
                continue
            if not (p1 <= t0 or p0 >= t1):
                if a is None:
                    a = i
                z = i + 1
        spans.append((a or 0, z or 0))
    return spans


def collate_token_level(batch, hf_tok):
    texts = [b["text"] for b in batch]
    enc = hf_tok(texts, return_offsets_mapping=True, return_tensors="pt", padding=True, truncation=True)
    B, S = enc["input_ids"].shape

    token_spans = []
    y_tok = []
    max_T = 0
    offs_batch = enc["offset_mapping"].tolist()
    for b_idx, ex in enumerate(batch):
        offs = offs_batch[b_idx]
        spans = build_token_piece_spans(offs, ex.get("tokens", []))
        token_spans.append(spans)
        y_tok.append(ex.get("labels", []))
        max_T = max(max_T, len(spans))

    import torch
    y = torch.full((B, max_T), fill_value=L2ID["O"], dtype=torch.long)
    tok_mask = torch.zeros((B, max_T), dtype=torch.bool)
    for b_idx in range(B):
        tlen = len(y_tok[b_idx])
        if tlen:
            y[b_idx, :tlen] = torch.tensor(y_tok[b_idx], dtype=torch.long)
            tok_mask[b_idx, :tlen] = True

    enc.pop("offset_mapping", None)
    return enc, token_spans, y, tok_mask


class Head(nn.Module):
    def __init__(self, hidden, n_labels=5):
        super().__init__()
        self.fc = nn.Linear(hidden, n_labels)
    def forward(self, x):
        return self.fc(x)


def train(args):
    set_seed(getattr(args, 'seed', 13))
    device = DEVICE
    tok = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    enc = AutoModel.from_pretrained(args.hf_model)
    enc.to(DEVICE)
    enc.eval()

    # start with backbone frozen
    for p in enc.parameters():
        p.requires_grad = False

    hidden = enc.config.hidden_size
    head = Head(hidden, n_labels=len(LABELS))

    # move model parts to DEVICE and print proof
    head.to(DEVICE)
    enc.to(DEVICE)
    if DEVICE.type == "cuda":
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(0)}")

    # prefer jsonl if available (one JSON per line), fall back to JSON array
    train_path = "artifacts/splitter/train.jsonl" if os.path.exists("artifacts/splitter/train.jsonl") else "artifacts/splitter/train.json"
    val_path = "artifacts/splitter/val.jsonl" if os.path.exists("artifacts/splitter/val.jsonl") else "artifacts/splitter/val.json"
    train_ds = JsonDataset(train_path)
    val_ds   = JsonDataset(val_path)
    # quick diagnostics
    def _counts(rows):
        n_tok = sum(len(r.get('tokens', [])) for r in rows)
        flat = [y for r in rows for y in r.get('labels', [])]
        from collections import Counter
        return len(rows), n_tok, dict(Counter(flat))
    tr_rows, tr_tok, tr_cnt = _counts(train_ds.rows)
    va_rows, va_tok, va_cnt = _counts(val_ds.rows)
    print(f"[DATA] train rows={tr_rows} tokens={tr_tok} label_counts={tr_cnt}")
    print(f"[DATA] val   rows={va_rows} tokens={va_tok} label_counts={va_cnt}")
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=2, pin_memory=pin,
                              collate_fn=lambda b: collate_token_level(b, tok))
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False,
                              num_workers=2, pin_memory=pin,
                              collate_fn=lambda b: collate_token_level(b, tok))

    # use recommended class-balanced weights (from analysis)
    # order: PRT_B,PRT_I,PRT_E,BND,O
    weights = torch.tensor([3.26, 3.74, 2.59, 2.75, 0.27], dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=weights)

    # unfreeze last N layers if requested (common HF layout)
    encoder_has_trainable = False
    unfreeze_n = getattr(args, 'unfreeze_last', None)
    if unfreeze_n is None:
        # support older option name
        unfreeze_n = getattr(args, 'unfreeze_last_layers', 0)
    try:
        layers = enc.encoder.layer
        if unfreeze_n and unfreeze_n > 0:
            for l in layers[-unfreeze_n:]:
                for p in l.parameters():
                    p.requires_grad = True
            encoder_has_trainable = True
            enc.train()
            print(f"[INFO] Unfroze last {unfreeze_n} encoder layers")
        else:
            encoder_has_trainable = any(p.requires_grad for p in enc.parameters())
    except Exception:
        encoder_has_trainable = any(p.requires_grad for p in enc.parameters())

    # optimizer with two LRs: head (1e-3) and backbone small (1e-5)
    lr_head = getattr(args, 'lr_head', None) or 1e-3
    lr_backbone = getattr(args, 'lr_backbone', None) or 1e-5
    head_params = list(head.parameters())
    backbone_params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": head_params, "lr": lr_head},
        {"params": backbone_params, "lr": lr_backbone},
    ], weight_decay=0.01)

    # scaler for AMP (enabled on CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # convenience for grad clipping
    model_params = list(head.parameters()) + backbone_params

    best = (-1.0, None)
    patience = int(getattr(args, 'early_stop_patience', 3))
    wait = 0

    def run_epoch(dl, train_mode=False):
        if train_mode:
            head.train()
            if encoder_has_trainable:
                enc.train()
            else:
                enc.eval()
        else:
            head.eval(); enc.eval()
        tot_loss = 0.0; nstep = 0
        all_pred = []; all_gold = []
        for enc_batch, token_spans, y_tok, tok_mask in dl:
            # move tensors to DEVICE; token_spans stays as python list
            enc_batch = {k: (v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in enc_batch.items()}
            y_tok = y_tok.to(DEVICE)
            tok_mask = tok_mask.to(DEVICE)

            if train_mode:
                # training: enable grad, use AMP autocast
                with torch.set_grad_enabled(True):
                    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda"), dtype=torch.float16):
                        out = enc(input_ids=enc_batch["input_ids"], attention_mask=enc_batch["attention_mask"])
                        X = out.last_hidden_state  # [B,S,H]
                        B,S,H = X.shape
                        T = y_tok.shape[1]
                        pooled = torch.zeros((B,T,H), dtype=X.dtype, device=DEVICE)
                        for b in range(B):
                            spans_b = token_spans[b]
                            for t,(a,z) in enumerate(spans_b):
                                if t >= T:
                                    break
                                if a < z:
                                    pooled[b,t] = X[b, a:z].mean(dim=0)
                        logits = head(pooled)
                        if bool(tok_mask.any()):
                            loss = crit(logits[tok_mask], y_tok[tok_mask])
                        else:
                            loss = torch.tensor(0.0, device=DEVICE)

                # backward with scaler
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model_params, 1.0)
                scaler.step(opt)
                scaler.update()

            else:
                # eval: no grad, AMP autocast for faster fp16 inference on CUDA
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda"), dtype=torch.float16):
                        out = enc(input_ids=enc_batch["input_ids"], attention_mask=enc_batch["attention_mask"])
                        X = out.last_hidden_state  # [B,S,H]
                        B,S,H = X.shape
                        T = y_tok.shape[1]
                        pooled = torch.zeros((B,T,H), dtype=X.dtype, device=DEVICE)
                        for b in range(B):
                            spans_b = token_spans[b]
                            for t,(a,z) in enumerate(spans_b):
                                if t >= T:
                                    break
                                if a < z:
                                    pooled[b,t] = X[b, a:z].mean(dim=0)
                        logits = head(pooled)

            tot_loss += float(loss); nstep += 1
            if bool(tok_mask.any()):
                pred = logits.argmax(-1)[tok_mask].detach().cpu().tolist()
                goldv = y_tok[tok_mask].detach().cpu().tolist()
                all_pred += [1 if p==L2ID["BND"] else 0 for p in pred]
                all_gold += [1 if g==L2ID["BND"] else 0 for g in goldv]

        if not all_gold:
            f1 = 0.0
        else:
            try:
                from sklearn.metrics import f1_score
                f1 = float(f1_score(all_gold, all_pred, zero_division=0))
            except Exception:
                f1 = 0.0
        return tot_loss / max(1, nstep), f1

    for ep in range(args.epochs):
        tl, tf1 = run_epoch(train_loader, train_mode=True)
        vl, vf1 = run_epoch(val_loader, train_mode=False)
        print(f"ep{ep} train_loss={tl:.3f} train_BND_F1={tf1:.3f} val_loss={vl:.3f} val_BND_F1={vf1:.3f}")
        if vf1 > best[0]:
            best = (vf1, {k: v.cpu() for k, v in head.state_dict().items()})
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"[INFO] Early stopping (no val BND-F1 improvement in {patience} epochs)")
            break

    best_state = best[1] if best[1] is not None else head.state_dict()
    os.makedirs("artifacts/models", exist_ok=True)
    torch.save(best_state, "artifacts/models/splitter_head.pt")
    json.dump({"labels": LABELS, "hf_model": args.hf_model}, open("artifacts/models/splitter_head_manifest.json", "w"))
    print("[OK] saved artifacts/models/splitter_head.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--unfreeze_last", type=int, default=2, help="Unfreeze last N transformer layers for fine-tuning (default 2)")
    ap.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for backbone when partially unfrozen")
    args = ap.parse_args()
    _cuda_info()
    train(args)
