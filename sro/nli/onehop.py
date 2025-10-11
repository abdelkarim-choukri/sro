# sro/nli/onehop.py
from typing import List, Tuple
import torch
from sro.utils.batching import batched

class OneHopNLI:
    def __init__(self, tokenizer, model):
        self.tok = tokenizer
        self.model = model.eval()

    @torch.inference_mode()
    def score(self, premises: List[str], hypothesis: str,
              batch_size: int = 32, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        # returns (p_entail, p_contra) as float32 on CPU
        pe_list, pc_list = [], []
        for chunk in batched(premises, batch_size):
            enc = self.tok(chunk, [hypothesis]*len(chunk), padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(device if torch.cuda.is_available() and device=="cuda" else "cpu") for k,v in enc.items()}
            out = self.model(**enc)
            logits = getattr(out, "logits", out)
            # Assume label order: [contra, neutral, entail]. Adjust if different.
            probs = torch.softmax(logits, dim=-1).float().cpu()
            pe_list.append(probs[:, -1])
            pc_list.append(probs[:, 0])
        p_entail = torch.cat(pe_list, dim=0)
        p_contra = torch.cat(pc_list, dim=0)
        return p_entail, p_contra
