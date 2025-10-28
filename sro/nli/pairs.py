# sro/nli/pairs.py

import torch

from sro.utils.batching import batched


class PairNLI:
    def __init__(self, tokenizer, model):
        self.tok = tokenizer
        self.model = model.eval()

    @torch.inference_mode()
    def score_pairs(self, pairs: list[tuple[str,str,str]],
                    batch_size: int = 32, device: str = "cuda") -> torch.Tensor:
        """
        pairs: list of (s1, s2, hypothesis) strings.
        returns: p_entail for the pair -> hypothesis, float32 on CPU, shape [len(pairs)]
        """
        out_list = []
        for chunk in batched(pairs, batch_size):
            s1, s2, hyp = zip(*chunk)
            enc = self.tok(list(s1), [f"{a} {b}" for a,b in zip(s1,s2)], padding=True, truncation=True, return_tensors="pt")
            # If you instead encode (s1+s2 -> hyp), adapt above accordingly.
            enc = {k: v.to(device if torch.cuda.is_available() and device=="cuda" else "cpu") for k,v in enc.items()}
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).float().cpu()
            out_list.append(probs[:, -1])  # entail prob
        return torch.cat(out_list, dim=0)
