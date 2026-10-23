"""Hierarchical context memory with provenance. Exact tier keeps original text; compressed tier keeps embeddings.
Eviction under budget moves exact text to disk (never deletes), and retrieval always returns the source span."""
import json, os, torch
from dataclasses import dataclass


@dataclass
class Segment:
    conv: str
    seg_id: int
    start: int  # token span in the original conversation
    end: int
    emb: torch.Tensor  # compressed (mean-pooled) representation
    text: str | None  # exact source; None once spilled (path then in spill_path)
    spill_path: str | None = None
    valid: bool = True


class ContextMemory:
    def __init__(self, spill_dir, ram_budget_segments=64):
        self.segs: dict[str, list[Segment]] = {}
        self.dir, self.budget = spill_dir, ram_budget_segments
        os.makedirs(spill_dir, exist_ok=True)

    def add(self, conv, start, end, token_embs, text):
        L = self.segs.setdefault(conv, [])
        L.append(Segment(conv, len(L), start, end, token_embs.mean(0), text))
        self._enforce_budget()

    def _enforce_budget(self):
        live = [s for L in self.segs.values() for s in L if s.text is not None]
        for s in live[: max(0, len(live) - self.budget)]:  # oldest first
            s.spill_path = os.path.join(self.dir, f"{s.conv}-{s.seg_id}.json")
            with open(s.spill_path, "w") as f:
                json.dump({"conv": s.conv, "start": s.start, "end": s.end, "text": s.text}, f)
            s.text = None

    def invalidate(self, conv, seg_id):
        self.segs[conv][seg_id].valid = False

    def retrieve(self, conv, query_emb, k=3):
        """Per-conversation isolation: only searches `conv`. Returns [(score, segment, exact_text)]."""
        cands = [s for s in self.segs.get(conv, []) if s.valid]
        sc = torch.stack([torch.cosine_similarity(query_emb, s.emb, dim=0) for s in cands]) if cands else []
        top = sorted(zip(map(float, sc), cands), key=lambda t: -t[0])[:k]
        return [(sc_, s, self.exact(s)) for sc_, s in top]

    def exact(self, s):
        if s.text is not None:
            return s.text
        with open(s.spill_path) as f:
            return json.load(f)["text"]
