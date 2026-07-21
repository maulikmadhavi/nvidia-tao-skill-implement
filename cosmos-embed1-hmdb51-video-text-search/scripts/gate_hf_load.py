"""Gate: an exported HF dir loads via transformers and can embed a text + video."""
import sys

sys.path.insert(0, "/exp/scripts")
from embed_lib import embed_texts, embed_video, load_model

hf_dir = sys.argv[1]
video = sys.argv[2] if len(sys.argv) > 2 else None

model, processor = load_model(hf_dir)
n_params = sum(p.numel() for p in model.parameters())
print(f"HF LOAD OK: {type(model).__name__}, {n_params:,} params")

t = embed_texts(model, processor, ["a video of a person riding a bike"])
print(f"text emb ok: shape {t.shape}, norm {float((t ** 2).sum() ** 0.5):.4f}")
if video:
    v = embed_video(model, processor, video)
    print(f"video emb ok: shape {v.shape}, cos(text,video) {float(t[0] @ v):.4f}")
print("GATE PASSED: exported HF model loads and embeds")
