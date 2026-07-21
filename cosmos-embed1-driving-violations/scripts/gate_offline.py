"""Gate: the OFFLINE image is self-sufficient — run with --network none.

Verifies baked deps (sklearn/matplotlib/protobuf<7), then loads Cosmos-Embed1
purely from the mounted /model local snapshot (HF_HUB_OFFLINE=1 is set in the
image) and embeds a text. Any network attempt fails hard under --network none.
"""

import sys

import google.protobuf
import matplotlib
import sklearn

print("deps ok:", sklearn.__version__, matplotlib.__version__, google.protobuf.__version__)
assert google.protobuf.__version__.split(".")[0] < "7", "protobuf must be <7 (wandb pitfall)"

sys.path.insert(0, "/exp/scripts")
from embed_lib import embed_texts, load_model  # noqa: E402

model, processor = load_model("/model/Cosmos-Embed1-224p")
t = embed_texts(model, processor, ["the driver has taken both hands off the steering wheel"])
print(f"model loaded OFFLINE from /model snapshot; text emb shape {t.shape}, norm {float((t**2).sum())**0.5:.4f}")
print("GATE PASSED: offline image is self-sufficient")
