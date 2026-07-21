"""Shared Cosmos-Embed1 embedding helpers (container-side; needs torch/transformers).

Used by 10_extract_embeddings.py, deploy/search_cli.py and demo/app.py so the
model-API handling lives in exactly one place. Loads either the HF repo id
(nvidia/Cosmos-Embed1-224p) or an exported HF directory, both via
trust_remote_code=True.

API per the nvidia/Cosmos-Embed1-224p model card:
  video: numpy BTCHW batch -> processor(videos=batch) -> model.get_video_embeddings(**inputs).visual_proj
  text:  processor(text=[...]) -> model.get_text_embeddings(**inputs).text_proj
"""

import numpy as np
import torch

NUM_FRAMES = 8


def load_model(model_path: str, device: str = "cuda"):
    from transformers import AutoModel, AutoProcessor
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device, dtype=dtype).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def read_frames(video_path: str, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Uniformly sample num_frames RGB frames -> uint8 [T, H, W, C].

    Tries decord (what the model card uses), then PyAV, then OpenCV.
    """
    try:
        import decord
        vr = decord.VideoReader(video_path)
        idx = np.linspace(0, len(vr) - 1, num_frames, dtype=int).tolist()
        return vr.get_batch(idx).asnumpy()
    except ImportError:
        pass
    try:
        import av
        with av.open(video_path) as container:
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        return np.stack([frames[i] for i in idx])
    except ImportError:
        pass
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return np.stack([frames[i] for i in idx])


@torch.no_grad()
def embed_texts(model, processor, texts: list[str], device: str = "cuda") -> np.ndarray:
    """L2-normalized text embeddings [N, D]."""
    dtype = next(model.parameters()).dtype
    inputs = processor(text=texts).to(device, dtype=dtype)
    out = model.get_text_embeddings(**inputs)
    proj = torch.nn.functional.normalize(out.text_proj.float(), dim=-1)
    return proj.cpu().numpy()


@torch.no_grad()
def embed_video(model, processor, video_path: str, device: str = "cuda") -> np.ndarray:
    """L2-normalized single-video embedding [D]."""
    frames = read_frames(video_path)                       # [T, H, W, C] uint8
    batch = np.transpose(frames[None], (0, 1, 4, 2, 3))    # BTCHW, as in the model card
    dtype = next(model.parameters()).dtype
    inputs = processor(videos=batch).to(device, dtype=dtype)
    out = model.get_video_embeddings(**inputs)
    proj = torch.nn.functional.normalize(out.visual_proj.float(), dim=-1)
    return proj.squeeze(0).cpu().numpy()
