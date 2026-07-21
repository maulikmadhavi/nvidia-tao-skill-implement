"""Shared Cosmos-Embed1 embedding helpers (container-side; needs torch/transformers).

Same verified API as the HMDB experiment (model card: numpy BTCHW batch ->
processor(videos=...) -> get_video_embeddings().visual_proj; text ->
get_text_embeddings().text_proj), extended with sliding-window batch embedding
over a full lesson video for trajectory scoring.
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


def read_all_frames(video_path: str):
    """Decode a full video -> (frames [N,H,W,C] uint8 accessor, fps, n_frames).

    decord keeps frames lazily (memory-safe for 5-min videos); PyAV/OpenCV
    fallbacks load eagerly.
    """
    try:
        import decord
        vr = decord.VideoReader(video_path)
        fps = float(vr.get_avg_fps())
        return vr, fps, len(vr)
    except ImportError:
        pass
    try:
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
        return frames, fps, len(frames)
    except ImportError:
        pass
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()
    return np.stack(frames), fps, len(frames)


def _gather(reader, idx: np.ndarray) -> np.ndarray:
    """[len(idx), H, W, C] uint8 from a decord reader or an ndarray."""
    if isinstance(reader, np.ndarray):
        return reader[idx]
    return reader.get_batch(idx.tolist()).asnumpy()


@torch.no_grad()
def embed_texts(model, processor, texts: list[str], device: str = "cuda") -> np.ndarray:
    dtype = next(model.parameters()).dtype
    inputs = processor(text=texts).to(device, dtype=dtype)
    out = model.get_text_embeddings(**inputs)
    return torch.nn.functional.normalize(out.text_proj.float(), dim=-1).cpu().numpy()


@torch.no_grad()
def embed_frame_batch(model, processor, frames_btchw_uint8: np.ndarray, device: str = "cuda") -> np.ndarray:
    """L2-normalized embeddings [B, D] from a uint8 [B, T, H, W, C] batch."""
    batch = np.transpose(frames_btchw_uint8, (0, 1, 4, 2, 3))  # BTCHW
    dtype = next(model.parameters()).dtype
    inputs = processor(videos=batch).to(device, dtype=dtype)
    out = model.get_video_embeddings(**inputs)
    return torch.nn.functional.normalize(out.visual_proj.float(), dim=-1).cpu().numpy()


@torch.no_grad()
def embed_video(model, processor, video_path: str, device: str = "cuda",
                num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Whole-clip single embedding [D] (uniform frames over the full clip)."""
    reader, _, n = read_all_frames(video_path)
    idx = np.linspace(0, n - 1, num_frames).round().astype(int)
    frames = _gather(reader, idx)[None]  # [1,T,H,W,C]
    return embed_frame_batch(model, processor, frames, device)[0]


@torch.no_grad()
def embed_video_windows(model, processor, video_path: str, chunk_sec: float = 2.0,
                        stride_sec: float = 1.0, batch_size: int = 8,
                        device: str = "cuda", num_frames: int = NUM_FRAMES):
    """Sliding-window embeddings over a full video.

    Returns (embs [W, D], starts [W] window start seconds).
    """
    reader, fps, n = read_all_frames(video_path)
    duration = n / fps
    starts = []
    t = 0.0
    while t + chunk_sec <= duration + 1e-6:
        starts.append(round(t, 3))
        t += stride_sec
    if not starts:
        starts = [0.0]

    embs = []
    for i in range(0, len(starts), batch_size):
        group = starts[i:i + batch_size]
        clips = []
        for t0 in group:
            f0 = int(t0 * fps)
            f1 = min(int((t0 + chunk_sec) * fps), n - 1)
            idx = np.linspace(f0, max(f1, f0), num_frames).round().astype(int)
            clips.append(_gather(reader, idx))
        embs.append(embed_frame_batch(model, processor, np.stack(clips), device))
    return np.concatenate(embs, axis=0), np.array(starts, dtype=np.float64)
