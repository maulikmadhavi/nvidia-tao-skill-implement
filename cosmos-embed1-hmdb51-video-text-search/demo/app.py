"""Video-text search demo over the HMDB51 TEST split only (container-side).

Zero-dependency web app: Python stdlib http.server + one embedded HTML page.
(The TAO container's pinned httpx/httpcore make `pip install gradio` fail with
ResolutionImpossible, so no third-party UI framework is used.)

Loads the deployed fine-tuned model from /exp/deploy/model and the precomputed
test-set index from /results/deploy/index.npz (build with search_cli.py index).
Only the text query is embedded at request time.

Launch (host):
  scripts\run_container.ps1 -Cmd "python /exp/demo/app.py" -Ports "7860:7860"
Then open http://127.0.0.1:7860 (IPv4 — Docker Desktop's ::1 proxy drops connections)
"""

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "/exp/scripts")
from embed_lib import embed_texts, load_model  # noqa: E402

MODEL_PATH = "/exp/deploy/model"
INDEX_PATH = "/results/deploy/index.npz"
VIDEO_DIR = Path("/data/video")
PORT = 7860

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cosmos-Embed1 HMDB51 video search</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; margin: 0; background: #101418; color: #e6e8ea; }
  header { padding: 20px 28px 8px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #9aa4ad; font-size: 13px; }
  .bar { display: flex; gap: 10px; padding: 14px 28px; align-items: center; flex-wrap: wrap; }
  input[type=text] { flex: 1 1 380px; padding: 10px 14px; font-size: 15px; border-radius: 8px;
                     border: 1px solid #2c3844; background: #1a2129; color: #e6e8ea; }
  select, button { padding: 10px 14px; font-size: 14px; border-radius: 8px;
                   border: 1px solid #2c3844; background: #1a2129; color: #e6e8ea; }
  button { background: #2f6fed; border-color: #2f6fed; cursor: pointer; font-weight: 600; }
  button:disabled { opacity: 0.5; }
  .examples { padding: 0 28px 6px; font-size: 13px; color: #9aa4ad; }
  .examples a { color: #7fb0ff; margin-right: 12px; cursor: pointer; text-decoration: none; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 16px; padding: 16px 28px 40px; }
  .card { background: #1a2129; border: 1px solid #232d38; border-radius: 10px; overflow: hidden; }
  .card video { width: 100%; display: block; background: #000; }
  .meta { padding: 10px 12px; font-size: 13px; }
  .rank { color: #7fb0ff; font-weight: 700; margin-right: 6px; }
  .score { float: right; color: #77d38f; font-variant-numeric: tabular-nums; }
  .cap { color: #9aa4ad; margin-top: 3px; }
  #status { padding: 0 28px; color: #9aa4ad; font-size: 13px; min-height: 18px; }
</style>
</head>
<body>
<header>
  <h1>Cosmos-Embed1 video-text search &mdash; HMDB51 test split</h1>
  <div class="sub">Fine-tuned LoRA model &middot; index of 306 held-out test clips only &middot; query text embedded live</div>
</header>
<div class="bar">
  <input id="q" type="text" placeholder="a video of a person riding a bike" autofocus>
  <select id="k"><option>3</option><option selected>6</option><option>9</option><option>12</option></select>
  <button id="go" onclick="search()">Search</button>
</div>
<div class="examples" id="ex"></div>
<div id="status"></div>
<div id="grid"></div>
<script>
const EXAMPLES = ["a video of a person riding a bike","someone doing a cartwheel",
  "a person drinking from a cup","people sword fighting","a person climbing stairs",
  "shooting a bow and arrow"];
const ex = document.getElementById("ex");
ex.innerHTML = "try: " + EXAMPLES.map(e => `<a onclick="runExample('${e}')">${e}</a>`).join("");
function runExample(t) { document.getElementById("q").value = t; search(); }
document.getElementById("q").addEventListener("keydown", e => { if (e.key === "Enter") search(); });
async function search() {
  const q = document.getElementById("q").value.trim();
  if (!q) return;
  const k = document.getElementById("k").value;
  const btn = document.getElementById("go");
  const status = document.getElementById("status");
  btn.disabled = true; status.textContent = "encoding query + searching...";
  try {
    const r = await fetch(`/search?q=${encodeURIComponent(q)}&k=${k}`);
    const data = await r.json();
    status.textContent = `query: "${data.query}" - showing top ${data.results.length} of 306 test clips`;
    document.getElementById("grid").innerHTML = data.results.map(h => `
      <div class="card">
        <video src="/video/${h.file}" controls muted loop playsinline preload="metadata"></video>
        <div class="meta">
          <span class="rank">#${h.rank}</span>${h.video_id}
          <span class="score">${h.score.toFixed(4)}</span>
          <div class="cap">${h.caption}</div>
        </div>
      </div>`).join("");
  } catch (err) { status.textContent = "error: " + err; }
  btn.disabled = false;
}
</script>
</body>
</html>
"""


class SearchState:
    def __init__(self):
        z = np.load(INDEX_PATH, allow_pickle=False)
        self.embeddings = z["embeddings"]
        self.video_ids = [str(v) for v in z["video_ids"]]
        self.captions = [str(c) for c in z["captions"]]
        self.files = [str(f) for f in z["files"]]
        print(f"index: {len(self.video_ids)} test videos; loading model {MODEL_PATH} ...", flush=True)
        self.model, self.processor = load_model(MODEL_PATH)
        self.lock = threading.Lock()
        print("model ready", flush=True)

    def search(self, query: str, k: int) -> dict:
        with self.lock:  # one GPU, serialize encodes
            q = embed_texts(self.model, self.processor, [query])[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:k]
        return {"query": query, "results": [
            {"rank": r + 1, "video_id": self.video_ids[i], "file": self.files[i],
             "caption": self.captions[i], "score": float(scores[i])}
            for r, i in enumerate(order)]}


STATE: SearchState = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif parsed.path == "/search":
            params = urllib.parse.parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            k = max(1, min(24, int(params.get("k", ["6"])[0])))
            if not query:
                self._send(400, b'{"error": "empty query"}', "application/json")
                return
            result = STATE.search(query, k)
            self._send(200, json.dumps(result).encode(), "application/json")
        elif parsed.path.startswith("/video/"):
            name = Path(urllib.parse.unquote(parsed.path[len("/video/"):])).name
            fp = VIDEO_DIR / name
            if fp.suffix == ".mp4" and fp.is_file():
                self._send(200, fp.read_bytes(), "video/mp4")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):  # quieter logs
        if "/video/" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main() -> None:
    global STATE
    STATE = SearchState()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Running on http://0.0.0.0:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
