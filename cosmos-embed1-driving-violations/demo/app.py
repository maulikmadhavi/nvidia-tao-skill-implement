"""Violation timeline review demo (stdlib only, no GPU/model needed).

Shows, per lesson video: the per-class score trajectories (SVG), detected event
bars (glue output), and the video itself — clicking the timeline seeks the
video to that moment. Reads only precomputed artifacts:
  --scores  dir with scores_<stem>.csv   (from 10_infer_chunks.py)
  --events  events.json                  (from 11_glue_postprocess.py)
  --videos  dir with full lesson mp4s    (workspace/data/full)

Launch in-container (host wrapper):
  run_container.ps1/.sh "python /exp/demo/app.py --scores /results/scores/finetuned \
      --events /results/events/finetuned_glue/events.json --videos /data/full" -Ports 7860:7860
Then open http://127.0.0.1:7860 (IPv4 — Docker Desktop drops ::1).
"""

import argparse
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, "/exp/scripts")
from glue_lib import read_scores_csv  # noqa: E402

COLORS = ["#e05c5c", "#e0a75c", "#c6e05c", "#5ce08c", "#5cc6e0", "#8c5ce0", "#9aa4ad"]

INDEX_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Violation review</title>
<style>body{{font-family:system-ui;background:#101418;color:#e6e8ea;padding:28px}}
a{{color:#7fb0ff;text-decoration:none;display:block;padding:6px 0;font-size:15px}}</style></head>
<body><h1>Driving-lesson violation review</h1><p>{n} videos with score trajectories:</p>{links}</body></html>"""

VIEW_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>{stem}</title>
<style>
 body{{font-family:system-ui;background:#101418;color:#e6e8ea;margin:0;padding:20px 28px}}
 h1{{font-size:18px}} .row{{display:flex;gap:20px;flex-wrap:wrap}}
 video{{width:520px;max-width:95vw;background:#000;border-radius:8px}}
 #chart{{background:#1a2129;border:1px solid #232d38;border-radius:8px}}
 .legend span{{display:inline-block;margin-right:14px;font-size:12px}}
 .legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}}
 .ev{{font-size:13px;color:#9aa4ad}} a{{color:#7fb0ff}}
</style></head><body>
<a href="/">&larr; all videos</a><h1>{stem}</h1>
<div class="row">
 <div><video id="v" src="/video/{stem}.mp4" controls muted></video></div>
 <div>
  <svg id="chart" width="700" height="360"></svg>
  <div class="legend" id="legend"></div>
  <div class="ev" id="evlist"></div>
 </div>
</div>
<script>
const W=700,H=300,PAD=36;
fetch('/traj/{stem}').then(r=>r.json()).then(d=>{{
 const svg=document.getElementById('chart');
 const tmax=d.starts[d.starts.length-1]+d.chunk_sec;
 const classes=Object.keys(d.classes);
 const x=t=>PAD+(W-2*PAD)*t/tmax, y=s=>H-PAD-(H-2*PAD)*Math.max(0,Math.min(1,(s+0.1)/0.7));
 let g='';
 // axes
 g+=`<line x1="${{PAD}}" y1="${{H-PAD}}" x2="${{W-PAD}}" y2="${{H-PAD}}" stroke="#39434d"/>`;
 for(let t=0;t<=tmax;t+=30) g+=`<text x="${{x(t)}}" y="${{H-PAD+14}}" fill="#9aa4ad" font-size="9" text-anchor="middle">${{t}}s</text>`;
 // event bars (stacked above axis)
 let bar=0;
 for(const c of classes){{
  const evs=(d.events[c]||[]);
  for(const e of evs) g+=`<rect x="${{x(e.start_sec)}}" y="${{H-PAD+18+bar*8}}" width="${{Math.max(2,x(e.end_sec)-x(e.start_sec))}}" height="6" fill="${{d.colors[c]}}" opacity="0.9"><title>${{c}} ${{e.start_sec}}-${{e.end_sec}}s peak ${{e.peak}}</title></rect>`;
  if(evs.length) bar++;
 }}
 // trajectories
 for(const c of classes){{
  const pts=d.starts.map((t,i)=>`${{x(t)}},${{y(d.classes[c][i])}}`).join(' ');
  g+=`<polyline points="${{pts}}" fill="none" stroke="${{d.colors[c]}}" stroke-width="1.5" opacity="0.9"/>`;
 }}
 g+=`<line id="cursor" x1="${{PAD}}" y1="${{PAD}}" x2="${{PAD}}" y2="${{H-PAD}}" stroke="#fff" opacity="0.6"/>`;
 svg.setAttribute('height',H-PAD+30+bar*8);
 svg.innerHTML=g;
 document.getElementById('legend').innerHTML=classes.map(c=>`<span><i style="background:${{d.colors[c]}}"></i>${{c}}</span>`).join('');
 const evl=[];
 for(const c of classes) for(const e of (d.events[c]||[])) evl.push(`${{c}}: ${{e.start_sec}}s → ${{e.end_sec}}s (peak ${{e.peak.toFixed(3)}})`);
 document.getElementById('evlist').innerHTML='<b>events:</b><br>'+(evl.join('<br>')||'none');
 const video=document.getElementById('v');
 svg.addEventListener('click',ev=>{{
  const r=svg.getBoundingClientRect();
  const t=Math.max(0,Math.min(tmax,(ev.clientX-r.left-PAD)/(W-2*PAD)*tmax));
  video.currentTime=t; video.play();
 }});
 video.addEventListener('timeupdate',()=>{{
  const c=document.getElementById('cursor');
  c.setAttribute('x1',x(video.currentTime)); c.setAttribute('x2',x(video.currentTime));
 }});
}});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    stems = sorted(p.stem[len("scores_"):] for p in args.scores.glob("scores_*.csv"))
    events_all = json.loads(args.events.read_text(encoding="utf-8")) if args.events.exists() else {}
    print(f"review server: {len(stems)} videos")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = urllib.parse.urlparse(self.path).path
            if path == "/":
                links = "".join(f'<a href="/view/{s}">{s}</a>' for s in stems)
                self._send(200, INDEX_PAGE.format(n=len(stems), links=links).encode(), "text/html; charset=utf-8")
            elif path.startswith("/view/"):
                stem = Path(path[len("/view/"):]).name
                self._send(200, VIEW_PAGE.format(stem=stem).encode(), "text/html; charset=utf-8")
            elif path.startswith("/traj/"):
                stem = Path(path[len("/traj/"):]).name
                starts, cols = read_scores_csv(args.scores / f"scores_{stem}.csv")
                classes = {c: t for c, t in cols.items()}
                colors = {c: COLORS[i % len(COLORS)] for i, c in enumerate(classes)}
                payload = {"starts": starts, "classes": classes, "colors": colors,
                           "chunk_sec": 2.0, "events": events_all.get(stem, {})}
                self._send(200, json.dumps(payload).encode(), "application/json")
            elif path.startswith("/video/"):
                fp = args.videos / Path(urllib.parse.unquote(path[len("/video/"):])).name
                if fp.suffix == ".mp4" and fp.is_file():
                    self._send(200, fp.read_bytes(), "video/mp4")
                else:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, fmt, *a):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Running on http://0.0.0.0:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
