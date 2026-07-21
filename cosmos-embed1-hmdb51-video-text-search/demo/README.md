# Web demo — video-text search on the HMDB51 test split

Search box + top-k selector -> grid of playable test videos with cosine
matching scores. Uses ONLY the 306 held-out test clips (never training data);
video embeddings are precomputed, the query text is embedded live by the
fine-tuned model.

Implementation note: a zero-dependency Python stdlib HTTP server with an
embedded HTML page — `pip install gradio` fails inside the TAO container
(ResolutionImpossible on its pinned httpx/httpcore), so no UI framework is used.

## Run

```powershell
# prereqs: deploy\model\ populated (Phase 6) and the index built:
..\deploy\run_search.ps1 -Index

.\run_demo.ps1
# then open http://127.0.0.1:7860   (use 127.0.0.1, NOT localhost — Docker
# Desktop's IPv6 proxy on this host accepts ::1 connections and drops them)
```

API: `GET /search?q=<text>&k=<n>` returns JSON `{query, results:[{rank, video_id, file, caption, score}]}`.
