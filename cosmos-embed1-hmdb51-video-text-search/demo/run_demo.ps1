# Launch the web demo at http://localhost:7860 (test-split search only).
# Zero-dependency stdlib server (the container's pinned httpx/httpcore make
# pip install gradio fail with ResolutionImpossible).
# Prereqs: deploy\model\ populated (Phase 6) and /results/deploy/index.npz built
# (.\..\deploy\run_search.ps1 -Index).

$ErrorActionPreference = "Stop"
$Runner = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) "scripts\run_container.ps1"
& $Runner -Cmd "python /exp/demo/app.py" -Ports "7860:7860"
exit $LASTEXITCODE
