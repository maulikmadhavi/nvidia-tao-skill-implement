# End-to-end pipeline self-test on synthetic lessons (no real driving data needed).
# NOTE: occupies workspace\data — run BEFORE loading real data, or clean afterwards.
#
#   .\run_selftest.ps1        # zero-shot chain 00→01→10→14→11→13→12, then heads 15→16→14→11→13
#
# Gates: chunking counts, planted events recovered by glue (event F1 > 0 on the
# test lesson), chunk-level metrics produced.

$ErrorActionPreference = "Stop"
$Exp = Split-Path -Parent $PSScriptRoot
$S = "$Exp\scripts"
$ST = "$Exp\selftest\out"
$Runner = "$S\run_container.ps1"
$Tax = "$ST\taxonomy_selftest.json"

Write-Host "== [1/11] synthetic lessons =="
python "$Exp\selftest\make_synthetic.py" --out $ST
if ($LASTEXITCODE) { exit 1 }

Write-Host "== [2/11] chunk + label (00) =="
python "$S\00_chunk_videos.py" --videos "$ST\videos" --annotations "$ST\annotations.csv" `
    --out "$Exp\workspace\data" --taxonomy $Tax --split 34,33,33 --neg-ratio 4 --seed 42
if ($LASTEXITCODE) { exit 1 }
Copy-Item "$ST\annotations.csv" "$Exp\workspace\data\annotations.csv" -Force
Copy-Item $Tax "$Exp\workspace\data\taxonomy_selftest.json" -Force

Write-Host "== [3/11] metadata + specs (01) =="
python "$S\01_make_metadata.py" --data "$Exp\workspace\data" --taxonomy $Tax `
    --templates "$Exp\specs" --specs "$Exp\workspace\specs"
if ($LASTEXITCODE) { exit 1 }

Write-Host "== [4/11] score all lessons (10, container, offline) =="
& $Runner -Cmd "python /exp/scripts/10_infer_chunks.py --model /model/Cosmos-Embed1-224p --videos /data/full --list all --prompts /data/prompts.json --out /results/scores/selftest"
if ($LASTEXITCODE) { exit 1 }

Write-Host "== [5/11] tune thresholds on val lesson (14) =="
python "$S\14_tune_thresholds.py" --scores "$Exp\workspace\results\scores\selftest" `
    --annotations "$Exp\workspace\data\annotations.csv" --videos "$Exp\workspace\data\val_videos.txt" `
    --out "$Exp\workspace\results\thresholds_selftest.json"
if ($LASTEXITCODE) { exit 1 }

Write-Host "== [6/11] glue + isolated post-processing (11) =="
python "$S\11_glue_postprocess.py" --scores "$Exp\workspace\results\scores\selftest" `
    --prompts "$Exp\workspace\data\prompts.json" --thresholds "$Exp\workspace\results\thresholds_selftest.json" `
    --mode glue --out "$Exp\workspace\results\events\selftest_glue"
if ($LASTEXITCODE) { exit 1 }
python "$S\11_glue_postprocess.py" --scores "$Exp\workspace\results\scores\selftest" `
    --prompts "$Exp\workspace\data\prompts.json" --thresholds "$Exp\workspace\results\thresholds_selftest.json" `
    --mode isolated --out "$Exp\workspace\results\events\selftest_isolated"
if ($LASTEXITCODE) { exit 1 }

Write-Host "== [7/11] event-level eval on test lesson (13) =="
python "$S\13_eval_event_level.py" --events "$Exp\workspace\results\events\selftest_glue\events.json" `
    --annotations "$Exp\workspace\data\annotations.csv" --videos "$Exp\workspace\data\test_videos.txt" `
    --label "selftest/zero-shot/glue" --out "$Exp\workspace\results\event_eval_selftest_glue.json"
if ($LASTEXITCODE) { exit 1 }

Write-Host "== [8/11] chunk-level eval (12, container) =="
& $Runner -Cmd "python /exp/scripts/12_eval_chunk_level.py --scores /results/scores/selftest --annotations /data/annotations.csv --videos /data/test_videos.txt --thresholds /results/thresholds_selftest.json --phase selftest-zeroshot --out /results/chunk_eval_selftest"
if ($LASTEXITCODE) { exit 1 }

$f1 = (Get-Content "$Exp\workspace\results\event_eval_selftest_glue.json" | ConvertFrom-Json).micro.f1
if ($f1 -gt 0) {
    Write-Host "SELFTEST GATE PASSED: glue recovered planted events (micro F1 = $f1)"
} else {
    Write-Host "SELFTEST GATE FAILED: micro F1 = 0 (no planted event recovered)"; exit 1
}

Write-Host "== [9/11] fit heads on train lesson (15, container) =="
& $Runner -Cmd "python /exp/scripts/15_train_heads.py --scores /results/scores/selftest --annotations /data/annotations.csv --train-videos /data/train_videos.txt --val-videos /data/val_videos.txt --out /results/heads/selftest_heads.npz --report /results/heads/selftest_heads_report.json"
if ($LASTEXITCODE) { exit 1 }

Write-Host "== [10/11] head probabilities + unknown queue (16, container) =="
& $Runner -Cmd "python /exp/scripts/16_score_heads.py --scores /results/scores/selftest --heads /results/heads/selftest_heads.npz --out /results/scores/selftest_heads"
if ($LASTEXITCODE) { exit 1 }

Write-Host "== [11/11] heads: tune (14, probability grid) -> glue (11) -> event eval (13) =="
python "$S\14_tune_thresholds.py" --scores "$Exp\workspace\results\scores\selftest_heads" `
    --annotations "$Exp\workspace\data\annotations.csv" --videos "$Exp\workspace\data\val_videos.txt" `
    --thr-grid 0.05:0.95:0.02 `
    --out "$Exp\workspace\results\thresholds_selftest_heads.json"
if ($LASTEXITCODE) { exit 1 }
python "$S\11_glue_postprocess.py" --scores "$Exp\workspace\results\scores\selftest_heads" `
    --prompts "$Exp\workspace\data\prompts.json" --thresholds "$Exp\workspace\results\thresholds_selftest_heads.json" `
    --mode glue --out "$Exp\workspace\results\events\selftest_heads_glue"
if ($LASTEXITCODE) { exit 1 }
python "$S\13_eval_event_level.py" --events "$Exp\workspace\results\events\selftest_heads_glue\events.json" `
    --annotations "$Exp\workspace\data\annotations.csv" --videos "$Exp\workspace\data\test_videos.txt" `
    --label "selftest/heads/glue" --out "$Exp\workspace\results\event_eval_selftest_heads_glue.json"
if ($LASTEXITCODE) { exit 1 }

$f1h = (Get-Content "$Exp\workspace\results\event_eval_selftest_heads_glue.json" | ConvertFrom-Json).micro.f1
$reportOk = Test-Path "$Exp\workspace\results\heads\selftest_heads_report.json"
if ($reportOk -and $f1h -gt 0) {
    Write-Host "SELFTEST HEADS GATE PASSED: heads glue recovered planted events (micro F1 = $f1h)"
} else {
    Write-Host "SELFTEST HEADS GATE FAILED: heads report missing or heads glue micro F1 = 0"; exit 1
}
