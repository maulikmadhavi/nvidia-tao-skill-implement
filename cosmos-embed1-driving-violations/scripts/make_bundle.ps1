# Build the air-gap migration bundle into bundle\ (run on the INTERNET machine).
#
#   .\make_bundle.ps1              # manifest dry-run (no 20GB docker save)
#   .\make_bundle.ps1 -Full        # also docker save the offline image (~20+GB)
#
# Bundle contents:
#   cosmos-embed-offline.tar   docker image with baked deps  (-Full only)
#   model\                     FULL local snapshots: Cosmos-Embed1-224p (incl.
#                              model_converted.pth — the CLI needs it and does NOT
#                              read the HF hub cache) + bert-base-uncased
#   kit\                       this experiment folder (scripts/specs/demo/deploy/selftest/docs)
#   MANIFEST.txt               sizes + SHA256 of the tar

param(
    [switch]$Full,
    [string]$ModelSource = ""
)

$ErrorActionPreference = "Stop"
$Exp = Split-Path -Parent $PSScriptRoot
$Bundle = Join-Path $Exp "bundle"
New-Item -ItemType Directory -Force $Bundle | Out-Null
$lines = @("bundle built: $(Get-Date -Format s)", "image: cosmos-embed-offline:7.0.1", "")

# 1. docker image
if ($Full) {
    $tar = Join-Path $Bundle "cosmos-embed-offline.tar"
    Write-Host "docker save (this is 20+GB, takes a while) ..."
    docker save -o $tar cosmos-embed-offline:7.0.1
    if ($LASTEXITCODE -ne 0) { throw "docker save failed" }
    $sha = (Get-FileHash $tar -Algorithm SHA256).Hash
    $lines += "cosmos-embed-offline.tar  $([math]::Round((Get-Item $tar).Length/1GB,2)) GB  sha256=$sha"
} else {
    $lines += "cosmos-embed-offline.tar  SKIPPED (dry-run; rerun with -Full)"
}

# 2. full model snapshots (populate first with scripts\fetch_snapshots.py)
if (-not $ModelSource) { $ModelSource = Join-Path $Exp "workspace\model" }
if ((Test-Path (Join-Path $ModelSource "Cosmos-Embed1-224p")) -and (Test-Path (Join-Path $ModelSource "bert-base-uncased"))) {
    Write-Host "copying model snapshots from $ModelSource ..."
    robocopy $ModelSource (Join-Path $Bundle "model") /E /NFL /NDL /NJH /NJS | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy model failed ($LASTEXITCODE)" }
    $size = (Get-ChildItem -Recurse (Join-Path $Bundle "model") | Measure-Object Length -Sum).Sum
    $lines += "model\  $([math]::Round($size/1GB,2)) GB (Cosmos-Embed1-224p full snapshot + bert-base-uncased)"
} else {
    $lines += "model\  MISSING: run scripts\fetch_snapshots.py first (needs internet once)"
}

# 3. experiment kit (code + docs, no workspace outputs)
Write-Host "copying kit ..."
$kit = Join-Path $Bundle "kit"
foreach ($d in @("scripts", "specs", "demo", "deploy", "selftest")) {
    robocopy (Join-Path $Exp $d) (Join-Path $kit $d) /E /NFL /NDL /NJH /NJS | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy $d failed" }
}
Get-ChildItem $Exp -File -Filter "*.md" | Copy-Item -Destination $kit
Copy-Item (Join-Path $Exp "Dockerfile.offline") $kit
$lines += "kit\  (scripts, specs, demo, deploy, selftest, docs, Dockerfile.offline)"

$lines | Set-Content (Join-Path $Bundle "MANIFEST.txt") -Encoding utf8
Write-Host "----"
Get-Content (Join-Path $Bundle "MANIFEST.txt")
exit 0   # robocopy's non-zero "files copied" codes are not failures
