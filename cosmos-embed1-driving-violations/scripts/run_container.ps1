# Canonical docker run wrapper (Windows) — driving-violations experiment.
# Uses the OFFLINE image (deps baked, no pip preamble, HF_HUB_OFFLINE=1 in image).
#
#   .\run_container.ps1 -Cmd "python /exp/scripts/10_infer_chunks.py ..."
#   .\run_container.ps1 -Cmd "python /exp/demo/app.py" -Ports "7860:7860"
# Optional -HfCache to point at an existing populated cache (e.g. the HMDB
# experiment's) instead of this experiment's workspace\hf_cache.

param(
    [Parameter(Mandatory = $true)][string]$Cmd,
    [string]$Ports = "",
    [string]$HfCache = "",
    [switch]$Detach
)

$ErrorActionPreference = "Stop"
$Image = "cosmos-embed-offline:7.0.1"
$Exp = Split-Path -Parent $PSScriptRoot
if (-not $HfCache) { $HfCache = "$Exp\workspace\hf_cache" }

$dockerArgs = @(
    "run", "--rm", "--gpus", "all", "--ipc=host", "--shm-size=16g",
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "-v", "$Exp\workspace\data:/data:ro",
    "-v", "$Exp\workspace\specs:/specs:ro",
    "-v", "$Exp\workspace\results:/results",
    "-v", "$Exp\workspace\model:/model",
    "-v", "${HfCache}:/hf_cache",
    "-v", "$Exp\scripts:/exp/scripts:ro",
    "-v", "$Exp\deploy:/exp/deploy",
    "-v", "$Exp\demo:/exp/demo:ro",
    "-v", "$Exp\selftest:/exp/selftest:ro"
)
if ($Ports) { $dockerArgs += @("-p", $Ports) }
if ($Detach) { $dockerArgs += "-d" }
$dockerArgs += @($Image, "bash", "-lc", $Cmd)

Write-Host "docker $($dockerArgs -join ' ')"
& docker @dockerArgs
exit $LASTEXITCODE
