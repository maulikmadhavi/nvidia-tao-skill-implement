# docker run wrapper (Windows) — driving-distracted clip-classification experiment.
# Uses the OFFLINE cosmos-embed image (deps baked). Reuses the model snapshot and
# hf_cache already living in the driving-violations experiment so we don't
# duplicate 7.6 GB. The raw dataset is mounted read-only at /data.
#
#   .\scripts\run_container.ps1 -Cmd "python /exp/scripts/10_embed.py ..."
#
# Mounts:
#   /data    dataset root (shortclips_pos_neg)           ro
#   /splits  workspace\splits (metadata + prompts)       ro
#   /results workspace\results                           rw
#   /model   shared Cosmos-Embed1-224p snapshot          ro
#   /exp/scripts  this experiment's scripts              ro

param(
    [Parameter(Mandatory = $true)][string]$Cmd,
    [string]$DataRoot = "D:\research_data\driving_violation\shortclips_pos_neg\shortclips_pos_neg"
)

$ErrorActionPreference = "Stop"
$Image = "cosmos-embed-offline:7.0.1"
$Exp = Split-Path -Parent $PSScriptRoot
$Shared = "D:\tao-skill-bank\exp\cosmos-embed1-driving-violations\workspace"

$dockerArgs = @(
    "run", "--rm", "--gpus", "all", "--ipc=host", "--shm-size=16g",
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "-v", "${DataRoot}:/data:ro",
    "-v", "$Exp\workspace\splits:/splits:ro",
    "-v", "$Exp\workspace\results:/results",
    "-v", "$Shared\model:/model:ro",
    "-v", "$Shared\hf_cache:/hf_cache",
    "-v", "$Exp\scripts:/exp/scripts:ro",
    $Image, "bash", "-lc", $Cmd
)
Write-Host "docker $($dockerArgs -join ' ')"
& docker @dockerArgs
exit $LASTEXITCODE
