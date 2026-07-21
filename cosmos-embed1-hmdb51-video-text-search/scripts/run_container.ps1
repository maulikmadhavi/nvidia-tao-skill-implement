# Canonical docker run wrapper for the cosmos-embed experiment.
# Single source of truth for image + mounts + flags so they never drift.
#
# Usage:
#   .\run_container.ps1 -Cmd "cosmos-embed1 evaluate -e /specs/evaluate_zeroshot.yaml results_dir=/results/evaluate_zeroshot"
#   .\run_container.ps1 -Cmd "python /exp/demo/app.py" -Ports "7860:7860"
#
# Deviations from the skill's DOCKER_COMMON (deliberate, host = Windows 11 / 32GB RAM):
#   --shm-size=16g (not 64g); no --network=host (unsupported semantics on Docker Desktop;
#   use -Ports for the demo). protobuf<7 preamble is mandatory (wandb import pitfall).

param(
    [Parameter(Mandatory = $true)][string]$Cmd,
    [string]$Ports = "",
    [switch]$Detach
)

$ErrorActionPreference = "Stop"
$Image = "nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-embed"
$Exp = Split-Path -Parent $PSScriptRoot   # experiment root (this file lives in scripts\)

$dockerArgs = @(
    "run", "--rm", "--gpus", "all", "--ipc=host", "--shm-size=16g",
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "-e", "HF_TOKEN",
    "-e", "WANDB_DISABLED=true",
    "-e", "WANDB_MODE=disabled",
    "-e", "HUGGINGFACE_HUB_CACHE=/hf_cache",
    "-v", "$Exp\workspace\data:/data:ro",
    "-v", "$Exp\workspace\specs:/specs:ro",
    "-v", "$Exp\workspace\results:/results",
    "-v", "$Exp\workspace\model:/model",
    "-v", "$Exp\workspace\hf_cache:/hf_cache",
    "-v", "$Exp\scripts:/exp/scripts:ro",
    "-v", "$Exp\deploy:/exp/deploy",
    "-v", "$Exp\demo:/exp/demo:ro"
)
if ($Ports) { $dockerArgs += @("-p", $Ports) }
if ($Detach) { $dockerArgs += "-d" }
$dockerArgs += @($Image, "bash", "-lc", "python -m pip install --quiet 'protobuf<7' && $Cmd")

Write-Host "docker $($dockerArgs -join ' ')"
& docker @dockerArgs
exit $LASTEXITCODE
