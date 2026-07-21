# Deployable search wrapper. Builds the index once, then searches.
#
#   .\run_search.ps1 -Index                        # build /results/deploy/index.npz
#   .\run_search.ps1 -Query "a video of a person riding a bike" [-TopK 5] [-Json]

param(
    [switch]$Index,
    [string]$Query = "",
    [int]$TopK = 5,
    [switch]$Json
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) "scripts\run_container.ps1"
$Model = "/exp/deploy/model"

if ($Index) {
    & $Runner -Cmd "python /exp/deploy/search_cli.py index --model $Model --metadata /data/test.json --videos /data/video --out /results/deploy/index.npz"
} elseif ($Query) {
    $jsonFlag = ""
    if ($Json) { $jsonFlag = " --json" }
    & $Runner -Cmd "python /exp/deploy/search_cli.py search --model $Model --index /results/deploy/index.npz --query \`"$Query\`" --topk $TopK$jsonFlag"
} else {
    Write-Host "usage: .\run_search.ps1 -Index | -Query '<text>' [-TopK 5] [-Json]"
    exit 1
}
exit $LASTEXITCODE
