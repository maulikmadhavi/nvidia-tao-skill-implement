# Launch the violation timeline review demo at http://127.0.0.1:7860
# (use 127.0.0.1, NOT localhost — Docker Desktop drops ::1 connections).
#
#   .\run_demo.ps1                          # finetuned glue results (default)
#   .\run_demo.ps1 -Phase baseline -Mode isolated
param(
    [string]$Phase = "finetuned",
    [string]$Mode = "glue"
)
$ErrorActionPreference = "Stop"
$Runner = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) "scripts\run_container.ps1"
& $Runner -Cmd "python /exp/demo/app.py --scores /results/scores/$Phase --events /results/events/${Phase}_${Mode}/events.json --videos /data/full" -Ports "7860:7860"
exit $LASTEXITCODE
