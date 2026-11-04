# Rafidain Windows compatibility installer — uv-first.
# Usage: powershell -ExecutionPolicy Bypass -File .\seed_vc_iraqi_windows.ps1 -Engine mock
[CmdletBinding()]
param(
  [ValidateSet("auto","seedvc","mock")][string]$Engine = "auto",
  [int]$Port = 8000,
  [switch]$NoTunnel,
  [switch]$NoServe,
  [switch]$NoModels
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "uv غير موجود؛ ثبّته من https://docs.astral.sh/uv/getting-started/installation/ ثم أعد التشغيل." -ForegroundColor Yellow
  exit 1
}
if ($Engine -eq "auto") {
  $Report = (& uv run --python 3.11 python scripts/manage.py detect | ConvertFrom-Json)
  if ($null -ne $Report.gpu) { $Engine = "seedvc" } else { $Engine = "mock" }
  Write-Host "engine=auto -> $Engine"
}
& uv run --python 3.11 python scripts/manage.py install --engine $Engine
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($NoServe) { exit 0 }
$env:RAFIDAIN__PORT = "$Port"
if ($Engine -eq "mock") { & uv run --python 3.11 python scripts/manage.py web --mock --port $Port }
else { & uv run --python 3.11 python scripts/manage.py web --port $Port }
if (-not $NoTunnel) { & uv run --python 3.11 python scripts/manage.py tunnel start --port $Port }
