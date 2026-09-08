# One-shot setup for local mode on Windows: checks Python, creates a venv, installs
# requirements-local.txt, checks for Ollama, then hands off to first_run.py for the
# config wizard + initial crawl.
#
# Usage (from an ordinary PowerShell prompt):
#   .\scripts\setup.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "== 1/6: Checking for Python 3.11+ =="

function Test-PythonVersion($exe, $preArgs) {
    try {
        $out = & $exe @preArgs "-c" "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $false }
        $parts = $out.Trim().Split(".")
        return ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 11)
    } catch {
        return $false
    }
}

$PythonExe = $null
$PythonArgs = @()
if (Get-Command python -ErrorAction SilentlyContinue) {
    if (Test-PythonVersion "python" @()) {
        $PythonExe = "python"
    }
}
if (-not $PythonExe -and (Get-Command py -ErrorAction SilentlyContinue)) {
    foreach ($ver in @("-3.13", "-3.12", "-3.11")) {
        if (Test-PythonVersion "py" @($ver)) {
            $PythonExe = "py"
            $PythonArgs = @($ver)
            break
        }
    }
}
if (-not $PythonExe) {
    Write-Host "Python 3.11+ not found."
    Write-Host "Install it from https://www.python.org/downloads/windows/ (check 'Add python.exe to PATH')"
    Write-Host "then re-run this script."
    exit 1
}
Write-Host "Using: $PythonExe $PythonArgs"

Write-Host ""
Write-Host "== 2/6: Creating virtual environment (.venv) =="
if (-not (Test-Path ".venv")) {
    & $PythonExe @PythonArgs "-m" "venv" ".venv"
}
. .\.venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "== 3/6: Installing dependencies (requirements-local.txt) =="
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements-local.txt

Write-Host ""
Write-Host "== 4/6: Checking for Ollama =="
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
    Write-Host "Found: $($ollama.Source)"
} else {
    Write-Host "Ollama not found. It's used for local generation and is optional:"
    Write-Host "  - Install it from https://ollama.com to answer questions with a local model, or"
    Write-Host "  - Skip it and set OPENAI_API_KEY instead (config.yaml or your environment) to use"
    Write-Host "    OpenAI, or skip both and search will still work (results just won't get a"
    Write-Host "    synthesized answer from /v1/ask -- /v1/find's file search works either way)."
}

Write-Host ""
Write-Host "== 5/6: First-run setup (config + initial index) =="
$env:PROFILE = "local"
python scripts\first_run.py

Write-Host ""
Write-Host "== 6/6: Done =="
