param(
    [switch]$SkipFetch,
    [string]$Config = "configs/default.yaml",
    [string]$DataDir = "data",
    [string]$OutputDir = "outputs/latest"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repoRoot "src"
$args = @("run-all", "--config", $Config, "--data-dir", $DataDir, "--output-dir", $OutputDir)
if ($SkipFetch) {
    $args += "--skip-fetch"
}
python -m cfpipeline @args
