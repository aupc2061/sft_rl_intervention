param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$RunDirectory
)

$ErrorActionPreference = "Stop"
$checkpointRoot = Join-Path $RunDirectory "checkpoints"
$checkpoints = Get-ChildItem -Directory $checkpointRoot | Where-Object {
    Test-Path (Join-Path $_.FullName "adapter_config.json")
}

if (-not $checkpoints) {
    throw "No PEFT checkpoints containing adapter_config.json found under $checkpointRoot"
}

foreach ($checkpoint in $checkpoints) {
    $output = Join-Path $RunDirectory ("evaluation_" + $checkpoint.Name + ".json")
    mats-evaluate --config $Config --checkpoint $checkpoint.FullName --output $output
}

