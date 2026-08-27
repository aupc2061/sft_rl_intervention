param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$DirectionArtifact,
    [Parameter(Mandatory = $true)][int[]]$Layers,
    [string]$Operation = "add",
    [double[]]$Scales = @(-1.0, -0.5, 0.0, 0.5, 1.0),
    [string]$OutputDirectory = "outputs/interventions"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

foreach ($layer in $Layers) {
    foreach ($scale in $Scales) {
        $safeScale = $scale.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture).Replace("-", "neg").Replace(".", "p")
        $output = Join-Path $OutputDirectory "layer${layer}_${Operation}_${safeScale}.json"
        mats-intervene --config $Config --checkpoint $Checkpoint --direction-artifact $DirectionArtifact --layer $layer --operation $Operation --scale $scale --output $output
    }
}

