param(
    [string]$Config = "configs/exp1_qwen_gsm8k.yaml",
    [int[]]$Seeds = @(1, 2, 3)
)

$ErrorActionPreference = "Stop"

foreach ($seed in $Seeds) {
    mats-train-sft --config $Config --seed $seed
    mats-train-grpo --config $Config --seed $seed
}

Write-Host "Training submitted/completed. Evaluate every saved checkpoint to construct Pareto fronts."

