param(
  [string]$Python = "python",
  [string]$DataRoot = "",
  [string]$OutputRoot = "",
  [string]$DeviceIds = "0"
)
$ErrorActionPreference = "Stop"
$CodeRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $DataRoot) { $DataRoot = Join-Path $CodeRoot "data\full" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $CodeRoot "artifacts\extended_baselines" }
if (!(Test-Path -LiteralPath (Join-Path $DataRoot "train\train_data.npy"))) {
  throw "Full benchmark data are required at DataRoot=$DataRoot"
}

foreach ($Seed in 44..48) {
  foreach ($Model in @("temdnet_style", "temnlnet_style", "joint_gated_moe")) {
    if ($Model -eq "joint_gated_moe") { $Microbatch = 8; $Accumulation = 16 }
    else { $Microbatch = 64; $Accumulation = 2 }
    & $Python (Join-Path $CodeRoot "training\train_external_baselines.py") `
      --model $Model `
      --train-data (Join-Path $DataRoot "train\train_data.npy") `
      --train-labels (Join-Path $DataRoot "train\train_labels.npy") `
      --val-data (Join-Path $DataRoot "val\val_data.npy") `
      --val-labels (Join-Path $DataRoot "val\val_labels.npy") `
      --test-data (Join-Path $DataRoot "test\test_data.npy") `
      --test-labels (Join-Path $DataRoot "test\test_labels.npy") `
      --data-root $DataRoot `
      --output-dir (Join-Path $OutputRoot "$Model\seed_$Seed") `
      --batch-size $Microbatch --grad-accum-steps $Accumulation --num-workers 4 `
      --max-epochs 120 --min-epochs 40 --patience 25 `
      --selection-metric rrmse --device cuda --device-ids $DeviceIds --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "Baseline failed: $Model seed $Seed" }
  }
}
