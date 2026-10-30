param(
  [string]$Python = "python",
  [string]$DataRoot = "",
  [string]$OutputRoot = "",
  [string]$DeviceIds = "0",
  [string]$GeometryName = "all"
)
$ErrorActionPreference = "Stop"
$Code = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $DataRoot) { $DataRoot = Join-Path $Code "data\full" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $Code "artifacts\receiver_geometry" }
$SourceData = $DataRoot
$WorkRoot = $OutputRoot
if (!(Test-Path -LiteralPath (Join-Path $SourceData "train\train_data.npy"))) {
  throw "Full benchmark data are required at DataRoot=$SourceData"
}
$Logs = Join-Path $WorkRoot "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$Geometries = @(
  @{ Name = "n5_s50";  Indices = "3,4,5,6,7";       Count = 5 },
  @{ Name = "n7_s50";  Indices = "2,3,4,5,6,7,8";   Count = 7 },
  @{ Name = "n5_s100"; Indices = "1,3,5,7,9";       Count = 5 }
)

foreach ($Geometry in $Geometries) {
  if ($GeometryName -ne "all" -and $Geometry.Name -ne $GeometryName) { continue }
  $DriverLog = Join-Path $Logs "driver_$($Geometry.Name).log"
  $GeometryDataRoot = Join-Path $WorkRoot "data\$($Geometry.Name)"
  $Manifest = Join-Path $GeometryDataRoot "receiver_geometry_manifest.json"
  if (!(Test-Path -LiteralPath $Manifest)) {
    "[$(Get-Date -Format s)] PREP $($Geometry.Name)" | Tee-Object $DriverLog -Append
    & $Python (Join-Path $Code "scripts\prepare_receiver_geometry_ablation.py") --source-root $SourceData --output-root $GeometryDataRoot --indices $Geometry.Indices
    if ($LASTEXITCODE -ne 0) { throw "Data preparation failed for $($Geometry.Name)" }
  }

  foreach ($Seed in 44..46) {
    $BaselineRoot = Join-Path $WorkRoot "runs\$($Geometry.Name)\seed_$Seed\baselines"
    $FinalRoot = Join-Path $WorkRoot "runs\$($Geometry.Name)\seed_$Seed\final"
    if (!(Test-Path -LiteralPath (Join-Path $BaselineRoot "suite_summary.json"))) {
      "[$(Get-Date -Format s)] START baselines $($Geometry.Name) seed=$Seed" | Tee-Object $DriverLog -Append
      & $Python (Join-Path $Code "training\run_pcbilstm_baselines_suite.py") --data-root $GeometryDataRoot --artifacts-root $BaselineRoot --models single spatial --device cuda --device-ids $DeviceIds --batch-size 8 --grad-accum-steps 16 --num-workers 4 --max-epochs 120 --min-epochs 40 --patience 25 --num-receivers $Geometry.Count --seed $Seed 2>&1 | Tee-Object (Join-Path $Logs "$($Geometry.Name)_seed_$($Seed)_baselines.log")
      if ($LASTEXITCODE -ne 0) { throw "Baselines failed for $($Geometry.Name) seed $Seed" }
      "[$(Get-Date -Format s)] DONE baselines $($Geometry.Name) seed=$Seed" | Tee-Object $DriverLog -Append
    }
    if (!(Test-Path -LiteralPath (Join-Path $FinalRoot "suite_summary.json"))) {
      "[$(Get-Date -Format s)] START final $($Geometry.Name) seed=$Seed" | Tee-Object $DriverLog -Append
      & $Python (Join-Path $Code "training\run_st_difftem_suite.py") --data-root $GeometryDataRoot --baseline-root $BaselineRoot --artifacts-root $FinalRoot --variants hybrid_base --skip-spatial-enhanced --device cuda --device-ids $DeviceIds --batch-size 64 --grad-accum-steps 2 --cache-batch-size 8 --num-workers 4 --max-epochs 120 --min-epochs 40 --patience 25 --num-receivers $Geometry.Count --relative-loss-mode global_energy --rho-max 0.75 --no-residual-scale-by-w-single --seed $Seed 2>&1 | Tee-Object (Join-Path $Logs "$($Geometry.Name)_seed_$($Seed)_final.log")
      if ($LASTEXITCODE -ne 0) { throw "Final model failed for $($Geometry.Name) seed $Seed" }
      "[$(Get-Date -Format s)] DONE final $($Geometry.Name) seed=$Seed" | Tee-Object $DriverLog -Append
    }
  }
}
