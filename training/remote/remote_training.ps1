param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('build-subset','sync-project','sync-subset','setup','smoke','launch','status','logs','stop','fetch')]
  [string]$Action,

  [Parameter(Mandatory = $true)]
  [string]$Remote,

  [string]$ProjectRoot = 'G:\Dev\Ortho4XP-SegFormer',
  [string]$RemoteProject = '~/Ortho4XP-SegFormer',
  [string]$RemoteDataRoot = '~/model_training',
  [string]$SubsetDir = 'H:\model_training\remote_subsets\nobara_80gb',
  [string]$WindowsTilesRoot = 'H:\XP12 Addons\Applications\Ortho4XP\Tiles',
  [string]$WindowsTrainingRoot = 'H:\model_training',
  [string]$LocalCheckpointDir = 'H:\model_training\remote_checkpoints\yolo_obb_height_nobara_v1\weights',
  [string]$BaseModel = '~/model_training/runs/yolo_obb_height_v1/weights/last.pt',
  [string]$SessionName = 'yolo_obb_height_nobara_v1',
  [int]$Batch = 4,
  [double]$CudaMemoryFraction = 0.85,
  [int]$Epochs = 20,
  [switch]$UseTmux,
  [switch]$UseWslRsync
)

$ErrorActionPreference = 'Stop'

function Convert-ToRsyncPath {
  param([string]$Path)
  if (-not $UseWslRsync) {
    return $Path
  }
  if ($Path -match '^([A-Za-z]):\\(.*)$') {
    $drive = $matches[1].ToLowerInvariant()
    $rest = $matches[2] -replace '\\','/'
    return "/mnt/$drive/$rest"
  }
  return ($Path -replace '\\','/')
}

function Invoke-Checked {
  param(
    [string]$FilePath,
    [string[]]$Arguments
  )
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$FilePath failed with exit code $LASTEXITCODE"
  }
}

function Invoke-Rsync {
  param([string[]]$Arguments)
  if ($UseWslRsync) {
    $quoted = foreach ($arg in $Arguments) {
      "'" + ($arg -replace "'", "'\\''") + "'"
    }
    Invoke-Checked -FilePath 'wsl' -Arguments @('sh', '-lc', ('rsync ' + ($quoted -join ' ')))
  }
  else {
    Invoke-Checked -FilePath 'rsync' -Arguments $Arguments
  }
}

function Invoke-RemoteScript {
  param([string]$Script)
  $payload = $Script -replace "`r`n", "`n"
  $payload = $payload -replace "`r", "`n"
  $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($payload))
  & ssh $Remote "printf '%s' '$encoded' | base64 -d | bash"
  if ($LASTEXITCODE -ne 0) {
    throw "ssh remote script failed with exit code $LASTEXITCODE"
  }
}

function Get-RemoteHome {
  $remoteHomeValue = (& ssh $Remote 'printf %s "$HOME"')
  if ($LASTEXITCODE -ne 0 -or -not $remoteHomeValue) {
    throw "Could not determine remote HOME for $Remote"
  }
  return [string]$remoteHomeValue
}

function New-RemoteTrainingScript {
  param(
    [string]$RunName,
    [int]$MaxTrain,
    [int]$MaxVal,
    [int]$EpochCount
  )
  $template = @'
set -euo pipefail
PROJECT_DIR="__REMOTE_PROJECT__"
PROJECT_DIR="${PROJECT_DIR/#\~/$HOME}"
DATA_ROOT="__REMOTE_DATA_ROOT__"
DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
VENV_DIR="$PROJECT_DIR/.venv-nobara-training"
SESSION="__SESSION_NAME__"
RUN_DIR="$DATA_ROOT/runs/__RUN_NAME__"
MANIFEST="$DATA_ROOT/manifests/dataset.nobara_subset.jsonl"
MODEL="__BASE_MODEL__"
MODEL="${MODEL/#\~/$HOME}"
mkdir -p "$RUN_DIR" "$RUN_DIR/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
CMD="cd \"$PROJECT_DIR\" && source \"$VENV_DIR/bin/activate\" && python training/train_yolo_obb_stream.py --manifest \"$MANIFEST\" --model \"$MODEL\" --run-dir \"$RUN_DIR\" --epochs __EPOCHS__ --imgsz 512 --batch __BATCH__ --workers 2 --device 0 --max-train __MAX_TRAIN__ --max-val __MAX_VAL__ --max-instances 1000 --height-labels bins --checkpoint-interval 1000 --progress-interval 25 --best-step-min-delta 0.0001 --cuda-empty-cache-interval 25 --cuda-memory-fraction __CUDA_MEMORY_FRACTION__ --resume"
if [ "__USE_TMUX__" = "true" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux requested but not installed" >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION" "$CMD > \"$RUN_DIR/stdout_remote.log\" 2> \"$RUN_DIR/stderr_remote.log\""
  echo "started tmux session $SESSION"
else
  nohup bash -lc "$CMD" > "$RUN_DIR/stdout_remote.log" 2> "$RUN_DIR/stderr_remote.log" &
  echo $! > "$RUN_DIR/pid"
  echo "started pid $(cat "$RUN_DIR/pid")"
fi
echo "$CMD" > "$RUN_DIR/launch_command.txt"
'@
  $result = $template
  $result = $result.Replace('__REMOTE_PROJECT__', $RemoteProject)
  $result = $result.Replace('__REMOTE_DATA_ROOT__', $RemoteDataRoot)
  $result = $result.Replace('__BASE_MODEL__', $BaseModel)
  $result = $result.Replace('__SESSION_NAME__', $SessionName)
  $result = $result.Replace('__RUN_NAME__', $RunName)
  $result = $result.Replace('__EPOCHS__', [string]$EpochCount)
  $result = $result.Replace('__BATCH__', [string]$Batch)
  $result = $result.Replace('__MAX_TRAIN__', [string]$MaxTrain)
  $result = $result.Replace('__MAX_VAL__', [string]$MaxVal)
  $result = $result.Replace('__CUDA_MEMORY_FRACTION__', [string]$CudaMemoryFraction)
  $result = $result.Replace('__USE_TMUX__', ($(if ($UseTmux) { 'true' } else { 'false' })))
  return $result
}

switch ($Action) {
  'build-subset' {
    $script = Join-Path $ProjectRoot 'training\remote\build_nobara_subset.py'
    $remoteHome = Get-RemoteHome
    $remoteDataRootForManifest = "$remoteHome/model_training"
    Invoke-Checked -FilePath 'python' -Arguments @(
      $script,
      '--manifest', (Join-Path $WindowsTrainingRoot 'manifests\dataset.jsonl'),
      '--output-dir', $SubsetDir,
      '--budget-gb', '80',
      '--windows-tiles-root', $WindowsTilesRoot,
      '--windows-training-root', $WindowsTrainingRoot,
      '--remote-tiles-root', "$remoteDataRootForManifest/tiles",
      '--remote-training-root', $remoteDataRootForManifest
    )
  }
  'sync-project' {
    $script = @'
set -euo pipefail
PROJECT_DIR="__REMOTE_PROJECT__"
PROJECT_DIR="${PROJECT_DIR/#\~/$HOME}"
mkdir -p "$PROJECT_DIR"
'@
    $script = $script.Replace('__REMOTE_PROJECT__', $RemoteProject)
    Invoke-RemoteScript $script
    $source = Convert-ToRsyncPath ((Resolve-Path -LiteralPath $ProjectRoot).Path)
    Invoke-Rsync @(
      '-azP',
      '--exclude', '.git/',
      '--exclude', '.venv/',
      '--exclude', '.venv-nobara-training/',
      '--exclude', 'tmp/',
      '--exclude', 'build/',
      '--exclude', 'dist/',
      '--exclude', 'dist_build/',
      "$source/",
      "${Remote}:$RemoteProject/"
    )
  }
  'sync-subset' {
    $tilesList = Convert-ToRsyncPath (Join-Path $SubsetDir 'tiles_files_from.txt')
    $trainingList = Convert-ToRsyncPath (Join-Path $SubsetDir 'training_files_from.txt')
    $subsetManifest = Convert-ToRsyncPath (Join-Path $SubsetDir 'dataset.nobara_subset.jsonl')
    $seedCheckpoint = Convert-ToRsyncPath (Join-Path $WindowsTrainingRoot 'runs\yolo_obb_height_v1\weights\last.pt')
    $script = @'
set -euo pipefail
DATA_ROOT="__REMOTE_DATA_ROOT__"
DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
mkdir -p "$DATA_ROOT/tiles" "$DATA_ROOT/manifests" "$DATA_ROOT/runs/yolo_obb_height_v1/weights"
'@
    $script = $script.Replace('__REMOTE_DATA_ROOT__', $RemoteDataRoot)
    Invoke-RemoteScript $script
    Invoke-Rsync @('-azP','--files-from', $tilesList, ((Convert-ToRsyncPath $WindowsTilesRoot) + '/'), "${Remote}:$RemoteDataRoot/tiles/")
    Invoke-Rsync @('-azP','--files-from', $trainingList, ((Convert-ToRsyncPath $WindowsTrainingRoot) + '/'), "${Remote}:$RemoteDataRoot/")
    Invoke-Rsync @('-azP', $subsetManifest, "${Remote}:$RemoteDataRoot/manifests/dataset.nobara_subset.jsonl")
    Invoke-Rsync @('-azP', $seedCheckpoint, "${Remote}:$RemoteDataRoot/runs/yolo_obb_height_v1/weights/last.pt")
  }
  'setup' {
    $script = @'
set -euo pipefail
PROJECT_DIR="__REMOTE_PROJECT__"
PROJECT_DIR="${PROJECT_DIR/#\~/$HOME}"
cd "$PROJECT_DIR"
bash training/remote/setup_nobara_training.sh
'@
    $script = $script.Replace('__REMOTE_PROJECT__', $RemoteProject)
    Invoke-RemoteScript $script
  }
  'smoke' {
    Invoke-RemoteScript (New-RemoteTrainingScript -RunName "${SessionName}_smoke" -MaxTrain 128 -MaxVal 32 -EpochCount 1)
  }
  'launch' {
    Invoke-RemoteScript (New-RemoteTrainingScript -RunName $SessionName -MaxTrain 0 -MaxVal 0 -EpochCount $Epochs)
  }
  'status' {
    $script = @'
set -euo pipefail
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
tmux ls 2>/dev/null || true
DATA_ROOT="__REMOTE_DATA_ROOT__"
DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
if [ -f "$DATA_ROOT/runs/__SESSION_NAME__/pid" ]; then
  PID="$(cat "$DATA_ROOT/runs/__SESSION_NAME__/pid")"
  if kill -0 "$PID" 2>/dev/null; then echo "pid $PID running"; else echo "pid $PID not running"; fi
fi
tail -n 5 "$DATA_ROOT/runs/__SESSION_NAME__/yolo_progress.jsonl" 2>/dev/null || true
'@
    $script = $script.Replace('__REMOTE_DATA_ROOT__', $RemoteDataRoot)
    $script = $script.Replace('__SESSION_NAME__', $SessionName)
    Invoke-RemoteScript $script
  }
  'logs' {
    $script = @'
set -euo pipefail
DATA_ROOT="__REMOTE_DATA_ROOT__"
DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
echo '--- stdout ---'
tail -n 40 "$DATA_ROOT/runs/__SESSION_NAME__/stdout_remote.log" 2>/dev/null || true
echo '--- stderr ---'
tail -n 40 "$DATA_ROOT/runs/__SESSION_NAME__/stderr_remote.log" 2>/dev/null || true
'@
    $script = $script.Replace('__REMOTE_DATA_ROOT__', $RemoteDataRoot)
    $script = $script.Replace('__SESSION_NAME__', $SessionName)
    Invoke-RemoteScript $script
  }
  'stop' {
    $script = @'
set -euo pipefail
DATA_ROOT="__REMOTE_DATA_ROOT__"
DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
if tmux has-session -t "__SESSION_NAME__" 2>/dev/null; then
  tmux send-keys -t "__SESSION_NAME__" C-c
elif [ -f "$DATA_ROOT/runs/__SESSION_NAME__/pid" ]; then
  kill "$(cat "$DATA_ROOT/runs/__SESSION_NAME__/pid")"
else
  echo "No tmux session or pid file found for __SESSION_NAME__" >&2
  exit 1
fi
'@
    $script = $script.Replace('__REMOTE_DATA_ROOT__', $RemoteDataRoot)
    $script = $script.Replace('__SESSION_NAME__', $SessionName)
    Invoke-RemoteScript $script
  }
  'fetch' {
    New-Item -ItemType Directory -Force -Path $LocalCheckpointDir | Out-Null
    Invoke-Rsync @('-azP', "${Remote}:$RemoteDataRoot/runs/$SessionName/weights/*.pt", (Convert-ToRsyncPath $LocalCheckpointDir) + '/')
  }
}
