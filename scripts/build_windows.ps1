Param(
  [string]$ProjectRoot = (Resolve-Path ".").Path
)

$ErrorActionPreference = "Stop"

Write-Host "== commentAI Windows build =="
Write-Host "ProjectRoot: $ProjectRoot"

Set-Location $ProjectRoot

# 1) venv
if (-Not (Test-Path ".venv")) {
  python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

# 2) build
pyinstaller commentAI-test.spec

# 3) assemble release folder
$releaseRoot = Join-Path $ProjectRoot "release"
$demoDir = Join-Path $releaseRoot "commentAI-demo"

if (Test-Path $demoDir) {
  Remove-Item $demoDir -Recurse -Force
}
New-Item -ItemType Directory -Path $demoDir | Out-Null

# PyInstaller может собрать бинарь как:
# - onefile: dist\commentAI-test.exe
# - onedir:  dist\commentAI-test\commentAI-test.exe
$exeCandidates = @(
  (Join-Path $ProjectRoot "dist\commentAI-test\commentAI-test.exe"),
  (Join-Path $ProjectRoot "dist\commentAI-test.exe")
)

$exePath = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-Not $exePath) {
  Write-Host "Contents of dist/:"
  Get-ChildItem -Path (Join-Path $ProjectRoot "dist") -Recurse -ErrorAction SilentlyContinue | Select-Object FullName | Format-Table -AutoSize
  throw "Built exe not found in dist/. Expected one of: $($exeCandidates -join ', ')"
}

Copy-Item -Path $exePath -Destination $demoDir

# data: copy minimal required files
New-Item -ItemType Directory -Path (Join-Path $demoDir "data") | Out-Null
Copy-Item -Path (Join-Path $ProjectRoot "data\reviews_labeled.csv") -Destination (Join-Path $demoDir "data")
if (Test-Path (Join-Path $ProjectRoot "data\hard_cases_labeled.csv")) {
  Copy-Item -Path (Join-Path $ProjectRoot "data\hard_cases_labeled.csv") -Destination (Join-Path $demoDir "data")
}
# optional: if you want to ship current feedback/status
if (Test-Path (Join-Path $ProjectRoot "data\feedback_buffer.jsonl")) {
  Copy-Item -Path (Join-Path $ProjectRoot "data\feedback_buffer.jsonl") -Destination (Join-Path $demoDir "data")
}
if (Test-Path (Join-Path $ProjectRoot "data\retrain_status.json")) {
  Copy-Item -Path (Join-Path $ProjectRoot "data\retrain_status.json") -Destination (Join-Path $demoDir "data")
}

# models: ship production model; versions folder will be created automatically
New-Item -ItemType Directory -Path (Join-Path $demoDir "models") | Out-Null
Copy-Item -Path (Join-Path $ProjectRoot "models\production") -Destination (Join-Path $demoDir "models") -Recurse
New-Item -ItemType Directory -Path (Join-Path $demoDir "models\versions") | Out-Null

# docs
Copy-Item -Path (Join-Path $ProjectRoot "README.md") -Destination $demoDir
if (Test-Path (Join-Path $ProjectRoot "INSTRUCTIONS_FOR_TEACHER.txt")) {
  Copy-Item -Path (Join-Path $ProjectRoot "INSTRUCTIONS_FOR_TEACHER.txt") -Destination $demoDir
}

# 4) zip
$zipPath = Join-Path $releaseRoot "commentAI-demo-windows.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path $demoDir -DestinationPath $zipPath

Write-Host "Done. Release folder: $demoDir"
Write-Host "ZIP: $zipPath"