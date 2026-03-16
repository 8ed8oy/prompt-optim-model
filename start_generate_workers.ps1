param(
    [int]$WorkerCount = 4,
    [int]$TargetSizePerWorker = 300,
    [string]$OutputDir = ".\data",
    [double]$Temperature = 0.9,
    [int]$MaxRetries = 6,
    [double]$SleepSeconds = 0.8,
    [string]$PythonExe = "D:/CondaData/envs/prompt-opt/python.exe",
    [string]$ModelName = $(if ($env:MODEL_NAME) { $env:MODEL_NAME } else { "deepseek-chat" }),
    [string]$BaseUrl = $(if ($env:BASE_URL) { $env:BASE_URL } else { "https://api.deepseek.com/v1" }),
    [string]$ApiKeys = $env:API_KEYS
)

$ErrorActionPreference = "Stop"

function Split-SecretList {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return @()
    }

    return @(
        $Text -split "[;,`r`n]+" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
    )
}

function Quote-Single {
    param([string]$Value)

    return "'" + ($Value -replace "'", "''") + "'"
}

$workspaceRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$generateScript = Join-Path $workspaceRoot "generate_data.py"
if (-not (Test-Path $generateScript)) {
    throw "generate_data.py not found: $generateScript"
}

if ($WorkerCount -lt 1) {
    throw "WorkerCount must be >= 1"
}

if ($TargetSizePerWorker -lt 1) {
    throw "TargetSizePerWorker must be >= 1"
}

$outputDirFull = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $workspaceRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $outputDirFull | Out-Null

$apiKeyList = Split-SecretList -Text $ApiKeys
if ($apiKeyList.Count -eq 0) {
    if ([string]::IsNullOrWhiteSpace($env:API_KEY)) {
            throw "API_KEY not found. Set `$env:API_KEY or pass -ApiKeys."
    }
    $apiKeyList = @($env:API_KEY)
}

if ($apiKeyList.Count -ne 1 -and $apiKeyList.Count -ne $WorkerCount) {
    throw "ApiKeys count must be 1 or equal to WorkerCount. Got $($apiKeyList.Count) keys, WorkerCount=$WorkerCount"
}

Write-Host "[INFO] WorkspaceRoot : $workspaceRoot"
Write-Host "[INFO] OutputDir     : $outputDirFull"
Write-Host "[INFO] WorkerCount   : $WorkerCount"
Write-Host "[INFO] PerWorker     : $TargetSizePerWorker"
Write-Host "[INFO] TotalExpected : $($WorkerCount * $TargetSizePerWorker) (before dedup)"

for ($i = 1; $i -le $WorkerCount; $i++) {
    $workerId = "w{0:d2}" -f $i
    $apiKey = if ($apiKeyList.Count -eq 1) { $apiKeyList[0] } else { $apiKeyList[$i - 1] }

    $commandParts = @()
    $commandParts += ('$env:API_KEY = ' + (Quote-Single $apiKey))

    if (-not [string]::IsNullOrWhiteSpace($BaseUrl)) {
        $commandParts += ('$env:BASE_URL = ' + (Quote-Single $BaseUrl))
    }

    if (-not [string]::IsNullOrWhiteSpace($ModelName)) {
        $commandParts += ('$env:MODEL_NAME = ' + (Quote-Single $ModelName))
    }

    $commandParts += ('Set-Location ' + (Quote-Single $workspaceRoot))
    $commandParts += (
        ('& {0} .\generate_data.py --target-size {1} --output {2} --worker-id {3} --temperature {4} --max-retries {5} --sleep {6}' -f
            (Quote-Single $PythonExe),
            $TargetSizePerWorker,
            (Quote-Single $outputDirFull),
            $workerId,
            $Temperature.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            $MaxRetries,
            $SleepSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        )
    )

    $commandText = $commandParts -join '; '

    Start-Process -FilePath "powershell.exe" `
        -WorkingDirectory $workspaceRoot `
        -ArgumentList @(
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-Command", $commandText
        ) | Out-Null

    Write-Host "[START] worker=$workerId -> $outputDirFull\train_data.$workerId.jsonl"
}

Write-Host "[DONE] All workers started."
Write-Host "[NEXT] After all workers finish, run:"
Write-Host "       python .\merge_clean_data.py --input-dir .\data --output .\train_data.cleaned.jsonl"
